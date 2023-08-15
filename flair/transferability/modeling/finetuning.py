"""
This script contains methods for Fine-tuning the whole or
partially FLAIR vision encoder to downstream tasks/domains.
Tasks: classification, segmentation.
"""

import copy
import torch
import numpy as np

from tqdm import tqdm
from torch.cuda.amp import autocast

from flair.utils.metrics import classification_metrics, segmentation_metrics
from flair.utils.losses import BinaryDice, BinaryDiceCE
from flair.transferability.modeling.adapters import LinearProbe
from flair.transferability.modeling.ResnetUNet import ResnetUNet
from flair.pretraining.data.transforms import augmentations_pretraining
from flair.transferability.data.transforms import AugmentationsSegmentation

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FineTuning(object):
    def __init__(self, model, targets, method, tta=False, fta=False, loaders=None, epochs=20, update_bn=True,
                 freeze_classifier=False, last_lp=False, lr=1e-4, task="classification", save_best=False,
                 patience=50):
        super().__init__()
        # Required inputs
        self.method = method
        self.model = copy.deepcopy(model)
        self.model.text_model.to("cpu")
        self.num_targets = len(targets)
        self.targets = targets
        self.c_in = self.model.vision_model.out_dim
        self.task = task

        # Training hyperparams
        self.tta = tta
        self.fta = fta
        self.epochs = epochs
        self.lr = lr
        self.update_bn = update_bn
        self.freeze_classifier = freeze_classifier
        self.last_lp = last_lp
        self.save_best, self.best_state_dict, self.patience, self.counter = save_best, [], patience, 0

        # Settings for grad scaler in mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
        self.max_grad_norm = 1

        # Training epoch counter for training
        self.i_epoch = 0

        # Set particularities between classification/segmentation
        if self.task == "classification":
            self.act = torch.nn.Softmax(dim=1)
            self.loss = torch.nn.functional.cross_entropy
            self.target_id = "label"
            self.metrics_fnc = classification_metrics
            self.criteria = "f1_avg"
            self.transforms = augmentations_pretraining
        elif self.task == "segmentation":
            self.act = torch.nn.Sigmoid()
            # self.loss = BinaryDice(activation=self.act)
            self.loss = BinaryDiceCE(activation=self.act)
            self.target_id = "mask"
            self.metrics_fnc = segmentation_metrics
            self.criteria = "auprc"  # "dsc", "auprc"
            self.transforms = AugmentationsSegmentation().to(device)
            self.model.vision_model = ResnetUNet(pretrained_encoder=self.model.vision_model,
                                                 update_bn=self.update_bn).to(device)

        # Freeze weights
        freeze_weights(self.model, self.method, self.model.vision_type)

        # Set classifier
        if self.task == "classification":
            if "LP" in method and loaders is not None:
                # Set LP adapter
                adapter = LinearProbe(model, targets, tta=False, fta=False)
                # Fit adapter
                adapter.fit(loaders)
                # Init backbone classifier with LP output
                self.model.classifier = adapter.model.classifier

                if adapter.model.classifier.weight.shape[0]:
                    self.act = torch.nn.Sigmoid()
                    self.loss = torch.nn.functional.binary_cross_entropy_with_logits
            else:
                self.model.classifier = torch.nn.Linear(self.c_in, len(targets), bias=True).to(device)
        elif self.task == "segmentation":
            self.model.classifier = torch.nn.Conv2d(64, 1, 1, bias=True).to(device)

        if self.freeze_classifier:
            for name, param in self.model.classifier.named_parameters():
                param.requires_grad = False

        # Set optimizer for training
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, eps=1e-4)

    def fit(self, loaders):

        # Train for each epoch
        best_val_metric = 0.0
        for i_epoch in range(self.epochs):
            self.i_epoch = i_epoch + 1

            # Train epoch
            self.train_epoch(loaders["train"])
            torch.cuda.empty_cache()

            # Validate
            if loaders["val"] is not None:
                # Validation predictions
                refs, preds = self.predict(loaders["val"])
                torch.cuda.empty_cache()

                # Validation metrics
                metrics = self.metrics_fnc(refs, preds)
                print('VALIDATION - Epoch=%d: metric=%2.4f' % (self.i_epoch, metrics[self.criteria]))

                # Save best model weights if save_best flag allowed
                if self.save_best:
                    # Check metric evolution on validation subset
                    if metrics[self.criteria] > best_val_metric:
                        print("Best " + self.criteria + " in validation improved!")
                        best_val_metric = metrics[self.criteria]
                        self.best_state_dict = copy.deepcopy(self.model.state_dict())
                        self.counter = 0
                    else:
                        self.counter += 1
                    # Keep training only for few epochs if no improvement is observed
                    if self.counter >= self.patience:
                        print("No improvement... early stopping!")
                        self.model.load_state_dict(self.best_state_dict)
                        return

        if self.save_best:
            self.model.load_state_dict(self.best_state_dict)

        if self.last_lp:
            # Set LP adapter
            adapter = LinearProbe(self.model, self.targets, tta=False, fta=False)
            # Fit adapter
            adapter.fit(loaders["train"])
            # Init backbone classifier with LP output
            self.model.classifier = adapter.model.classifier

    def predict(self, loader):
        self.model.eval()
        loss_ave = 0

        epoch_iterator = tqdm(
            loader, desc="Prediction (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        with torch.no_grad():
            with autocast():
                refs, preds = [], []
                for step, batch in enumerate(epoch_iterator):
                    images = batch["image"].to(device).to(torch.float32)
                    Y = batch[self.target_id].to(device).to(torch.long)

                    # Forward
                    x = self.model.vision_model(images)
                    logits = self.model.classifier(x)

                    # Get loss
                    if logits.shape[-1] == 1:  # Fix for binary case and bce with logits
                        Y = Y.unsqueeze(-1).to(torch.float)
                    loss = self.loss(logits, Y)

                    # Get prediction
                    score = self.act(logits)
                    # Activation for prediction
                    if score.shape[-1] == 1:  # Binary case
                        score = torch.concat([1 - score, score], -1)

                    # Overall losses track
                    loss_ave += loss.item()
                    torch.cuda.empty_cache()

                    refs.append(Y.cpu().detach().numpy().astype(np.int32))
                    preds.append(score.cpu().detach().numpy())
            torch.cuda.empty_cache()

        refs = np.concatenate(refs, 0)
        preds = np.concatenate(preds, 0)
        return refs, preds

    def train_epoch(self, data_loader):

        if self.task == "classification":
            if self.update_bn:
                self.model.train()
            else:
                self.model.eval()
        else:   # For segmentation, bn update is regulated inside ResnetUNet
            self.model.train()

        epoch_iterator = tqdm(
            data_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        loss_ave = 0.0
        for step, batch in enumerate(epoch_iterator):
            images = batch["image"].to(device).to(torch.float32)
            Y = batch[self.target_id].to(device).to(torch.long)

            with autocast():
                # Image augmentation
                if self.fta:
                    if self.task == "classification":
                        images = self.transforms(images)
                    elif self.task == "segmentation":
                        images, Y = self.transforms(images, Y.to(torch.float32))

                # Forward
                x = self.model.vision_model(images)
                logits = self.model.classifier(x)

                # Get loss
                if logits.shape[-1] == 1:  # Fix for binary case and bce with logits
                    Y = Y.unsqueeze(-1).to(torch.float)
                loss = self.loss(logits, Y)

            # Update model with scaler
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # Overall losses track
            loss_ave += loss.item()
            torch.cuda.empty_cache()

            # Display training track
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps): ce=%2.5f" % (
                self.i_epoch, step + 1, len(data_loader), loss.item())
            )

        # Display epoch-wise loss
        print('TRAINING - Epoch=%d: ave_loss=%2.5f' % (self.i_epoch, loss_ave / len(epoch_iterator)))


def freeze_weights(model, method, architecture):

    if architecture == "resnet":
        last_block_name = "model.layer4.2"
        bn_name = "bn"
    elif architecture == "efficientnet":
        last_block_name = "model.features.7.3."
        bn_name = "block.3.1."
    else:
        print("Architecture not supported for freezing weights. ")
        return

    for name, param in model.named_parameters():
        param.requires_grad = False

        if "freeze_all" not in method:

            if "last" in method:
                if last_block_name in name:
                    if "bn" in method:
                        if bn_name in name:
                            param.requires_grad = True
                    else:
                        param.requires_grad = True
            else:
                if "bn" in method:
                    if bn_name in name:
                        param.requires_grad = True
                else:
                    param.requires_grad = True

    return


