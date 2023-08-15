"""
This script contains adapters for fast adaptation of
FLAIR modelo to downstream tasks/domains.

In particular, these adapters work over the vision and text
embeddings. Also, this code contains a Wrapper for zero-shot
classification

Implemented adapters:
Zero-shot, Linear Probe (LP), ClipAdapter, TipAdapter, TipAdapter-f
"""

import copy
import random
import torch
import numpy as np

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from flair.pretraining.data.transforms import augmentations_pretraining

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
The first section contains only-vision adapters (i.e. linear probing)
"""


class AdapterWrapper(object):
    def __init__(self, model, targets, tta=False, fta=False):
        # Set model and number of targets
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.num_targets = len(targets)
        # Augmentation for training and for test-time augmentation
        self.tta = tta
        self.fta = fta
        self.number_augmentations = 20

    def extract_vision_features(self, data_loader, transforms=None):
        self.model.eval()

        epoch_iterator = tqdm(
            data_loader, desc="Extracting features (X / X Steps)", dynamic_ncols=True
        )

        X, Y = [], []
        for step, batch in enumerate(epoch_iterator):
            images = batch["image"].to(device).to(torch.float32)

            with torch.no_grad():

                # Image augmentation
                if transforms is not None:
                    images = transforms(images)

                # Forward vision encoder
                x = self.model.vision_model(images)

            X.extend(x.cpu().detach().numpy())
            Y.extend(batch["label"].numpy())

        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    def fit(self, loaders, transforms=None):
        data_loader = loaders["train"]

        if self.fta:
            transforms = augmentations_pretraining

        # Extract features and labels from generator
        if self.fta and transforms is not None:
            X, Y = [], []
            for i in range(self.number_augmentations):
                Xa, Ya = self.extract_vision_features(data_loader, transforms=transforms)
                X.append(Xa), Y.append(Ya)
            X = np.concatenate(X, 0)
            Y = np.concatenate(Y, 0)
        else:
            X, Y = self.extract_vision_features(data_loader, transforms=transforms)

        # Perform logistic regression
        self.train(X, Y)

    def train(self, X, Y):
        """
        Placeholder: function to be developed in a concrete adapter.
        """
        return

    def predict(self, loader, transforms=None):
        """
        Placeholder: function to be developed in a concrete adapter.
        """
        return


class LinearProbe(AdapterWrapper):
    def __init__(self, model, targets, tta=False, fta=False, c=0.316):
        super().__init__(model, targets, tta=tta, fta=fta)
        self.classifier = LogisticRegression(random_state=0, C=c, max_iter=1000, verbose=0,
                                             class_weight="balanced")

    def train(self, X, Y):

        # Train classifier
        self.classifier.fit(X, Y)

        # Set Linear Probe classifier into FLAIR model
        self.model.classifier = torch.nn.Linear(X.shape[-1], self.num_targets, bias=True)
        self.model.classifier.weight = torch.nn.Parameter(torch.tensor(self.classifier.coef_).to(torch.float32))
        self.model.classifier.bias = torch.nn.Parameter(torch.tensor(self.classifier.intercept_).to(torch.float32))
        self.model.classifier.to(device)

    def predict(self, loader, transforms=None):
        self.model.eval()

        # Set transforms on test-time augmentation
        if self.tta:
            transforms = augmentations_pretraining

        epoch_iterator = tqdm(
            loader, desc="Predicting (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        with torch.no_grad():
            refs, preds = [], []
            for step, batch in enumerate(epoch_iterator):
                images = batch["image"].to(device).to(torch.float32)
                Y = batch["label"].to(device).to(torch.long)

                # Forward
                if self.tta:
                    preds_tta = []
                    for i in range(self.number_augmentations):
                        x = self.model.vision_model(transforms(images))
                        score = self.model.classifier(x)
                        preds_tta.append(score.unsqueeze(-1))
                    score = torch.concat(preds_tta, -1).mean(-1)
                else:
                    x = self.model.vision_model(images)
                    score = self.model.classifier(x)
                # Activation for prediction
                if score.shape[-1] == 1:  # Binary case
                    score = torch.sigmoid(score)
                    score = torch.concat([1 - score, score], -1)
                else:  # Multi-class case
                    score = torch.softmax(score, -1)
                torch.cuda.empty_cache()

                refs.append(Y.cpu().detach().numpy())
                preds.append(score.cpu().detach().numpy())

        refs = np.concatenate(refs, 0)
        preds = np.concatenate(preds, 0)
        return refs, preds


"""
This section contains multimodal (vision-language) adapters.
"""


class LanguageAdapterWrapper(AdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, tta=tta, fta=fta)

        # Compute text prototypes
        self.text_embeds_dict, self.text_embeds = model.compute_text_embeddings(list(targets.keys()),
                                                                                domain_knowledge=domain_knowledge)


class ZeroShot(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

    def fit(self, loaders, transforms=None):
        """
        No training in zero-shot prediction
        """
        return

    def predict(self, loader, transforms=None):

        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i = torch.matmul(torch.tensor(X), self.text_embeds.t()) * self.model.logit_scale.exp()
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                score = torch.matmul(X, self.text_embeds.t().to(device)) * self.model.logit_scale.exp()

        # Softmax probs from scores
        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()
        return refs, preds


class ClipAdapter(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        self.c_in = self.model.vision_model.out_dim
        self.reduction = 4
        self.ratio = 0.2

        # Set adapter
        self.adapter = torch.nn.Sequential(torch.nn.Linear(self.c_in, self.c_in // self.reduction, bias=False),
                                           torch.nn.ReLU(inplace=True),
                                           torch.nn.Linear(self.c_in // self.reduction, self.c_in, bias=False),
                                           torch.nn.ReLU(inplace=True)).to(device)

    def predict(self, loader, transforms=None):

        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    # Compute residual CLIP-Adapter
                    X = self.residual_adapter(X)
                    # Compute similarity
                    score_i = torch.matmul(torch.tensor(X), self.text_embeds.t()) * self.model.logit_scale.exp()
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                # Compute residual CLIP-Adapter
                X = self.residual_adapter(X)
                # Compute similarity
                score = torch.matmul(torch.tensor(X), self.text_embeds.t()) * self.model.logit_scale.exp()

        # Softmax probs from scores
        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()

        return refs, preds

    def train(self, X, Y):
        X = torch.tensor(X)
        Y = torch.tensor(Y)

        # TRAINING
        epochs, lr, bs = 40, 0.001, 1

        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * X.shape[0])

        indexes = np.arange(0, X.shape[0])
        random.shuffle(indexes)
        for i_epoch in range(epochs):
            loss_epoch = 0.0
            for i_sample in range(X.shape[0]):
                X_batch = X[indexes[i_sample], :].unsqueeze(0).to(device)
                target = Y[indexes[i_sample]].unsqueeze(0).to(device)

                # Compute residual CLIP-Adapter
                X_batch = self.residual_adapter(X_batch)

                # Compute logits
                logits = self.model.logit_scale.exp() * X_batch @ self.text_embeds.t().to(device)

                # Compute loss
                loss = torch.nn.functional.cross_entropy(logits, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_epoch += loss.item()/X.shape[0]

            print('loss=%2.5f' % loss_epoch, end="\n")

    def residual_adapter(self, X):
        # Compute residual CLIP-Adapter
        X_res = self.adapter(X)
        X = self.ratio * X_res + (1 - self.ratio) * X
        X = X / X.norm(dim=-1, keepdim=True)
        return X


class TipAdapter(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False, train=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        self.beta = 5
        self.alpha = 1
        self.train_tip = train

        # Init cache values
        self.cache_keys = []
        self.cache_values = []
        self.adapter_layer = []

    def predict(self, loader, transforms=None):

        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i = self.adapter(X)
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                score = self.adapter(X)

        # Softmax probs from scores
        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()

        return refs, preds

    def train(self, X, Y):
        X = torch.tensor(X)
        Y = torch.tensor(Y)

        self.cache_keys = torch.transpose(X, 1, 0).to(torch.float32).to(device)
        self.cache_values = torch.nn.functional.one_hot(Y).to(torch.float32).to(device)

        if self.train_tip:

            # TRAINING
            epochs, lr, bs = 40, 0.001, 1

            # Enable the cached keys to be learnable
            adapter_layer = torch.nn.Linear(self.cache_keys.shape[0], self.cache_keys.shape[1], bias=False).to(device)
            adapter_layer.weight = torch.nn.Parameter(self.cache_keys.t())
            adapter_layer = adapter_layer.to(device)

            optimizer = torch.optim.AdamW(adapter_layer.parameters(), lr=lr, eps=1e-4)

            indexes = np.arange(0, self.cache_keys.shape[1])
            random.shuffle(indexes)
            for i_epoch in range(epochs):
                loss_epoch = 0.0
                for i_sample in range(self.cache_keys.shape[1]):
                    image = self.cache_keys[:, indexes[i_sample]].unsqueeze(0).to(device)
                    target = self.cache_values[indexes[i_sample], :].argmax().unsqueeze(0).to(device)

                    # Zero-shot CLIP
                    clip_logits = self.model.logit_scale.exp() * (image @ self.text_embeds.t())

                    # Tip-Adapter
                    affinity = adapter_layer(image)
                    cache_logits = torch.exp(((-1) * (self.beta - self.beta * affinity))) @ self.cache_values
                    cache_logits /= X.shape[0]
                    cache_logits *= self.model.logit_scale.exp()

                    tip_logits = clip_logits + cache_logits * self.alpha

                    loss = torch.nn.functional.cross_entropy(tip_logits, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_epoch += loss.item()/self.cache_keys.shape[1]

                print('loss=%2.5f' % loss_epoch, end="\n")

            # Storage trained adapter
            self.adapter_layer = adapter_layer

    def adapter(self, X):
        # Zero-shot CLIP
        clip_logits = 100 * (X @ self.text_embeds.t().to(device))

        # Tip-Adapter
        if not self.train_tip:
            affinity = X @ self.cache_keys
        else:
            affinity = self.adapter_layer(X)

        cache_logits = torch.exp(((-1) * (self.beta - self.beta * affinity))) @ self.cache_values
        logits = clip_logits + cache_logits * self.alpha

        return logits