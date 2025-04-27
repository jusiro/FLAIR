
"""
Main function to transfer pretrained FLAIR model
to downstream classification and segmentation tasks.
It includes fine-tuning, linear probing, and vision-language
adapters. Also, it allows to directly testing zero-shot
generalization trough text prompts.
"""

import argparse
import torch

from flair.modeling.model import FLAIRModel
from flair.transferability.data.dataloader import get_dataloader_splits
from flair.utils.metrics import evaluate, average_folds_results, save_results
from flair.modeling.misc import set_seeds
from flair.transferability.modeling.adapters import LinearProbe, ClipAdapter, ZeroShot, TipAdapter
from flair.transferability.modeling.finetuning import FineTuning

from local_data.constants import *
from local_data.experiments import get_experiment_setting

import warnings
warnings.filterwarnings("ignore")

set_seeds(42, use_cuda=torch.cuda.is_available())


def init_adapter(model, args):

    if "FT" in args.method:
        print("Transferability by Fine-tuning...", end="\n")
        adapter = FineTuning(model, args.setting["targets"], args.method, tta=args.tta, fta=args.fta,
                             loaders=args.loaders, epochs=args.epochs, update_bn=args.update_bn,
                             freeze_classifier=args.freeze_classifier, last_lp=args.last_lp, lr=args.lr,
                             task=args.setting["task"], save_best=args.save_best, patience=args.patience)
    elif args.method == "lp":
        print("Transferability by Linear Probing...", end="\n")
        adapter = LinearProbe(model, args.setting["targets"], tta=args.tta, fta=args.fta)
    elif args.method == "clipAdapter":
        print("Transferability by CLIP Adapter...", end="\n")
        adapter = ClipAdapter(model, args.setting["targets"], tta=args.tta, fta=args.fta,
                              domain_knowledge=args.domain_knowledge)
    elif args.method == "tipAdapter":
        print("Transferability by TIP-Adapter Adapter...", end="\n")
        adapter = TipAdapter(model, args.setting["targets"], tta=args.tta, fta=args.fta,
                             domain_knowledge=args.domain_knowledge, train=False)
    elif args.method == "tipAdapter-f":
        print("Transferability by TIP-Adapter-f Adapter...", end="\n")
        adapter = TipAdapter(model, args.setting["targets"], tta=args.tta, fta=args.fta,
                             domain_knowledge=args.domain_knowledge, train=True)
    elif args.method == "zero_shot":
        print("Zero-shot classification...", end="\n")
        adapter = ZeroShot(model, args.setting["targets"], tta=args.tta, fta=args.fta,
                           domain_knowledge=args.domain_knowledge)
    else:
        print("Adapter not implemented... using LP", end="\n")
        adapter = LinearProbe(args, model.vision_model)

    return adapter


def generate_experiment_id(args):
    id = args.experiment + '_vision_' + args.architecture + '_method_' + args.method + '_pretrained_' \
         + str(args.load_weights) + '_shots_train_' + args.shots_train + '_shots_test_' + args.shots_test + \
         '_balance_' + str(args.balance)
    return id


def process(args):

    # KFold cross-validation
    args.metrics_test, args.metrics_external, args.weights = [], [[] for i in range(len(args.experiment_test))], []
    for iFold in range(args.folds):
        print("\nTransferability (fold : " + str(iFold + 1) + ")", end="\n")
        args.iFold = iFold

        # Get specific experiment settings (i.e. dataframe path, classes, tasks, ...)
        args.setting = get_experiment_setting(args.experiment)

        # Init FLAIR model
        model = FLAIRModel(from_checkpoint=args.load_weights, weights_path=args.weights_path,
                           projection=args.project_features, norm_features=args.norm_features,
                           vision_pretrained=args.init_imagenet)

        # Set datasets
        args.loaders = get_dataloader_splits(args.setting["dataframe"], args.data_root_path, args.setting["targets"],
                                             shots_train=args.shots_train, shots_val=args.shots_val,
                                             shots_test=args.shots_test, balance=args.balance,
                                             batch_size=args.batch_size, num_workers=args.num_workers, seed=iFold,
                                             task=args.setting["task"], size=args.size,
                                             resize_canvas=args.resize_canvas, batch_size_test=args.batch_size_test,
                                             crop_foreground=args.crop_foreground)

        # Set adapter
        adapter = init_adapter(model, args)

        # Fit adapter
        adapter.fit(args.loaders)

        # Test model - predict and evaluate
        if args.loaders["test"] is not None:
            refs, preds = adapter.predict(args.loaders["test"])
            metrics_fold = evaluate(refs, preds, args.setting["task"])
            args.metrics_test.append(metrics_fold)

        # Store weights
        args.weights.append(adapter.model.state_dict())

        # External testing for OOD
        if args.experiment_test[0] != "":
            for i_external in range(len(args.experiment_test)):
                print("External testing: " + args.experiment_test[i_external])

                # Get setting
                setting_external = get_experiment_setting(args.experiment_test[i_external])

                # Prepare dataloaders
                loaders_external = get_dataloader_splits(setting_external["dataframe"], args.data_root_path,
                                                         args.setting["targets"], shots_train="0%", shots_val="0%",
                                                         shots_test="100%", balance=False,
                                                         batch_size=args.batch_size_test,
                                                         batch_size_test=args.batch_size_test,
                                                         num_workers=args.num_workers, seed=iFold,
                                                         task=args.setting["task"], size=args.size,
                                                         resize_canvas=args.resize_canvas,
                                                         crop_foreground=args.crop_foreground)
                # Test model - predict and evaluate
                refs, preds = adapter.predict(loaders_external["test"])
                metrics = evaluate(refs, preds, args.setting["task"])
                args.metrics_external[i_external].append(metrics)

    # Get metrics averaged across folds
    if args.loaders["test"] is not None:
        print("\nTransferability (cross-validation)", end="\n")
        args.metrics = average_folds_results(args.metrics_test, args.setting["task"])
    else:
        args.metrics = None

    # Save experiment metrics
    save_results(args.metrics, args.out_path, id_experiment=generate_experiment_id(args),
                 id_metrics="metrics", save_model=args.save_model, weights=args.weights)

    # Get metrics averaged across fold for external testing
    if args.experiment_test[0] != "":
        for i_external in range(len(args.experiment_test)):
            print("External testing: " + args.experiment_test[i_external])
            metrics = average_folds_results(args.metrics_external[i_external], args.setting["task"])
            # Save experiment metrics
            save_results(metrics, args.out_path, id_experiment=generate_experiment_id(args),
                         id_metrics=args.experiment_test[i_external], save_model=False)


def main():
    parser = argparse.ArgumentParser()

    # Folders, data, etc.
    parser.add_argument('--data_root_path', default=PATH_DATASETS)
    parser.add_argument('--out_path', default=PATH_RESULTS_TRANSFERABILITY, help='output path')
    parser.add_argument('--experiment_description', default=None)
    parser.add_argument('--save_model', default=False, type=lambda x: (str(x).lower() == 'true'))

    # Experiment
    parser.add_argument('--experiment', default='02_MESSIDOR',
                        help='02_MESSIDOR - 13_FIVES - 25_REFUGE - 08_ODIR200x3 - 05_20x3 - 38_MMAC23_train')
    parser.add_argument('--experiment_test', default='',
                        help='02_MESIDOR, 37_DeepDRiD_online_test, 38_MMAC23A_test, 38_MMAC23B_test',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('--method', default='zero_shot',
                        help='lp - tipAdapter - tipAdapter-f - clipAdapter - FT - zero_shot -')

    # Model base weights and architecture
    parser.add_argument('--weights_path', default=None, help='./local_data/results/pretraining/resnet_v2_epoch15.pth')
    parser.add_argument('--load_weights', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--init_imagenet', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--architecture', default='resnet_v1', help='resnet_v1')
    parser.add_argument('--project_features', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--norm_features', default=True, type=lambda x: (str(x).lower() == 'true'))

    # Dataloaders: Training Validation - Testing
    parser.add_argument('--shots_train', default="0%", type=lambda x: (str(x)))
    parser.add_argument('--shots_val', default="0%", type=lambda x: (str(x)))
    parser.add_argument('--shots_test', default="100%", type=lambda x: (str(x)))
    parser.add_argument('--balance', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--folds', default=1, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--batch_size_test', default=4, type=int)
    parser.add_argument('--size', default=(512, 512), help="(512, 512) | (2048, 4096) ")
    parser.add_argument('--crop_foreground', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--resize_canvas', default=True, type=lambda x: (str(x).lower() == 'true'))

    # Vision adapters setting
    parser.add_argument('--domain_knowledge', default=True, type=lambda x: (str(x).lower() == 'true'))

    # Adapters augmentation strategies
    parser.add_argument('--fta', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--tta', default=False, type=lambda x: (str(x).lower() == 'true'))

    # Fine tuning setting
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--update_bn', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--freeze_classifier', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--last_lp', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--save_best', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--patience', default=10, type=int)

    # Saving test predictions option
    parser.add_argument('--test_from_folder', default=[], type=list)

    # Resources
    parser.add_argument('--num_workers', default=8, type=int, help='workers number for DataLoader')

    args, unknown = parser.parse_known_args()

    process(args=args)


if __name__ == "__main__":
    main()
