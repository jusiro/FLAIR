"""
This script contains evaluation utilities for classification
and segmentation problems.

Also, it contains utilities for managing metrics dictionaries,
averaging in k-fold cross-validations and saving results.
"""

import os
import numpy as np
import json

from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_auc_score, f1_score, recall_score


def evaluate(refs, preds, task="classification"):
    if task == "classification":
        metrics = classification_metrics(refs, preds)
        print('Metrics: aca=%2.5f - kappa=%2.3f - macro f1=%2.3f' % (metrics["aca"], metrics["kappa"],
                                                                     metrics["f1_avg"]))
    elif task == "segmentation":
        metrics = segmentation_metrics(refs, preds)
        print('Metrics: dsc=%2.5f - auprc=%2.3f' % (metrics["dsc"], metrics["auprc"]))
    else:
        metrics = {}
    return metrics


def dice(true_mask, pred_mask, non_seg_score=1.0):

    assert true_mask.shape == pred_mask.shape

    true_mask = np.asarray(true_mask).astype(np.bool_)
    pred_mask = np.asarray(pred_mask).astype(np.bool_)

    # If both segmentations are all zero, the dice will be 1. (Developer decision)
    im_sum = true_mask.sum() + pred_mask.sum()
    if im_sum == 0:
        return non_seg_score

    # Compute Dice coefficient
    intersection = np.logical_and(true_mask, pred_mask)
    return 2. * intersection.sum() / im_sum


def au_prc(true_mask, pred_mask):

    # Calculate pr curve and its area
    precision, recall, threshold = precision_recall_curve(true_mask, pred_mask)
    au_prc = auc(recall, precision)

    # Search the optimum point and obtain threshold via f1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    f1[np.isnan(f1)] = 0

    th = threshold[np.argmax(f1)]

    return au_prc, th


def specificity(refs, preds):
    cm = confusion_matrix(refs, preds )
    specificity = cm[0,0]/(cm[0,0]+cm[0,1])
    return specificity


def segmentation_metrics(refs, preds):

    au_prc_metric, th = au_prc(refs.flatten(), preds.flatten())
    dsc = dice(refs, preds > 0.5, non_seg_score=1.0)

    metrics = {"dsc": dsc, "auprc": au_prc_metric}

    return metrics


def classification_metrics(refs, preds):

    # Kappa quadatic
    k = np.round(cohen_kappa_score(refs, np.argmax(preds, -1), weights="quadratic"), 3)

    # Confusion matrix
    cm = confusion_matrix(refs, np.argmax(preds, -1))
    cm_norm = (cm / np.expand_dims(np.sum(cm, -1), 1))

    # Accuracy per class - and average
    acc_class = list(np.round(np.diag(cm_norm), 3))
    aca = np.round(np.mean(np.diag(cm_norm)), 3)

    # recall
    recall_class = [np.round(recall_score(refs == i, np.argmax(preds, -1) == i), 3) for i in np.unique(refs)]
    # specificity
    specificity_class = [np.round(specificity(refs == i, np.argmax(preds, -1) == i), 3) for i in np.unique(refs)]

    # class-wise metrics
    auc_class = [np.round(roc_auc_score(refs == i, preds[:, i]), 3) for i in np.unique(refs)]
    f1_class = [np.round(f1_score(refs == i, np.argmax(preds, -1) == i), 3) for i in np.unique(refs)]

    metrics = {"aca": aca, "kappa": k, "acc_class": acc_class, "f1_avg": np.mean(f1_class),
               "auc_avg": np.mean(auc_class),
               "auc_class": auc_class, "f1_class": f1_class,
               "sensitivity_class": recall_class, "sensitivity_avg": np.mean(recall_class),
               "specificity_class": specificity_class, "specificity_avg": np.mean(specificity_class),
               "cm": cm, "cm_norm": cm_norm}

    return metrics


def average_folds_results(list_folds_results, task):
    metrics_name = list(list_folds_results[0].keys())

    out = {}
    for iMetric in metrics_name:
        values = np.concatenate([np.expand_dims(np.array(iFold[iMetric]), -1) for iFold in list_folds_results], -1)
        out[(iMetric + "_avg")] = np.round(np.mean(values, -1), 3).tolist()
        out[(iMetric + "_std")] = np.round(np.std(values, -1), 3).tolist()

    if task == "classification":
        print('Metrics: aca=%2.3f(%2.3f) - kappa=%2.3f(%2.3f) - macro f1=%2.3f(%2.3f)' % (
            out["aca_avg"], out["aca_std"], out["kappa_avg"], out["kappa_std"], out["f1_avg_avg"], out["f1_avg_std"]))

    return out


def save_results(metrics, out_path, id_experiment=None, id_metrics=None, save_model=False, weights=None):

    # Create results folder
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    # Create experiments folder in results
    if id_experiment is None:
        id_experiment = "experiment" + str(np.random.rand())
    else:
        id_experiment = id_experiment

    # Create main experiment folder
    if not os.path.isdir(out_path + id_experiment):
        os.mkdir(out_path + id_experiment)

    # Store metrics in experiment dataset
    with open(out_path + id_experiment + '/metrics_' + id_metrics + '.json', 'w') as fp:
        json.dump(metrics, fp)

    # Store weights
    if save_model:
        import torch
        for i in range(len(weights)):
            torch.save(weights[i], out_path + id_experiment + '/weights_' + str(i) + '.pth')