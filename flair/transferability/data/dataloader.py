"""
Dataset and Dataloaser preparation for vision-language for generalization/transferability.
It contains utils for balancing datasets with regard the categories.
"""

import random
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from flair.pretraining.data.dataset import Dataset
from flair.pretraining.data.transforms import LoadImage, ImageScaling, CopyDict


def get_dataloader_splits(dataframe_path, data_root_path, targets_dict, shots_train="80%", shots_val="0%",
                          shots_test="20%", balance=False, batch_size=8, num_workers=0, seed=0, task="classification",
                          size=(512, 512), resize_canvas=False, batch_size_test=1):

    # Prepare data transforms for pre-processing
    if task == "classification":
        transforms = Compose([CopyDict(), LoadImage(), ImageScaling(size=size)])
    elif task == "segmentation":
        transforms = Compose([CopyDict(), LoadImage(target="image_path"), LoadImage(target="mask_path"),
                              ImageScaling(size=size, target="image", canvas=resize_canvas),
                              ImageScaling(size=size, target="mask", canvas=resize_canvas)])
    else:
        transforms = Compose([CopyDict(), LoadImage(), ImageScaling()])

    # Load data from dict file in txt
    data = []
    dataframe = pd.read_csv(dataframe_path)
    for i in range(len(dataframe)):
        sample_df = dataframe.loc[i, :].to_dict()
        # Image path
        data_i = {"image_path": data_root_path + sample_df["image"]}
        if task == "classification":
            # Image label
            data_i["label"] = targets_dict[eval(sample_df["categories"])[0]]
        if task == "segmentation":
            # Mask path
            data_i["mask_path"] = data_root_path + sample_df["mask"]
            data_i["label"] = 1
        data.append(data_i)

    # Shuffle
    random.seed(seed)
    random.shuffle(data)

    # Train-Val-Test split
    labels = [data_i["label"] for data_i in data]
    unique_labels = np.unique(labels)

    data_train, data_val, data_test = [], [], []
    for iLabel in unique_labels:
        idx = list(np.squeeze(np.argwhere(labels == iLabel)))

        train_samples = get_shots(shots_train, len(idx))
        val_samples = get_shots(shots_val, len(idx))
        test_samples = get_shots(shots_test, len(idx))

        [data_test.append(data[iidx]) for iidx in idx[:test_samples]]
        [data_train.append(data[iidx]) for iidx in idx[test_samples:test_samples+train_samples]]
        [data_val.append(data[iidx]) for iidx in idx[test_samples+train_samples:test_samples+train_samples+val_samples]]

    if balance:
        data_train = balance_data(data_train)

    train_loader = get_loader(data_train, transforms, "train", batch_size, num_workers)
    val_loader = get_loader(data_val, transforms, "val", batch_size_test, num_workers)
    test_loader = get_loader(data_test, transforms, "test", batch_size_test, num_workers)

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    return loaders


def get_loader(data, transforms, split, batch_size, num_workers):
    if len(data) == 0:
        loader = None
    else:
        dataset = Dataset(data=data, transform=transforms)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=split == "train", num_workers=num_workers,
                            drop_last=False)
    return loader


def balance_data(data):

    labels = [iSample["label"] for iSample in data]
    unique_labels = np.unique(labels)
    counts = np.bincount(labels)
    N_max = np.max(counts)

    data_out = []
    for iLabel in unique_labels:
        idx = list(np.argwhere(np.array(labels) == iLabel)[:, 0])
        if N_max-counts[iLabel] > 0:
            idx += random.choices(idx, k=N_max-counts[iLabel])
        [data_out.append(data[iidx]) for iidx in idx]

    return data_out


def get_shots(shots_str, N):

    if "%" in str(shots_str):
        shots_int = int(int(shots_str[:-1]) / 100 * N)
    else:
        shots_int = int(shots_str)
    return shots_int
