"""
Dataset and Dataloader preparation for vision-language pre-training
"""

import pandas as pd

from torchvision.transforms import Compose
from torch.utils.data import DataLoader

from flair.pretraining.data.dataset import Dataset, UniformDataset
from flair.pretraining.data.transforms import LoadImage, ImageScaling, SelectRelevantKeys, CopyDict,\
    ProduceDescription, AugmentDescription


def get_loader(dataframes_path, data_root_path, datasets, balance=False, batch_size=8, num_workers=0,
               banned_categories=None, caption="A fundus photograph of [CLS]", augment_description=True):

    """
    Dataloaders generation for vision-language pretraining. Read all dataframes from assembly model and combines
    them into a unified dataframe. Also, a dataloader is conditioned for training.
    """

    # Prepare data sample pre-processing transforms
    transforms = Compose([
        CopyDict(),
        LoadImage(),
        ImageScaling(),
        ProduceDescription(caption=caption),
        AugmentDescription(augment=augment_description),
        SelectRelevantKeys()
    ])

    # Assembly dataframes into a combined data structure
    print("Setting assebly data...")
    data = []
    for iDataset in datasets:
        print("Processing data: " + iDataset)

        dataframe = pd.read_csv(dataframes_path + iDataset + ".csv")

        for i in range(len(dataframe)):
            data_i = dataframe.loc[i, :].to_dict()
            data_i["categories"] = eval(data_i["categories"])
            data_i["atributes"] = eval(data_i["atributes"])

            # Remove banned words - for evaluating on incremental categories
            banned = False
            if banned_categories is not None:
                for iCat in data_i["categories"]:
                    for iiCat in banned_categories:
                        if iiCat in iCat:
                            banned = True
            if banned:
                continue

            # Add sample to general data
            data_i["image_name"] = data_i["image"]
            data_i["image_path"] = data_root_path + data_i["image"]
            data.append(data_i)

    print('Total assembly data samples: {}'.format(len(data)))

    # Set data
    if balance:
        train_dataset = UniformDataset(data=data, transform=transforms)
    else:
        train_dataset = Dataset(data=data, transform=transforms)

    # Set dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Set dataloaders in dict
    datalaoders = {"train": train_loader}

    return datalaoders
