
"""
Specialized Datasets implementation from pytorch to create balanced datasets
with regard the used datasets for pretraining.
"""

import collections.abc
import numpy as np

from torch.utils.data import Dataset as _TorchDataset
from typing import Any, Callable, Optional, Sequence, Union
from torch.utils.data import Subset


class Dataset(_TorchDataset):
    """
    A generic data with a length property and an optional callable data transform
    when fetching a data sample.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = data[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]
    """

    def __init__(self, data: Sequence, transform: Optional[Callable] = None) -> None:
        """
        Args:
            data: input data to load and transform to generate data for model.
            transform: a callable data transform on input data.

        """
        self.data = data
        self.transform: Any = transform

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.data[index]
        return self.transform(data_i) if self.transform is not None else data_i

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            # data[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # data[[1, 3, 4]]
            return Subset(dataset=self, indices=index)
        return self._transform(index)


class UniformDataset(Dataset):
    def __init__(self, data, transform):
        super().__init__(data=data, transform=transform)
        self.datasetkey = []
        self.data_dic = []
        self.datasetnum = []
        self.datasetlen = 0
        self.dataset_split(data)

    def dataset_split(self, data):
        keys = []
        for img in data:
            keys.append(img["image_name"].split("/")[0])

        self.datasetkey = list(np.unique(keys))

        data_dic = {}
        for iKey in self.datasetkey:
            data_dic[iKey] = [data[iSample] for iSample in range(len(keys)) if keys[iSample]==iKey]
        self.data_dic = data_dic

        self.datasetnum = []
        for key, item in self.data_dic.items():
            assert len(item) != 0, f'the data {key} has no data'
            self.datasetnum.append(len(item))
        self.datasetlen = len(self.datasetkey)

    def _transform(self, set_key, data_index):
        data_i = self.data_dic[set_key][data_index]
        return self.transform(data_i) if self.transform is not None else data_i

    def __getitem__(self, index):
        ## the index generated outside is only used to select the data
        ## the corresponding data in each data is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        set_key = self.datasetkey[set_index]

        data_index = np.random.randint(self.datasetnum[set_index], size=1)[0]
        return self._transform(set_key, data_index)