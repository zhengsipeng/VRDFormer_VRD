# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Factory of tracking datasets.
"""
from typing import Union
from torch.utils.data import ConcatDataset
from .vrd_wrapper import VRDWrapper


DATASETS = {}

# Fill all available datasets, change here to modify / add new datasets.
for dbname in ["VIDVRD", "VIDOR"]:
    for split in ["VAL"]:
        for dets in ['PRIVATE', 'PUBLIC', 'ANNO']:
            # PRIVATE: detection from VRDFormer
            # PUBLIC: detection from previous works
            # ANNO: detection from annotation
            name = dbname + "-" + split + '-' + dets
            DATASETS[name] = (lambda kwargs, dbname=dbname, dets=dets: VRDWrapper(dbname, dets, **kwargs))
        
        

class VRDTrackDatasetFactory:
    """A central class to manage the individual dataset loaders.
    This class contains the datasets. Once initialized the individual parts (e.g. sequences)
    can be accessed.
    """

    def __init__(self, datasets: Union[str, list], **kwargs) -> None:
        """Initialize the corresponding dataloader.
        Keyword arguments:
        datasets --  the name of the dataset or list of dataset names
        kwargs -- arguments used to call the datasets
        """
        if isinstance(datasets, str):
            datasets = [datasets]

        self._data = None
        for dataset in datasets:
            assert dataset in DATASETS, f"[!] Dataset not found: {dataset}"

            if self._data is None:
                self._data = DATASETS[dataset](kwargs)
            else:
                self._data = ConcatDataset([self._data, DATASETS[dataset](kwargs)])

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]