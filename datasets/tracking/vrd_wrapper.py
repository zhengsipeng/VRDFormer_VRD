# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
VRD wrapper which combines sequences to a dataset.
"""
from torch.utils.data import Dataset

from .vrd_sequence import VRDSequence


class VRDWrapper(Dataset):
    """A Wrapper for the VRD_Sequence class to return multiple sequences."""
    def __init__(self, dbname: str, dets: str, **kwargs) -> None:
        """Initliazes all subset of the dataset.
        Keyword arguments:
        dbname -- the dataset name
        split -- the split of the dataset to use
        kwargs -- kwargs for the MOT17Sequence dataset
        """
        val_videos = []
        ann_files = []

        self.data = []
        for i, video in enumerate(val_videos):
            # PRIVATE: use the detection provided by our e2e work
            # PUBLIC: use the detection provided by other public works
            # ANNO: use the detection provided by annos
            self.data.append(VRDSequence(dbname=dbname, video_name=video, dets=dets, 
                             ann_file=ann_files[i], **kwargs))


    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self.data[idx]