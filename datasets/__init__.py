# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Submodule interface.
"""
import torch
from argparse import Namespace
from pycocotools.coco import COCO
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader, DistributedSampler
import util.misc as utils


class DistributedWeightedSampler(torch.utils.data.DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, replacement=True):
        super(DistributedWeightedSampler, self).__init__(dataset, num_replicas, rank, shuffle)
        assert replacement
        self.replacement = replacement

    def __iter__(self):
        iter_indices = super(DistributedWeightedSampler, self).__iter__()
        if hasattr(self.dataset, 'sample_weight'):
            indices = list(iter_indices)

            weights = torch.tensor([self.dataset.sample_weight(idx) for idx in indices])

            g = torch.Generator()
            g.manual_seed(self.epoch)

            weight_indices = torch.multinomial(
                weights, self.num_samples, self.replacement, generator=g)
            indices = torch.tensor(indices)[weight_indices]

            iter_indices = iter(indices.tolist())
        return iter_indices

    def __len__(self):
        return self.num_samples


def get_coco_api_from_dataset(dataset: Subset) -> COCO:
    """Return COCO class from PyTorch dataset for evaluation with COCO eval."""
    for _ in range(10):
        # if isinstance(dataset, CocoDetection):
        #     break
        if isinstance(dataset, Subset):
            dataset = dataset.dataset

    if not isinstance(dataset, CocoDetection):
        raise NotImplementedError

    return dataset.coco


def build_dataset(split: str, args: Namespace) -> Dataset:
    """Helper function to build dataset for different splits ('train' or 'val')."""

    if args.dataset == "vidvrd":
        from .vidvrd import build_dataset
    else:
        from .vidor import build_dataset
    
    dataset = build_dataset(split, args)

    return dataset


def dataloader_initializer(args):
    dataset_train, sampler_train, data_loader_train = None, None, None
    
    dataset_train = build_dataset(split='train', args=args)
    dataset_val = build_dataset(split='val', args=args)
    dataset_val.zeroshot_triplets = dataset_val.get_triplets().difference(dataset_train.get_triplets())
    
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        #sampler_train = DistributedWeightedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers)

    data_loader_val = DataLoader(
        dataset_val, 
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers)
    
    return data_loader_train, sampler_train, data_loader_val
