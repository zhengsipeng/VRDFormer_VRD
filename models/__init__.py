# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from copy import deepcopy

def build_model(args):
    if args.task=="gt":
        from .vrdformer import build as build_vrdformer
    else:
        from .vrdformer_v2 import build as build_vrdformer
    return build_vrdformer(args)


def model_initializer(args, device):
    # Build the model
    model, criterion, weight_dict = build_model(args)
    model.to(device)

    # Get a copy of the model for exponential moving averaged version of the model
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu], 
            broadcast_buffers=False, 
            find_unused_parameters=True
            )
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("number of trainable model params: %.2f M"%(n_parameters/1000000.))
    
    return model, model_without_ddp, criterion, n_parameters