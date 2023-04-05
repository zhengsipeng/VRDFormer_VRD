# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from .backbone import build_backbone
from .transformer_stage2 import build_transformer
from .deformable_transformer import build_deformable_transformer
from .matcher import build_matcher


def build_model(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)
    
    if args.deformable:
        transformer = build_deformable_transformer(args)
    else:
        transformer = build_transformer(args)
        
    matcher = None
    #matcher = build_matcher(args) if args.task=="detection" else None
    
    num_obj_classes = 80 if args.dataset == 'vidor' else 35
    num_verb_classes = 42 if args.dataset == 'vidor' else 132  # multi-label
    
    detr_kwargs = {
        'backbone': backbone,
        'transformer': transformer,
        'num_obj_classes': num_obj_classes - 1 if args.focal_loss else num_obj_classes,
        'num_verb_classes': num_verb_classes - 1 if args.focal_loss else num_verb_classes,
        'num_queries': args.num_queries,
        'aux_loss': args.aux_loss,
        'overflow_boxes': args.overflow_boxes,
        'deformable': args.deformable,
        'num_feature_levels': args.num_feature_levels,
        'with_box_refine': args.with_box_refine,
        'multi_frame_attention': args.multi_frame_attention,
        'multi_frame_encoding': args.multi_frame_encoding,
        'merge_frame_features': args.merge_frame_features,
        }
    
    if args.stage==1:
        from .vrdformer import VRDFormer, SetCriterion
        tracking_kwargs = {
            'track_query_false_positive_prob': args.track_query_false_positive_prob,
            'track_query_false_negative_prob': args.track_query_false_negative_prob,
            'matcher': matcher,
            'backprop_prev_frame': args.track_backprop_prev_frame
        }
        if args.tracking:
            model = VRDFormerTracing(tracking_kwargs, detr_kwargs)
        else:
            model = VRDFormer(**detr_kwargs)
    else:
        from .vrdformer_stage2 import VRDFormer, SetCriterion
        model = VRDFormer(**detr_kwargs)
        
    weight_dict = {'loss_sub_ce': args.sub_loss_coef,
                   'loss_obj_ce': args.obj_loss_coef,
                   'loss_verb_ce': args.verb_loss_coef,}
    
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    losses = ["sub_labels", "obj_labels", "verb_labels"]
    
    criterion = SetCriterion(
        num_obj_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        track_query_false_positive_eos_weight=args.track_query_false_positive_eos_weight,
    )

    criterion.to(device)
    
    return model, criterion, weight_dict


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