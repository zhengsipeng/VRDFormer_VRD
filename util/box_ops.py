# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import enum
import os
import torch
from torchvision.ops.boxes import box_area
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pdb


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """
    Equivalent to nn.functional.interpolate, but with support for empty channel sizes.
    """
    if input.numel() > 0:
        return torch.nn.functional.interpolate(
            input, size, scale_factor, mode, align_corners
        )

    assert (
        input.shape[0] != 0 or input.shape[1] != 0
    ), "At least one of the two first dimensions must be non zero"

    if input.shape[1] == 0:
        # Pytorch doesn't support null dimension on the channel dimension, so we transpose to fake a null batch dim
        return torch.nn.functional.interpolate(
            input.transpose(0, 1), size, scale_factor, mode, align_corners
        ).transpose(0, 1)

    # empty batch dimension is now supported in pytorch
    return torch.nn.functional.interpolate(
        input, size, scale_factor, mode, align_corners
    )


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
 

def debug_and_vis(dbname, samples, targets, step_num):
    with open("data/%s/obj.txt"%dbname, "r") as f:
        obj_names = [l.strip() for l in f.readlines()]
    with open("data/%s/action.txt"%dbname, "r") as f:
        action_names = [l.strip() for l in f.readlines()]

    debug_path = "data/%s/debugging"%dbname
    if not os.path.exists(debug_path):
        os.makedirs(debug_path)
    unloader = transforms.ToPILImage()
    imgs = samples.tensors.cpu()  # N, 3, H, W
    targets = targets[0]  # N
    for i, target in enumerate(targets):
        img = unloader(imgs[i])
        font = ImageFont.truetype(font='data/Gemelli.ttf', size=np.floor(1.5e-2 * np.shape(img)[1] + 15).astype('int32'))
        draw = ImageDraw.Draw(img)
        sboxes = target["sboxes"].tolist()
        oboxes = target["oboxes"].tolist()
        sclss = target["sclss"].tolist()
        oclss = target["oclss"].tolist()
        sclss = [obj_names[sclss[j]]+"%02d"%j for j in range(len(sboxes))]
        oclss = [obj_names[oclss[j]] + "%02d"%j for j in range(len(oboxes))]
        vclss = target["raw_vclss"]
        
        for j, sbox in enumerate(sboxes):
            draw.rectangle([sbox[0], sbox[1], sbox[2], sbox[3]], outline='red', width=2)
            sclass = sclss[j]
            label_size = draw.textsize(sclass, font)
            text_origin = np.array([sbox[0], sbox[1] - label_size[1]])
            draw.text(text_origin, str(sclass), fill=(255, 255, 255), font=font)

        for j, obox in enumerate(oboxes):
            draw.rectangle([obox[0], obox[1], obox[2], obox[3]], outline='green', width=2)
            oclass = oclss[j]
            label_size = draw.textsize(oclass, font)
            text_origin = np.array([obox[0], obox[1] - label_size[1]])
            draw.text(text_origin, str(oclass), fill=(255, 255, 255), font=font)
        
        img_name = "%d-%d"%(step_num, i)
        for j, verb_class in enumerate(vclss):
            if j > 5:
                break
            sclass, oclass = sclss[j], oclss[j]
            img_name += "_"+sclass+"-"
            for v in verb_class:
                img_name += action_names[v]+"-"
                break
            img_name += "-"+oclass
        img_name += ".jpg"
        del draw
        img.save(debug_path+"/"+img_name)