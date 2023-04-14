#!/usr/bin/env bash
set -x


python src/train_tag.py with \
    deformable \
    tracking \
    vidvrd_tag \
    full_res \
    output_dir=models/vidvrd_tag_seqlen-8_maxsize-1280

#resume=/data3/zsp/models/r50_deformable_detr-checkpoint.pth \