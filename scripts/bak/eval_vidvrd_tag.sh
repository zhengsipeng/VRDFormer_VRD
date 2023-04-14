#!/usr/bin/env bash
set -x


python src/main.py with \
    deformable \
    tracking \
    vidvrd_tagging \
    full_res \
    output_dir=models/vidvrd_tagging_tmp