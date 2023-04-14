# VRDFormer: End-to-end Video Relation Detection with Transformers
This repository provides the implementation of the [VRDFormer: End-to-end Video Relation Detection with Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_VRDFormer_End-to-End_Video_Visual_Relation_Detection_With_Transformers_CVPR_2022_paper.pdf) paper.
The codebase builds upon [DETR](https://github.com/facebookresearch/detr), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [TrackFormer](https://github.com/timmeinhardt/trackformer).

## Abstract

Visual relation understanding plays an essential role for holistic video understanding. Most previous works adopt a multi-stage framework for video visual relation detection (VidVRD), which cannot capture long-term spatiotemporal contexts in different stages and also suffers from inefficiency. In this paper, we propose a transformerbased framework called VRDFormer to unify these decoupling stages. Our model exploits a query-based approach to autoregressively generate relation instances. We specifically design static queries and recurrent queries to enable efficient object pair tracking with spatio-temporal contexts. The model is jointly trained with object pair detection and relation classification. Extensive experiments
on two benchmark datasets, ImageNet-VidVRD and VidOR, demonstrate the effectiveness of the proposed VRDFormer, which achieves the state-of-the-art performance on both relation detection and relation tagging tasks.

## Installation
We refer to our [docs/INSTALL.md](docs/INSTALL.md) for detailed installation instructions.

## Train VRDFormer
Train VidVRD based on detr with 8GPUs and batchsize=32
```
sh script/stage/train_mgpu.sh
```

Train VidVRD based on deformable detr
```
sh script/stage/train_deform_mgpu.sh
```

## Evaluate VRDFormer