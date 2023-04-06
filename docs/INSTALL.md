# Installation
1. Clone and enter this repository:
    ```
    git clone https://github.com/zhengsipeng/VRDFormer_VRD.git
    cd VRDFormer_VRD
    ```

2. Install packages for Python 3.7:

    1. `pip install -r requirements.txt`
    2. Install PyTorch 1.10+cu111 and torchvision 0.11 from [here](https://pytorch.org/get-started/previous-versions/#v150) (pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html).
    3. Install pycocotools (with fixed ignore flag): `pip3 install -U 'git+https://github.com/timmeinhardt/cocoapi.git#subdirectory=PythonAPI'`
    5. Install MultiScaleDeformableAttention package: `python src/vrdformer/models/ops/setup.py build --build-base=src/vrdformer/models/ops/ install`

3. or, install packages for Python 3.8 (deformable-detr is not allowed):
    1. pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html


# Old Version (abandon)
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI & make & make install

python src/models/ops/setup.py build --build-base=src/vrdformer/models/ops install
cp models/resnet50-19c8e357.pth /root/.cache/torch/hub/checkpoints/
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

#AT_DISPATCH_FLOATING_TYPES_AND_HALF
#AT_DISPATCH_FLOATING_TYPES=