# RALF
Official implementation of CVPR 2024 paper "[Retrieval-Augmented Open-Vocabulary Object Detection](https://arxiv.org/abs/2404.05687)".

## Introduction
This `DetPro` branch integrates RALF and [DetPro](https://github.com/dyabel/detpro).

## Installation
- Python 3.8
- PyTorch 1.7.0
- Cuda 11.0
```
conda create -n ralf python=3.8 -y
conda activate ralf
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=11.0 -c pytorch -y
pip install cython==0.29.33

cd DetPro
pip install -e .
pip install git+https://github.com/openai/CLIP.git
pip uninstall pycocotools -y
pip uninstall mmpycocotools -y
pip install mmpycocotools
pip install git+https://github.com/lvis-dataset/lvis-api.git
pip install mmcv-full==1.2.5 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
pip install "numpy<1.24"
pip install yapf==0.40.1
```

## Preparation
Following the [DetPro documentation](https://github.com/dyabel/detpro/blob/main/README.md), prepare the data for baseline training as shown below.
```
~/DetPro
    ├── data
    │   ├── lvis_v1
    │   ├── current_mmdetection_Head.pth
    │   └── lvis_clip_image_embedding.zip
    └── iou_neg5_ens.pth
```

### Files for RALF
To run RALF on DetPro, several files are required.
- `neg_feature_lvis.pkl` is the final output of the [`prerequisite`](https://github.com/mlvlab/RALF/tree/prerequisite) branch. This file can be downloaded from [here](https://drive.google.com/drive/folders/1vKMwGkjqUV6u3AQNzT8Lxsa4VcQ0CkdL?usp=sharing).
- `lvis_strict.pth` is the final output of the [`RAF`](https://github.com/mlvlab/RALF/tree/RAF) branch.
- `v3det_gpt_noun_chunk_lvis_strict.pkl` can be obtained through the [`RAF`](https://github.com/mlvlab/RALF/tree/RAF) branch.

Then, put the files under `~/DetPro/ralf` folder.
```
~/DetPro/ralf
           ├── neg_feature_lvis.pkl
           ├── lvis_strict.pth
           └── v3det_gpt_noun_chunk_lvis_strict.pkl
```

## RAL training
```
./tools/dist_train.sh configs/ralf/ral.py 4 --work-dir workdirs/ral --cfg-options model.roi_head.prompt_path=iou_neg5_ens.pth model.roi_head.load_feature=True
```

## RALF inference
```
./tools/dist_test.sh configs/ralf/raf.py workdirs/ral/epoch_20.pth 4 --eval segm --cfg-options model.roi_head.prompt_path=iou_neg5_ens.pth model.roi_head.load_feature=False
```

## Results
The checkpoint for RALF is available [here](https://drive.google.com/drive/folders/1vKMwGkjqUV6u3AQNzT8Lxsa4VcQ0CkdL?usp=sharing).
|Method|$\text{AP}_\text{r}$|
|---|---|
|DetPro + RAL| 21.3 |
|DetPro + RALF| 21.1 |