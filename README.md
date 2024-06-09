# RALF
Official implementation of CVPR 2024 paper "[Retrieval-Augmented Open-Vocabulary Object Detection](https://arxiv.org/abs/2404.05687)".

## Introduction
This `OADP` branch integrates RALF and [OADP](https://github.com/LutingWang/OADP).

## Installation
- Python 3.10.13
- PyTorch 1.12.1
- Cuda 11.3
```
conda create -n ralf python=3.10
conda activate ralf
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install openmim
mim install mmcv_full==1.7.
pip install mmdet==2.25.2

pip install todd_ai==0.3.0
pip install git+https://github.com/LutingWang/CLIP.git
pip install git+https://github.com/lvis-dataset/lvis-api.git@lvis_challenge_2021
pip install nni scikit-learn==1.1.3
pip install yapf==0.40.1
```

## Preparation
Following the [OADP documentation](https://github.com/LutingWang/OADP/blob/main/README.md), prepare the data for baseline training as shown below.
```
~/OADP
    ├── pretrained
    └── data
        ├── coco
        ├── lvis_v1
        └── prompts
```

### Files for RALF
To run RALF on OADP, several files are required.
- `neg_feature_{dataset}.pkl` is the final output of the [`prerequisite`](https://github.com/mlvlab/RALF/tree/prerequisite) branch. This file can be downloaded from [here](https://drive.google.com/drive/folders/1ptNaoSlbvP4CXFXrI2gySCwtaiH3mOwA?usp=sharing).
- `{dataset}_strict.pth` is the final output of the [`RAF`](https://github.com/mlvlab/RALF/tree/RAF) branch.
- `v3det_gpt_noun_chunk_{dataset}_strict.pkl` can be obtained through the [`RAF`](https://github.com/mlvlab/RALF/tree/RAF) branch.

Then, put the files under `~/OADP/ralf` folder.
```
~/OADP/ralf
        ├── neg_feature_coco.pkl
        ├── neg_feature_lvis.pkl
        ├── coco_strict.pth
        ├── lvis_strict.pth
        ├── v3det_gpt_noun_chunk_coco_strict.pkl
        └── v3det_gpt_noun_chunk_lvis_strict.pkl
```

## RAL training
### COCO
```
torchrun --nproc_per_node=4 -m oadp.dp.train coco_ral ./configs/dp/ralf/ral/coco_ral.py
```
### LVIS
```
torchrun --nproc_per_node=4 -m oadp.dp.train lvis_ral ./configs/dp/ralf/ral/lvis_ral.py
```

## RALF inference
### COCO
```
torchrun --nproc_per_node=4 -m oadp.dp.test ./configs/dp/ralf/raf/coco_raf.py work_dirs/coco_ral/iter_32000.pth
```
### LVIS
```
torchrun --nproc_per_node=4 -m oadp.dp.test ./configs/dp/ralf/raf/lvis_raf.py work_dirs/lvis_ral/epoch_24.pth
```

## Results
The checkpoints for RALF are available [here](https://drive.google.com/drive/folders/1ptNaoSlbvP4CXFXrI2gySCwtaiH3mOwA?usp=sharing).
### COCO
|Method|$\text{AP}^\text{N}_\text{50}$|
|---|---|
|OADP + RAL| 31.3 |
|OADP + RALF| 33.4 |

### LVIS
|Method|$\text{AP}_\text{r}$|
|---|---|
|OADP + RAL| 21.5 |
|OADP + RALF| 21.9 |

## RALF demo
To view demo results of applying RALF to an image or video, run the below commands.
### Image
```
torchrun --nproc_per_node=1 -m oadp.dp.demo ./configs/dp/ralf/raf/lvis_raf.py work_dirs/lvis_ral/epoch_24.pth --image demo/input/sample.jpg
```
### Video
```
torchrun --nproc_per_node=1 -m oadp.dp.demo ./configs/dp/ralf/raf/lvis_raf.py work_dirs/lvis_ral/epoch_24.pth --video demo/input/sample.mp4
```