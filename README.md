# RALF
Official implementation of CVPR 2024 paper "[Retrieval-Augmented Open-Vocabulary Object Detection](https://arxiv.org/abs/2404.05687)".

## Introduction
This `Centric` branch integrates RALF and [Object-Centric-OVD](https://github.com/hanoonaR/object-centric-ovd).

## Installation
- Python 3.8
- PyTorch 1.10.0
- Cuda 11.3
```
conda create -n ralf python=3.8 -y
conda activate ralf
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge -y

cd external
python -m pip install -e detectron2
cd ..

pip install -r requirements.txt
pip install "numpy<1.24"
pip install setuptools==59.5.0
```

## Preparation
Following the [Object-Centric-OVD documentation](https://github.com/hanoonaR/object-centric-ovd/blob/main/docs/DATASETS.md), prepare the data for baseline training as shown below.
```
~/Object-Centric-OVD
    ├── datasets
    │   ├── coco
    │   ├── lvis
    │   ├── imagenet
    │   └── MAVL_proposals
    └── saved_models
```

### Files for RALF
To run RALF on Object-Centric-OVD, several files are required.
- `neg_feature_{dataset}.pkl` is the final output of the [`prerequisite`](https://github.com/mlvlab/RALF/tree/prerequisite) branch. This file can be downloaded from [here](https://drive.google.com/drive/folders/1iiXRx8NFL7sMlpuRwaXacvAm3rTzq-b2?usp=sharing).
- `{dataset}_strict.pth` is the final output of the [`RAF`](https://github.com/mlvlab/RALF/tree/RAF) branch.
- `v3det_gpt_noun_chunk_{dataset}_strict.pkl` can be obtained through the [`RAF`](https://github.com/mlvlab/RALF/tree/RAF) branch.

Then, put the files under `~/Object-Centric-OVD/ralf` folder.
```
~/Object-Centric-OVD/ralf
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
python train_net.py --num-gpus 4 --config-file configs/coco/ral.yaml
```
### LVIS
```
python train_net.py --num-gpus 4 --config-file configs/lvis/ral.yaml
```

## RALF inference
### COCO
```
python train_net.py --num-gpus 4 --config-file configs/coco/raf.yaml --eval-only
```
### LVIS
```
python train_net.py --num-gpus 4 --config-file configs/lvis/raf.yaml --eval-only
```

## Results
The checkpoints for RALF are available [here](https://drive.google.com/drive/folders/1iiXRx8NFL7sMlpuRwaXacvAm3rTzq-b2?usp=sharing).
### COCO
|Method|$\text{AP}^\text{N}_\text{50}$|
|---|---|
|Object-Centric-OVD + RAL| 41.0 |
|Object-Centric-OVD + RALF| 41.3 |

### LVIS
|Method|$\text{AP}_\text{r}$|
|---|---|
|Object-Centric-OVD + RAL| 18.5 |
|Object-Centric-OVD + RALF| 18.5 |