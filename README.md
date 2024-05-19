# RALF
Official implementation of CVPR 2024 paper "[Retrieval-Augmented Open-Vocabulary Object Detection](https://arxiv.org/abs/2404.05687)".

## Introduction
This `prerequisite` branch contains the necessary prerequisites for RALF.

## Installation
- Python 3.10
- PyTorch 1.12.1
```
conda create -n prerequisite python=3.10 -y
conda activate prerequisite
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
``` 
## Preparation
### Datasets
Download [COCO](https://cocodataset.org/#download) and [LVIS](https://www.lvisdataset.org/dataset) to `data`.
```
~/data
    ├── coco
    │   └── annotations/instances_val2017.json
    └── lvis
        └── lvis_v1_val.json
```

## Sampling
First, sample vocabularies with the rank variance.
### COCO
```
python vocab_sampling.py --ds_path data/coco/annotations/instances_val2017.json --v3det_path ./v3det_coco.json --result_path sampled_coco.json
```
### LVIS
```
python vocab_sampling.py --ds_path data/lvis/lvis_v1_val.json --v3det_path ./v3det_lvis.json --result_path sampled_lvis.json
```

## Feature generation
Then, retrieve negative vocabularies based on the similarity score and save them.
### COCO
```
python feature_gen.py --sampled_path ./sampled_coco.json --ds_path data/coco/annotations/instances_val2017.json --result_path ./neg_feature_coco.pkl
```
### LVIS
```
python feature_gen.py --sampled_path ./sampled_lvis.json --ds_path data/lvis/lvis_v1_val.json --result_path ./neg_feature_lvis.pkl
```