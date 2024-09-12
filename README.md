# Retrieval-Augmented Open-Vocabulary Object Detection
This is the official implementation of CVPR 2024 paper "[Retrieval-Augmented Open-Vocabulary Object Detection](https://arxiv.org/abs/2404.05687)".

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/retrieval-augmented-open-vocabulary-object/open-vocabulary-object-detection-on-mscoco)](https://paperswithcode.com/sota/open-vocabulary-object-detection-on-mscoco?p=retrieval-augmented-open-vocabulary-object)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/retrieval-augmented-open-vocabulary-object/open-vocabulary-object-detection-on-lvis-v1-0)](https://paperswithcode.com/sota/open-vocabulary-object-detection-on-lvis-v1-0?p=retrieval-augmented-open-vocabulary-object)
> Jooyean Kim*, Eulrang Cho*, Sehyung Kim, Hyunwoo J. Kim.
> 
> Department of Computer Science and Engineering, Korea University
> 
![ralf_figure](https://github.com/user-attachments/assets/09e8ac36-135e-42cd-a6f2-877288e4a3da)
## Introduction

RALF is structured into multiple branches.

- [`prerequisite`](https://github.com/mlvlab/RALF/tree/prerequisite) branch: The code for prerequisites necessary for running RALF. 
- [`RAF`](https://github.com/mlvlab/RALF/tree/RAF) branch: The code for training RAF.

The other branches are the integration of existing OVD model and RALF.
- [`OADP`](https://github.com/mlvlab/RALF/tree/OADP) branch: The baseline is [OADP](https://github.com/LutingWang/OADP).
- [`Centric`](https://github.com/mlvlab/RALF/tree/Centric) branch: The baseline is [Object-Centric-OVD](https://github.com/hanoonaR/object-centric-ovd).
- [`DetPro`](https://github.com/mlvlab/RALF/tree/DetPro) branch: The baseline is [DetPro](https://github.com/dyabel/detpro).

## Results
### COCO
|Model|$\text{AP}^\text{N}_\text{50}$|
|---|---|
|RALF + OADP| 33.4 |
|RALF + Object-Centric-OVD| 41.3 |

### LVIS
|Model|$\text{AP}_\text{r}$|
|---|---|
|RALF + OADP| 21.9 |
|RALF + DetPro| 21.1 |
|RALF + Object-Centric-OVD| 18.5 |

## Citation
```
@inproceedings{kim2024retrieval,
  title={Retrieval-Augmented Open-Vocabulary Object Detection},
  author={Kim, Jooyeon and Cho, Eulrang and Kim, Sehyung and Kim, Hyunwoo J},
  booktitle={CVPR},
  year={2024}
}
```
## References
This code is built on [CLIP](https://github.com/openai/CLIP), [V3Det](https://github.com/V3Det/V3Det), [GPT-3](https://github.com/openai/gpt-3), [OADP](https://github.com/LutingWang/OADP), [Object-Centric-OVD](https://github.com/hanoonaR/object-centric-ovd) and [DetPro](https://github.com/dyabel/detpro).
