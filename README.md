# MagicNet: Semi-Supervised Multi-Organ Segmentation via Magic-Cube Partition and Recovery (CVPR 2023)

![framework](https://github.com/DeepMed-Lab-ECNU/MagicNet/blob/main/framework.png)
by Duowen Chen, Yunhao Bai, Wei Shen, Qingli Li, Lequan Yu and Yan Wang.
## Introduction
Official code for "[MagicNet: Semi-Supervised Multi-Organ Segmentation via Magic-Cube Partition and Recovery](https://arxiv.org/abs/2212.14310)" (CVPR 2023).

## News
- [2023.06.25] Our codes are released!
- [2023.03.16] Repo created. Paper and code will come soon.

## Installation
- PyTorch 1.12.0
- CUDA 11.3 
- Python 3.8.13

## Usage
We train our model on one single NVIDIA 3090 GPU for each dataset.

To produce the claimed results for MACT dataset:
```
# For 10% labeled data,
CUDA_VISIBLE_DEVICES=0 python train_main_mact.py --labelnum=7

# For 20% labeled data, 
CUDA_VISIBLE_DEVICES=0 python train_main_mact.py --labelnum=13
```
To produce the claimed results for BTCV dataset:
```
# For 30% labeled data,
CUDA_VISIBLE_DEVICES=0 python train_main_btcv.py --labelnum=5

# For 40% labeled data, 
CUDA_VISIBLE_DEVICES=0 python train_main_btcv.py --labelnum=7
```
## Citation
If this code is useful for your research, please consider citing:
```
@InProceedings{Chen_2023_CVPR, 
	author = {Chen, Duowen and Bai, Yunhao and Shen, Wei and Li, Qingli and Yu, Lequan and Wang, Yan}, 
	title = {MagicNet: Semi-Supervised Multi-Organ Segmentation via Magic-Cube Partition and Recovery}, 
	booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
	month = {June}, 
	year = {2023}, 
	pages = {23869-23878} 
}
```
