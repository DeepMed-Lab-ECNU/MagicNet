# MagicNet: Semi-Supervised Multi-Organ Segmentation via Magic-Cube Partition and Recovery (CVPR 2023)
## Introduction
> [**MagicNet: Semi-Supervised Multi-Organ Segmentation via Magic-Cube Partition and Recovery**](https://arxiv.org/abs/2212.14310),   <br/>
> **Duowen Chen**, Yunhao Bai, Wei Shen, Qingli Li, Lequan Yu and Yan Wang. <br/>
> In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023*  <br/>
> [[arXiv](https://arxiv.org/abs/2212.14310)][[bibtex](https://github.com/DeepMed-Lab-ECNU/MagicNet)][[supp](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Chen_MagicNet_Semi-Supervised_Multi-Organ_CVPR_2023_supplemental.pdf)]

<div align="center" border=> <img src=framework.png width="700" > </div>

## News
- [2024.11.06] We have released our checkpoints and training logs on [model checkpoint](https://github.com/DeepMed-Lab-ECNU/MagicNet/releases) for 30% and 40% BTCV setting.
- [2023.12.15] We have updated 'cube_losses.py'.
- [2023.10.25] We have uploaded the data-splitting file 'btcv.txt' for BTCV dataset to help you reproduce/follow our work^_^!
- [2023.07.01] We have updated the preprocessed data!
- [2023.06.25] Our codes are released!
- [2023.03.16] Repo created. Paper and code will come soon.

## Installation
- PyTorch 1.12.0
- CUDA 11.3 
- Python 3.8.13

## Usage
### Dataset and Pre-processing
The datasets used in our paper are MACT dataset and BTCV dataset. You can download directly our preprocessed data from [baidu netdisk]([https://pan.baidu.com/s/1OVbDXzE_XaTtFGeILQtRyQ](https://pan.baidu.com/s/1RybG0Lr0WFGG85FkW8mFgA) (password: 638u).
### Training Steps
1. Clone the repo and create data path:
```
git clone https://github.com/DeepMed-Lab-ECNU/MagicNet.git
cd MagicNet
mkdir data # create data path
```
2. Put the preprocessed data in ./data/MACT_h5 for MACT dataset. (./data/btcv_h5 for BTCV dataset) and then
```cd code```
3. We train our model on one single NVIDIA 3090 GPU for each dataset.

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
If this code is useful for your research, please consider giving star to our repository and citing our work:
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
## Questions
If you have any questions, welcome contact me at 'duowen_chen@hotmail.com'
