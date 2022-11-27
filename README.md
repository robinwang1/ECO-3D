# ECO-3D: Equivariant Contrastive Learning for Pre-training on Perturbed 3D Point Cloud
![image](https://github.com/robinwang1/ECO-3D/blob/main/figs/eco.png)

## Introduction
PyTorch implementation for the paper [ECO-3D: Equivariant Contrastive Learning for Pre-training on Perturbed 3D Point Cloud (AAAI 2023)](http://arxiv.org).

Repository still under construction/refactoring. 

## Installation
#### Install Requirements
    $ cd ECO-3D/
    $ conda env create -f environment.yaml

#### Compile CUDA Files
In VAE model, the CUDA version of Chamfer and EM Distance are used. To compile these files, please refer to the instructions at:
* [PointBERT](https://github.com/lulutang0608/Point-BERT)

#### Download Datasets
Two perturbation datasets are selected:
* [RobustPointSet](https://github.com/AutodeskAILab/RobustPointSet)
* [ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/)

Download the data and place them in corresponding folders.

## VAE Training
To train VAE to obtain the perturbation tokens on RobustPointSet run:
```
$ python main_dvae_hierarchical_distinct_recon.py --config cfgs/RobustPointSet/dvaehdrm.yaml --exp_name pretrain_dvae_hierarchical_distinct_recon
```
You can set the perturbation type in config files.

To train VAE to obtain the perturbation tokens on ScanObjectNN run:
```
$ python main_dvae_hierarchical_distinct_recon.py --config cfgs/ScanObjectNN_models/dvaehdrm_hard.yaml --exp_name pretrain_dvae_scanobjnn_hard
```

After training, you should move the pre-trained VAE models into folders at "eco3d/pretrained".


## ECO-3D Pre-Training and Fine-Tuning
We use the folloing implemetations to respectively verify ECO-3D on RobustPointSet and ScanObjectNN.
* [DGCNN](https://github.com/WangYueFt/dgcnn/tree/master/pytorch)
* [PointNet/PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)


#### RobustPointSet
To pre-train and fine-tune on RobustPointSet with noise perturbations using PointNet backends run: 
```
$ python train_eco3d_cls.py --use_equ pretrained/vae_ckpt.pth --use_con --model pointnet_eco_cls --transform noise
```

To pre-train and fine-tune on RobustPointSet with noise perturbations using DGCNN backends run:  
```
$ python train_eco3d_cls.py --use_equ pretrained/vae_ckpt.pth --use_con --model dgcnn_eco_cls --transform noise
```

#### ScanObjectNN
To pre-train and fine-tune on ScanObjectNN with hardest perturbations using PointNet backends run: 
```
$ python train_eco3d_cls.py --use_equ pretrained/vae_ckpt.pth --use_con --model pointnet_eco_cls --dataset ScanObjectNN --transform hard
```

To pre-train and fine-tune on ScanObjectNN with hardest perturbations using DGCNN backends run:  
```
$ python train_eco3d_cls.py --use_equ pretrained/vae_ckpt.pth --use_con --model dgcnn_eco_cls --dataset ScanObjectNN --transform hard
```

## Contact 
You are welcome to send pull requests or share some ideas with us. Contact information: Robin Wang (robin_wang@pku.edu.cn).

