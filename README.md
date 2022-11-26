# ECO-3D: Equivariant Contrastive Learning for Pre-training on Perturbed 3D Point Cloud
![image](https://github.com/robinwang1/ECO-3D/blob/main/assets/fig1.png)

## Introduction
PyTorch implementation for the paper [ECO-3D: Equivariant Contrastive Learning for Pre-training on Perturbed 3D Point Cloud (AAAI 2023)](http://arxiv.org/abs/2203.03888).

Repository still under construction/refactoring. 

## Installation
#### Install Requirements
    $ cd ECO-3D/
    $ conda env create -f environment.yaml

#### Download RobustPointSet and ScanObjectNN
We use two datasets:
* [ModelNet-C](https://github.com/AutodeskAILab/RobustPointSet)
* [ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/)
Place the dataset in the dataset

#### Training VAE
To train VAE to obtain the perturbation tokens on ModelNet-C run:
```
$ python main_dvae_hierarchical_distinct_recon.py --config cfgs/RobustPointSet/dvaehdrm.yaml --exp_name pretrain_dvae_hierarchical_distinct_recon
```
You can set the perturbation type in the correspounding config files.

To train VAE to obtain the perturbation tokens on ScanObjectNN run:
```
$ python main_dvae_hierarchical_distinct_recon.py --config cfgs/ScanObjectNN_models/dvaehdrm_hard.yaml --exp_name pretrain_dvae_scanobjnn_hard
```

After training, you should move the pre-trained VAE models into corresponding folders at "./pretrained_models/" to obtain.


We use the folloing implemetations to respectively verify ECO-3D on ModelNet40 and ShapeNet16.
* [DGCNN](https://github.com/WangYueFt/dgcnn/tree/master/pytorch)
* [PointNet/PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)



## Train and Evaluate

#### ModelNet40
To pre-train and fine-tune on ModelNet-C with noise perturbations using PointNet backends run: 
```
$ python train_eco3d_cls.py  --log_dir test_equ_dvaeh --use_equ pretrained/dvaeh/ckpt-epoch-018.pth --transform rotation
```

To pre-train and fine-tune on ModelNet-C with noise perturbations using DGCNN backends run:  
```
$ python train_eco3d_cls.py  --log_dir test_equ_dvaehdrm --use_equ pretrained/dvaehdrm/ckpt-epoch-017.pth --transform noise
```

#### ScanObjectNN
To pre-train and fine-tune on ScanObjectNN with hardest perturbations using PointNet backends run: 
```
$ python train_eco3d_cls.py  --log_dir test_equ_dvaehdrm --use_equ pretrained/dvaehdrm/vae_ckpt.pth --dataset ScanObjectNN --transform hard
```

To pre-train and fine-tune on ScanObjectNN with hardest perturbations using DGCNN backends run:  
```
$ python train_eco3d_cls.py  --log_dir test_equ_dvaehdrm --use_equ pretrained/dvaehdrm/vae_ckpt.pth --dataset ScanObjectNN --transform hard
```

## Contact 
You are welcome to send pull requests or share some ideas with us. Contact information: Robin Wang (robin_wang@pku.edu.cn).
