# upernet and residual recurrent unet for image segmentation

By [Shu Lei](https://github.com/maosengshulei/improved_Unet/)

# Introduction

UPerNet based on Feature Pyramid Network (FPN) and Pyramid Pooling Module (PPM), with down-sampling rate of 4, 8 and 16. It doesn't need dilated convolution, a operator that is time-and-memory consuming. Without bells and whistles, it is comparable or even better compared with PSPNet, while requires much shorter training time and less GPU memory. E.g., you cannot train a PSPNet-101 on TITAN Xp GPUs with only 12GB memory, while you can train a UPerNet-101 on such GPUs.

for more detail,please refer to [arXiv paper](https://arxiv.org/abs/1801.05746).





 One deep learning technique, U-Net, has become one of the most popular for these applications. In this project, a Recurrent Convolutional Neural Network (RCNN) based on U-Net as well as a Recurrent Residual Convolutional Neural Network (RRCNN) based on U-Net models was proposed, which are named RU-Net and R2U-Net respectively. The proposed models utilize the power of U-Net, Residual Network, as well as RCNN. There are several advantages of these proposed architectures for segmentation tasks. First, a residual unit helps when training deep architecture. Second, feature accumulation with recurrent residual convolutional layers ensures better feature representation for segmentation tasks. Third, it allows us to design better U-Net architecture with same number of network parameters with better performance for medical image segmentation. [arxiv paper](https://arxiv.org/abs/1802.06955).


## Environment
The code is developed under the following configurations.
- Hardware: 1 GPUs (with at least 8G GPU memories) (change ```[--num_gpus NUM_GPUS]``` accordingly)
- Software: Ubuntu 16.04 LTS, CUDA 8.0, ***Python>=3.5***, ***PyTorch=0.3.0***


## Training
Train a UPerNet (e.g., ResNet-50 or ResNet-101)
Train a UPerNet (e.g., ResNet-50 or ResNet-101)
```bash
python train.py --num_gpus NUM_GPUS --arch_encoder resnet50 --arch_decoder upernet 
--deep_sup_factor 0 
```

Evaluate a UPerNet (e.g, UPerNet-50

model path and predict image path should be configured in code
```bash
python generate_predict_image.py 
```


## pretrained model

pretrained resnet model can be found in code resnet.py.

pretrained resnext model can be downloaded from [github](https://github.com/Youngkl0726/ResNext_pytorch_PretrainedModel).

