# The Urban Tree Canopy Cover in Brazil - Nationwide Perspectives from High-Resolution Remote Sensing Images

This Pytorch repository contains the code for our work [Semi-supervised Semantic Segmentation with High- and Low-level Consistency](https://arxiv.org/pdf/1908.05724.pdf). The approach uses two network branches that link semi-supervised classification with semi-supervised segmentation including self-training. The approach attains significant improvement over existing methods, especially when trained with very few labeled samples.

Urban tree canopy maps are essential for providing urban ecosystem services. The relationship between urban
trees and urban climate change, air pollution, urban noise, biodiversity, urban crime, health, poverty, 
and social inequality provides important information for better understanding and management of cities. To better service Brazil’s
urban ecosystem, we developed a semi-supervised deep learning method, which is able to learn semantic segmentation 
knowledge from both labeled and unlabeled images, to segment urban trees from high spatial resolution remote
sensing images. Using this approach, we created 0.5 m fine-scale tree canopy products for 472 cities in Brazil and
made them freely available to the community. Results showed that the urban tree canopy coverage in Brazil is between
5% and 35%, and the average urban tree canopy cover is approximately 18.68%. The statistical results of these tree
canopy maps quantify the nationwide urban tree canopy inequality problem in Brazil. Urban tree canopy coverage
from 130 cities that can accommodate approximately 27.22% of the total population is greater than 20%, whereas 342
cities that can accommodate approximately 42% of the total population have tree canopy cover less than 20%. 
Moreover, we found that many small-area cities with low tree canopy coverage require great attention from the Brazilian
government. We expect that urban tree canopy maps will encourage research on Brazilian urban ecosystem services
to support urban development and improve inhabitants’ quality of life to achieve the goals of the Agenda for Sustainable Development. 
	In addition, it can serve as a benchmark dataset for other high-resolution and mid/low-resolution
remote sensing images urban tree canopy mapping results assessments.


## Package pre-requisites
The code runs on Python 3 and Pytorch 0.4 The following packages are required. 

```
pip install scipy tqdm matplotlib numpy opencv-python
```

## Dataset preparation

Download ImageNet pretrained Resnet-101([Link](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)) and place it ```./pretrained_models/```

### PASCAL VOC
Download the dataset([Link](https://lmb.informatik.uni-freiburg.de/resources/datasets/voc_dataset.tar.gz)) and extract in ```./data/voc_dataset/```

### PASCAL Context
Download the annotations([Link](https://lmb.informatik.uni-freiburg.de/resources/datasets/pascal_context_labels.tar.gz)) and extract in ```./data/pcontext_dataset/```

### Cityscapes
Download the dataset from the Cityscapes dataset server([Link](https://www.cityscapes-dataset.com/)). Download the files named 'gtFine_trainvaltest.zip', 'leftImg8bit_trainvaltest.zip' and extract in ```./data/city_dataset/```  

## Training and Validation on PASCAL-VOC Dataset

### Training fully-supervised Baseline (FSL)
```
python train_full.py    --dataset pascal_voc  \
                        --checkpoint-dir ./checkpoints/voc_full \
                        --ignore-label 255 \
                        --num-classes 21 
```
### Training semi-supervised s4GAN (SSL)
```
python train_s4GAN.py   --dataset pascal_voc  \
                        --checkpoint-dir ./checkpoints/voc_semi_0_125 \
                        --labeled-ratio 0.125 \
                        --ignore-label 255 \ 
                        --num-classes 21 \
                        --split-id ./splits/voc/split_0.pkl
``` 
### Validation 
```
python evaluate.py --dataset pascal_voc  \
                   --num-classes 21 \
                   --restore-from ./checkpoints/voc_semi_0_125/VOC_30000.pth 
```

## Training and Validation on PASCAL-Context Dataset
```
python train_full.py    --dataset pascal_context  \
                        --checkpoint-dir ./checkpoints/pc_full \
                        --ignore-label -1 \
                        --num-classes 60

python train_s4GAN.py  --dataset pascal_context  \
                       --checkpoint-dir ./checkpoints/pc_semi_0_125 \
                       --labeled-ratio 0.125 \
                       --ignore-label -1 \
                       --num-classes 60 \
                       --split-id ./splits/pc/split_0.pkl
                       --num-steps 60000

python evaluate.py     --dataset pascal_context  \
                       --num-classes 60 \
                       --restore-from ./checkpoints/pc_semi_0_125/VOC_40000.pth
```

## Training and Validation on Cityscapes Dataset
```
python train_full.py    --dataset cityscapes \
                        --checkpoint-dir ./checkpoints/city_full_0_125 \
                        --ignore-label 250 \
                        --num-classes 19 \
                        --input-size '256,512'  

python train_s4GAN.py   --dataset cityscapes \
                        --checkpoint-dir ./checkpoints/city_semi_0_125 \
                        --labeled-ratio 0.125 \
                        --ignore-label 250 \
                        --num-classes 19 \
                        --split-id ./splits/city/split_0.pkl \
                        --input-size '256,512' \
                        --threshold-st 0.7 \
                        --learning-rate-D 1e-5 

python evaluate.py      --dataset cityscapes \
                        --num-classes 19 \
                        --restore-from ./checkpoints/city_semi_0_125/VOC_30000.pth 
```

## Instructions for setting-up Multi-Label Mean-Teacher branch
This work is based on the [Mean-Teacher](https://arxiv.org/abs/1703.01780) Semi-supervised Learning work. To use the MLMT branch, follow the instructions below. 
1. Fork the [mean-teacher](https://github.com/CuriousAI/mean-teacher) repo. 
2. Modify the fully connected layer, according to the number of classes and add Sigmoid activation for multi-label classification.
3. Use Binary Cross Entropy loss fucntion instead of multi-class Cross entropy. 
4. Load the pretrained ImageNet weights for ResNet-101 from ```./pretrained_models/```.
5. Use student/teacher predictions for Network output fusion with s4GAN branch. 
6. For lower labeled-ratio, early stopping might be required.  


## Acknowledgement

Parts of the code have been adapted from: 
[DeepLab-Resnet-Pytorch](https://github.com/speedinghzl/Pytorch-Deeplab), [AdvSemiSeg](https://github.com/hfslyc/AdvSemiSeg), [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)


## Citation

```
@article{1908.05724,
    Author = {Sudhanshu Mittal and Maxim Tatarchenko and Thomas Brox},
    Title = {Semi-Supervised Semantic Segmentation with High- and Low-level Consistency},
    journal = {arXiv preprint arXiv:1908.05724},
    Year = {2019},
}
```

