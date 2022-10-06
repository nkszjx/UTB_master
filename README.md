# The Urban Tree Canopy Cover in Brazil - Nationwide Perspectives from High-Resolution Remote Sensing Images

This is the pytorch codes for our paper: The Urban Tree Canopy Cover in Brazil - Nationwide Perspectives from High-Resolution Remote Sensing Images.

To better service Brazil’s urban ecosystem, we developed a semi-supervised deep learning method, which is able to learn semantic segmentation knowledge from both labeled and unlabeled images, to segment urban trees from high spatial resolution remote sensing images. Using this approach, we created 0.5 m fine-scale tree canopy products for 472 cities in Brazil and made them freely available to the community. 

The approach uses two network branches that link semi-supervised classification with semi-supervised segmentation including self-training. The approach attains significant improvement over existing methods, especially when trained with very few labeled samples.


## Package pre-requisites
The code runs on Python 3 and Pytorch 0.4 The following packages are required. 

```
pip install scipy tqdm matplotlib numpy opencv-python
```

## Dataset preparation

Download ImageNet pretrained Resnet-101([Link](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)) and place it ```./pretrained_models/```

### Training semi-supervised (SSL)
```
python train.py   
```
or
```
nohup python -u train.py > ./log/out_list.log 2>&1 &
``` 
### Validation 
```
python evaluate.py
```

## Acknowledgement

Parts of the code have been adapted from: 
[DeepLab-Resnet-Pytorch](https://github.com/speedinghzl/Pytorch-Deeplab), [AdvSemiSeg](https://github.com/hfslyc/AdvSemiSeg), [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)


## Citation
```
@article{Guo2022UTB,
      title={The Urban Tree Canopy Cover in Brazil - Nationwide Perspectives from High-Resolution Remote Sensing Images},
      author={Guo, Jianhua and Xu, Qingsong and Zeng, Yue and Liu, Zhiheng and Zhu, Xiaoxiang},
      journal={Remote Sensing of Environment},
      year={2022},
    }
```

