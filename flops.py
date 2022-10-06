
"""
    使用 Flops 工具计算模型的计算参数量和计算量。
    author:czjing
    source: https://github.com/Lyken17/pytorch-OpCounter
    
    # pip install thop
"""

import torch

from thop import profile

# from   torchvision.models import resnet50
# model = resnet50()

from deeplabv2 import Res_Deeplab

from discriminator import s4GAN_discriminator11, FCDiscriminator, s4GAN_discriminator_DAN

from deeplabv2_unsupervised import Split_feature_discriminator_0


net =Split_feature_discriminator_0(layer_name='layer4')
input = torch.randn(1, 64*4*8, 321, 321)

# net =Split_feature_discriminator_0(layer_name='layer3')
# input = torch.randn(1, 64*4*4, 321, 321)




# net = FCDiscriminator() 
# input = torch.randn(1, 2, 321, 321)

# net = s4GAN_discriminator11() 
# input = torch.randn(1, 6,321, 321)


# net = s4GAN_discriminator_DAN()
# input = torch.randn(1, 5, 321, 321)

# net = Res_Deeplab() 
# input = torch.randn(1, 3, 321, 321)

flops,   params = profile(net, inputs=(input, ))
print('flops:', flops)
print('params:', params)