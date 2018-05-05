import torch
from torch.autograd import Variable

from resnet_unet import ModelBuilder, SegmentationModule
from visualize import  make_dot

x = Variable(torch.randn(1,3,704,704))#change 12 to the channel number of network input
builder = ModelBuilder()
net_encoder = builder.build_encoder(
        arch='resnet50')
net_decoder = builder.build_decoder(
        arch='c2_bilinear',
        num_class=1)
model = SegmentationModule(
            net_encoder, net_decoder)
y = model(x)
g = make_dot(y)
g.view()