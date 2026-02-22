# coding:utf-8
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from torch.nn import Softmax
from thop import profile


class DeepLab(nn.Module):
    def __init__(self, n_class):
        super(DeepLab, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    
    def forward(self, input):

        rgb = input
        verbose = False
        if verbose: print("rgb.size() original: ", rgb.size())  # (480, 640)
        ######################################################################
        output = self.model(rgb)
        return output


def unit_test():
    net = DeepLab(12).cuda(1)
    image = torch.randn(5, 3, 224, 224).cuda(1)
    with torch.no_grad():
        output = net.forward(image)
    flops, params = profile(net, inputs=(image, ))
    print(f"FLOPs: {flops}, Params: {params}")


unit_test()


