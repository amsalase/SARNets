# coding:utf-8
# Email: 2443434059@qq.com

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from torch.nn import Softmax
from thop import profile


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, n_classes=12, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024):
        super().__init__()
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 512)
        self.up_2 = PSPUpsample(512, 256)
        self.up_3 = PSPUpsample(256, 128)
        self.up_4 = PSPUpsample(128, 64)
        self.up_5 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax()
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        f = x 
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)
        
        p = self.up_4(p)
        p = self.drop_2(p)

        p = self.up_5(p)
        p = self.drop_2(p)


        return self.final(p)


class PSPNet_Entire(nn.Module):

    def __init__(self, n_class):
        super(PSPNet_Entire, self).__init__()

        self.num_resnet_layers = 152

        if self.num_resnet_layers == 50:
            resnet_raw_model = models.resnet50(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 101:
            resnet_raw_model = models.resnet101(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 152:
            resnet_raw_model = models.resnet152(pretrained=True)
            self.inplanes = 2048
            
        ########  RGB ENCODER  ########
        
        self.encoder_rgb_conv1 = resnet_raw_model.conv1
        self.encoder_rgb_bn1 = resnet_raw_model.bn1
        self.encoder_rgb_relu = resnet_raw_model.relu
        self.encoder_rgb_maxpool = resnet_raw_model.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model.layer1
        self.encoder_rgb_layer2 = resnet_raw_model.layer2
        self.encoder_rgb_layer3 = resnet_raw_model.layer3
        self.encoder_rgb_layer4 = resnet_raw_model.layer4
        
        ########  DECODER  ########

        self.deconv = PSPNet(n_classes=12, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024)
         
    def forward(self, input):

        rgb = input        
        verbose = False

        # encoder

        ######################################################################
        if verbose: print("rgb.size() original: ", rgb.size())  # (480, 640)
        ######################################################################
        rgb = self.encoder_rgb_conv1(rgb)
        if verbose: print("rgb.size() after conv1: ", rgb.size())  # (240, 320)
        rgb = self.encoder_rgb_bn1(rgb)
        if verbose: print("rgb.size() after bn1: ", rgb.size())  # (240, 320)
        rgb = self.encoder_rgb_relu(rgb)
        if verbose: print("rgb.size() after relu: ", rgb.size())  # (240, 320)
        ######################################################################
        encoder_features = [rgb]
        ######################################################################
        rgb = self.encoder_rgb_maxpool(rgb)
        if verbose: print("rgb.size() after maxpool: ", rgb.size())  # (120, 160)
        rgb = self.encoder_rgb_layer1(rgb)
        encoder_features.append(rgb)
        if verbose: print("rgb.size() after layer1: ", rgb.size())  # (120, 160)
        rgb = self.encoder_rgb_layer2(rgb)
        encoder_features.append(rgb)
        if verbose: print("rgb.size() after layer2: ", rgb.size())  # (60, 80)
        rgb = self.encoder_rgb_layer3(rgb)
        encoder_features.append(rgb)
        if verbose: print("rgb.size() after layer3: ", rgb.size())  # (30, 40)
        rgb = self.encoder_rgb_layer4(rgb)
        encoder_features.append(rgb)
        if verbose: print("rgb.size() after layer4: ", rgb.size())  # (15, 20)
        # decoder
        deco = self.deconv(rgb)
        return deco

def unit_test():
    net = PSPNet_Entire(12).cuda(1)
    image = torch.randn(1, 3, 480, 640).cuda(1)
    with torch.no_grad():
        output = net.forward(image)
    flops, params = profile(net, inputs=(image, ))
    print(f"FLOPs: {flops}, Params: {params}")


unit_test()


