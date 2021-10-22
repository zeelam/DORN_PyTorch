import torch
import torch.nn as nn
from modules.backbones.ResNet import ResNetBackbone
from modules.SceneUnderstandingModule import SceneUnderstandingModule
from modules.OrdinalRegressionModule import OrdinalRegressionModule


class Dorn(nn.Module):

    def __init__(self, K=80, input_size=(385, 513), kernel_size=16, pyramid=[8, 12, 16], pretrained=True,
                 pretrained_model_path=''):
        super(Dorn, self).__init__()
        self.pretrained = pretrained

        self.K = K

        self.backbone = ResNetBackbone(pretrained=pretrained, pretrained_model_path=pretrained_model_path)
        self.sceneUnderstandingModule = SceneUnderstandingModule(K, size=input_size, kernel_size=kernel_size,
                                                                 pyramid=pyramid)
        self.ordinalRegressionModule = OrdinalRegressionModule()

    def forward(self, x):
        x = self.backbone(x)
        x = self.sceneUnderstandingModule(x)
        labels, prob = self.ordinalRegressionModule(x)
        return labels, prob

    def train(self, mode=True):
        super().train(mode)
        if self.pretrained:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
            self.backbone.backbone.conv1.eval()
            self.backbone.backbone.conv2.eval()

        return self

    def get_1x_lr_params(self):
        for k in self.backbone.parameters():
            if k.requires_grad:
                yield k

    def get_10x_lr_params(self):
        for module in [self.sceneUnderstandingModule, self.ordinalRegressionModule]:
            for k in module.parameters():
                if k.requires_grad:
                    yield k
