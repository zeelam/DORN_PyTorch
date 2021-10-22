import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, kernel_size, dilation=1):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(2048, 512, kernel_size, padding=dilation, dilation=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(512, 512, 1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


class FullImageEncoder(nn.Module):

    def __init__(self, h, w, kernel_size):
        super(FullImageEncoder, self).__init__()
        self.h = h
        self.w = w

        self.global_pooling = nn.AvgPool2d(kernel_size, stride=kernel_size, padding=kernel_size // 2, ceil_mode=True)
        self.dropout = nn.Dropout2d(p=0.5)
        self.h_ = self.h // kernel_size + 1
        self.w_ = self.w // kernel_size + 1
        self.global_fc = nn.Linear(2048 * self.h_ * self.w_, 512)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(512, 512, 1)

    def forward(self, x):
        x = self.global_pooling(x)
        x = self.dropout(x)
        x = x.view(-1, 2048 * self.h_ * self.w_)
        x = self.global_fc(x)
        x = self.relu(x)
        x = x.view(-1, 512, 1, 1)
        x = self.conv(x)
        x = self.relu(x)
        # upsampling
        x = F.interpolate(x, (self.h, self.w), mode='bilinear', align_corners=True)
        return x

class SceneUnderstandingModule(nn.Module):

    def __init__(self, K, size, kernel_size, pyramid=[6, 12, 18]):
        super(SceneUnderstandingModule, self).__init__()
        self.K = K
        self.size = size
        height, width = self.size

        self.encoder = FullImageEncoder(height // 8 + 1, width // 8 + 1, kernel_size)
        self.conv = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.aspp1 = ASPP(kernel_size=3, dilation=pyramid[0])
        self.aspp2 = ASPP(kernel_size=3, dilation=pyramid[1])
        self.aspp3 = ASPP(kernel_size=3, dilation=pyramid[2])
        self.concat_process = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(512 * 5, 2048, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(2048, self.K * 2, 1)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        # print("x1 shape: ", x1.shape)
        x2 = self.conv(x)
        # print("x2 shape: ", x2.shape)
        x3 = self.aspp1(x)
        # print("x3 shape: ", x3.shape)
        x4 = self.aspp2(x)
        # print("x4 shape: ", x4.shape)
        x5 = self.aspp3(x)
        # print("x5 shape: ", x5.shape)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.concat_process(x)
        x = F.interpolate(x, size=self.size, mode='bilinear', align_corners=True)
        return x
