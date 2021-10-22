import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalRegressionModule(nn.Module):

    def __init__(self):
        super(OrdinalRegressionModule, self).__init__()

    def forward(self, x):
        B, K, H, W = x.shape
        K = K // 2

        even = x[:, ::2, :, :].clone()
        odd = x[:, 1::2, :, :].clone()

        even = even.view(B, 1, K * H * W)
        odd = odd.view(B, 1, K * H * W)
        paired = torch.cat((even, odd), dim=1).clamp(1e-8, 1e8)
        prob = F.softmax(paired, dim=1)
        prob = prob[:, 1, :]
        prob = prob.view(-1, K, H, W)
        labels = torch.sum((prob >= 0.5), dim=1).view(-1, 1, H, W)
        return labels, prob
