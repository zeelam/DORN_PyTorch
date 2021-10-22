import torch
import torch.nn as nn
import numpy as np

class OrdinalRegressionLoss(nn.Module):

    def __init__(self, alpha=0, beta=80, K=80):
        super(OrdinalRegressionLoss, self).__init__()
        self.alpha = alpha
        self.eta = 1 - alpha
        self.beta = beta
        self.K = K

    # [alpha, beta] -> [alpha_star, beta_star] -> [0, K]
    def depthToLabel(self, depth):
        depth_ = depth + torch.tensor(self.eta, dtype=torch.float).to(depth.device)
        beta_ = self.beta + self.eta
        label = self.K * torch.log(depth_) / np.log(beta_)
        return label.round().long()

    # [0, K] -> [alpha_star, beta_star] -> [alpha, beta]
    def labelToDepth(self, label):
        alpha_ = self.alpha + self.eta
        beta_ = self.beta + self.eta
        K_ = self.K
        depth = torch.exp(np.log(alpha_) + np.log(beta_ / alpha_) * label / K_) - self.eta
        return depth

    def forward(self, pred, gt_depth):
        B, _, H, W = gt_depth.shape
        valid_mask = gt_depth > 0.
        valid_mask = valid_mask.view(B, H, W)
        gt_label = self.depthToLabel(gt_depth)
        before_k = torch.ones(B, self.K, H, W).to(gt_depth.device)
        mask = torch.linspace(0, self.K - 1, self.K, requires_grad=False).view(1, self.K, 1, 1).to(gt_depth.device)
        mask = mask.repeat(B, 1, H, W).contiguous().long()
        mask = (mask > gt_label).detach()
        before_k[mask] = 0
        after_k = 1 - before_k
        true_predict = (torch.log(pred.clamp(1e-8, 1e8)) * before_k)
        false_predict = (torch.log((1 - pred).clamp(1e-8, 1e8)) * after_k)
        loss = torch.sum(true_predict, dim=1) + torch.sum(false_predict, dim=1)
        return -loss[valid_mask].mean()
