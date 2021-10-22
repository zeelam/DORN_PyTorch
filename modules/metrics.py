import torch
import math
import numpy as np


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)

class Result(object):
    def __init__(self):
        self.silog = 0
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.sqrel = 0, 0
        self.lg10, self.rmse_log = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0

        self.set_to_worst()

    def set_to_worst(self):
        self.silog = np.inf
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.sqrel = np.inf, np.inf
        self.lg10, self.rmse_log = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0

    def update(self, silog, irmse, imae, mse, rmse, mae, absrel, sqrel, lg10, rmse_log, delta1, delta2, delta3):
        self.silog = silog
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.sqrel = absrel, sqrel
        self.lg10, self.rmse_log = lg10, rmse_log
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3

    def evaluate(self, pred, gt, cap=None):
        valid_mask = gt > 0
        pred = pred[valid_mask]
        gt = gt[valid_mask]

        if cap != None:
            cap_mask = gt <= cap
            pred = pred[cap_mask]
            gt = gt[cap_mask]

        abs_diff = (gt - pred).abs()
        abs_diff_log = (torch.log(gt) - torch.log(pred)).abs()

        # Scale-Invariant Error
        di = (torch.log(pred) - torch.log(gt))
        self.silog = torch.pow(di, 2).mean() + torch.pow(di.mean(), 2)

        # mean square error
        self.mse = float((torch.pow(abs_diff, 2)).mean())
        # root mean square error
        self.rmse = math.sqrt(self.mse)
        # root mean square error (log)
        self.rmse_log = math.sqrt(float((torch.pow(abs_diff_log, 2)).mean()))

        # mean absolute error
        self.mae = float(abs_diff.mean())

        # average Lg10 error
        self.lg10 = float((log10(pred) - log10(gt)).abs().mean())

        # absolute relative error
        self.absrel = float((abs_diff / gt).mean())
        # square relative error
        self.sqrel = ((abs_diff ** 2) / gt).mean()

        # accuracy
        maxRatio = torch.max(pred / gt, gt / pred)
        self.delta1 = float((maxRatio < 1.25).float().mean())  # diff_ratio < 1.25
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())  # diff_ratio < 1.5625
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())  # diff_ratio < 1.953125

        inv_pred = 1 / pred
        inv_gt = 1 / gt
        abs_inv_diff = (inv_gt - inv_pred).abs()
        # inverse root mean square error
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        # inverse mean absolute error
        self.imae = float(abs_inv_diff.mean())

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_silog = 0
        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_sqrel = 0, 0
        self.sum_lg10, self.sum_rmse_log = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0

    def update(self, result, n=1):
        self.count += n

        self.sum_silog += n * result.silog
        self.sum_irmse += n * result.irmse
        self.sum_imae += n * result.imae
        self.sum_mse += n * result.mse
        self.sum_rmse += n * result.rmse
        self.sum_mae += n * result.mae
        self.sum_absrel += n * result.absrel
        self.sum_sqrel += n * result.sqrel
        self.sum_lg10 += n * result.lg10
        self.sum_rmse_log += n * result.rmse_log
        self.sum_delta1 += n * result.delta1
        self.sum_delta2 += n * result.delta2
        self.sum_delta3 += n * result.delta3

    def average(self):
        avg = Result()
        avg.update(
            self.sum_silog / self.count, self.sum_irmse / self.count, self.sum_imae / self.count,
            self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count,
            self.sum_absrel / self.count, self.sum_sqrel / self.count,
            self.sum_lg10 / self.count, self.sum_rmse_log / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count)
        return avg