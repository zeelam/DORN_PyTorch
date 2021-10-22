import torch
import numpy as np
import matplotlib.pyplot as plt


class ImageBuilder(object):
    """
    Builds an image iteratively row by row where the columns are (input image, target depth map, output depth map).
    """

    def __init__(self):
        self.count = 0
        self.img_merge = None

    def has_image(self):
        return self.img_merge is not None

    def get_image(self):
        return torch.from_numpy(np.transpose(self.img_merge, (2, 0, 1)) / 255.0)

    def add_row(self, input, target, depth):
        if self.count == 0:
            self.img_merge = self.merge_into_row(input, target, depth)
        else:
            row = self.merge_into_row(input, target, depth)
            self.img_merge = np.vstack([self.img_merge, row])

        self.count += 1

    @staticmethod
    def colored_depthmap(depth, d_min=None, d_max=None):
        if d_min is None:
            d_min = np.min(depth)
        if d_max is None:
            d_max = np.max(depth)
        depth_relative = (depth - d_min) / (d_max - d_min)
        return 255 * plt.cm.jet(depth_relative)[:, :, :3]  # H, W, C

    @staticmethod
    def merge_into_row(input, depth_target, depth_pred):
        rgb = np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
        depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
        depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

        d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
        d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
        depth_target_col = ImageBuilder.colored_depthmap(depth_target_cpu, d_min, d_max)
        depth_pred_col = ImageBuilder.colored_depthmap(depth_pred_cpu, d_min, d_max)
        img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])

        return img_merge