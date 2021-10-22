import os
import torchvision.transforms as tf
import torchvision.transforms.functional as TF
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class KittiDataset(Dataset):

    def __init__(self, data_path, gt_path, annotation_file_path, input_size=(385, 513)):
        super(KittiDataset, self).__init__()
        self.data_path = data_path
        self.gt_path = gt_path
        self.annotation_file_path = annotation_file_path
        self.input_size = input_size
        self.image_label_list = []
        with open(annotation_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.image_label_list.append(line.rstrip().split(" "))

    def __getitem__(self, index):
        i = index % len(self)
        image_file_path = self.image_label_list[i][0]
        label_file_path = self.image_label_list[i][1]

        image_file = os.path.join(self.data_path, image_file_path)
        label_file = os.path.join(self.gt_path, 'train', label_file_path)
        if not os.path.exists(label_file):
            label_file = os.path.join(self.gt_path, 'val', label_file_path)

        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        W = image.shape[1]
        image = cv2.resize(image, (W, 385), interpolation=cv2.INTER_LINEAR)
        image = np.float32(image)
        image_tensor = torch.from_numpy(image.transpose((2, 0, 1)))

        label = cv2.imread(label_file, cv2.IMREAD_UNCHANGED)
        label = label / 1000
        W = label.shape[1]
        label = cv2.resize(label, (W, 385), interpolation=cv2.INTER_LINEAR)
        label = np.float32(label)
        label = np.expand_dims(label, -1)
        label_tensor = torch.from_numpy(label.transpose((2, 0, 1)))

        i, j, h, w = tf.RandomCrop.get_params(image_tensor, output_size=self.input_size)
        crop_image = TF.crop(image_tensor, i, j, h, w)
        # cv2.imshow("crop_image", crop_image.numpy().astype(np.uint8).transpose((1, 2, 0)))
        crop_label = TF.crop(label_tensor, i, j, h, w)
        # cv2.imshow("crop_label", crop_label.numpy().astype(np.uint16).transpose((1, 2, 0)))
        # cv2.waitKey(0)
        return crop_image, crop_label

    def __len__(self):
        return len(self.image_label_list)