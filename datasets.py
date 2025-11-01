import torch
from albumentations import (ShiftScaleRotate, Compose, HorizontalFlip, VerticalFlip)
import numpy as np
from torch.utils.data import Dataset
import os
import random
import cv2

class Dataset_train(Dataset):
    def __init__(self, train_dir, sample_num_per_epoch):
        super(Dataset_train, self).__init__()
        self.train_dir = train_dir
        self.FB_name_list = os.listdir(os.path.join(train_dir, 'CBCT'))
        self.CT_name_list = os.listdir(os.path.join(train_dir, 'CT'))
        self.len = sample_num_per_epoch
        self.transforms = Compose([ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                                                    rotate_limit=30, p=0.3,
                                                    border_mode=cv2.BORDER_CONSTANT, value=0,
                                                    interpolation=cv2.INTER_NEAREST),
                                   HorizontalFlip(p=0.3), VerticalFlip(p=0.3)], p=0.5)

    def __getitem__(self, index):
        FB_name = random.sample(self.FB_name_list, 1)[0]
        CT_name = random.sample(self.CT_name_list, 1)[0]

        FB_img = np.load(os.path.join(self.train_dir, 'CBCT', FB_name))
        CT_img = np.load(os.path.join(self.train_dir, 'CT', CT_name))

        FB_CT_img = np.concatenate((FB_img[:, :, np.newaxis], CT_img[:, :, np.newaxis]), axis=2)
        FB_CT_img = self.transforms(image=FB_CT_img)['image']
        FB_img = FB_CT_img[:, :, 0]
        CT_img = FB_CT_img[:, :, 1]
        FB_img = torch.from_numpy(FB_img).float().unsqueeze(0)
        CT_img = torch.from_numpy(CT_img).float().unsqueeze(0)
        return FB_img, CT_img

    def __len__(self):
        return self.len

class Dataset_val(Dataset):
    def __init__(self, val_dir):
        super(Dataset_val, self).__init__()
        self.val_dir = val_dir
        self.name_list = os.listdir(os.path.join(val_dir, 'CBCT'))
        self.len = len(self.name_list)

    def __getitem__(self, index):
        name = self.name_list[index]

        FB_img = np.load(os.path.join(self.val_dir, 'CBCT', name))
        CT_img = np.load(os.path.join(self.val_dir, 'CT', name))
        mask = np.load(os.path.join(self.val_dir, 'body_mask', name))

        FB_img = torch.from_numpy(FB_img).float().unsqueeze(0)
        CT_img = torch.from_numpy(CT_img).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return FB_img, CT_img, mask

    def __len__(self):
        return self.len