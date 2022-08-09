#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12  2022

@author: kurtlab
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import SimpleITK as sitk
import nibabel as nib
from LoadVisualNIFTI import read_img_nii, read_img_sitk, np_BrainImg, np_PFMaskImg
from volumentations import *
import numpy as np

patch_size = (128, 128, 128)   # Whole MRI image


def get_augmentation(patch_size):
    return Compose([
        RemoveEmptyBorder(always_apply=True),
        RandomScale((0.8, 1.2)),
        PadIfNeeded(patch_size, always_apply=True),
        # RandomCrop(patch_size, always_apply=True),
        # CenterCrop(patch_size, always_apply=True),
        # RandomCrop(patch_size, always_apply=True),
        Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
        CropNonEmptyMaskIfExists(patch_size, always_apply=True),
        Normalize(always_apply=True),
        # ElasticTransform((0, 0.25)),
        # Rotate((-15,15),(-15,15),(-15,15)),
        # Flip(0),
        # Flip(1),
        # Flip(2),
        # Transpose((1,0,2)), # only if patch.height = patch.width
        # RandomRotate90((0,1)),
        # RandomGamma(),
        # GaussianNoise(),
    ], p=1)


class BrainDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("T1_", "PFseg_"))  # Differences of filename between images and masks
        read_img_sitk(img_path)
        read_img_sitk(mask_path)
        BrainImg = read_img_nii(img_path)
        PFMaskImg = read_img_nii(mask_path)

        PFMask = sitk.GetImageFromArray(PFMaskImg)
        BrainT1 = sitk.GetImageFromArray(BrainImg)
        image = sitk.GetArrayFromImage(BrainT1)
        mask = sitk.GetArrayFromImage(PFMask)

        # image = np.array(Image.open(img_path).convert("RGB"))
        # mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0



        if self.transform is not None:
            aug = get_augmentation(patch_size)

            # with mask
            data = {'image': image, 'mask': mask}
            aug_data = aug(**data)
            image, mask = aug_data['image'], aug_data['mask']


        return image, mask
