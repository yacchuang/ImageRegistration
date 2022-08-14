#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 15:13:54 2022

@author: kurtlab
"""

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import os
import cv2
import torch


## Data path
T1Address = '/Users/kurtlab/Desktop/Image_registration/ChiariSubj1/NIFTI/T1.nii'
BrainStemAddress = '/Users/kurtlab/Desktop/Image_registration/ChiariSubj1/NIFTI/brainstemSsLabels.v12.FSvoxelSpace.nii'
PCFAddress = '/Users/kurtlab/Desktop/Image_registration/ChiariSubj1/NIFTI/PCFExtractLabel.nii'
CineAddress ="/Users/kurtlab/Desktop/Image_registration/ChiariSubj1/CineAllTimestep/"
TimeSteps = 21

OUTPUT_DIR = '/Users/kurtlab/Desktop/Image_registration/ChiariSubj1/Registered/'


## Load T1 and Cine images
# T1 images: whole brain and PCF mask
T1image = sitk.ReadImage(T1Address)
BSimage = sitk.ReadImage(BrainStemAddress)
PCFimage = sitk.ReadImage(PCFAddress)

# Reorient T1 images
T1imageNP = nib.load(T1Address).get_fdata()
T1imageTensor = torch.from_numpy(T1imageNP)
T1imagePermute = torch.permute(T1imageTensor, (0, 2, 1))
T1imagePermuteNP = T1imagePermute.cpu().detach().numpy()
T1imageFinal = cv2.flip(T1imagePermuteNP, 1)
T1imageShape = T1imageFinal.shape

# Save Reoriented T1 to disk
T1_reoriented = nib.Nifti1Image(T1imageFinal, None)
nib.save(T1_reoriented, os.path.join(OUTPUT_DIR, "T1_reoriented.nii.gz"))

# Cine images (stacked all time points)
CineImageNP = nib.load(CineAddress+"WholeVolume_Time1.nii").get_fdata()
CineImageShape = CineImageNP.shape
CineFinalShape = np.append(CineImageShape, TimeSteps)
CineStacked = np.zeros(CineFinalShape)
CineData =os.listdir(CineAddress)
# print(CineData)

Time = 0

for EachCine in CineData:
    EachCineNP = nib.load(CineAddress+EachCine).get_fdata()
    CineStacked[:, :, :, Time] = EachCineNP
    Time = Time + 1
    
    
# Save stacked 4D Cine to disk
nifti = nib.Nifti1Image(CineStacked, None)
nib.save(nifti, os.path.join(CineAddress, "CineAllTime.nii.gz"))