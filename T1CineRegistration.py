import SimpleITK as sitk
import nibabel as nib
import numpy as np
import os

## Data path
T1Address = '/Users/kurtlab/Desktop/Image_registration/ChiariSubj1/NIFTI/T1.nii'
BrainStemAddress = '/Users/kurtlab/Desktop/Image_registration/ChiariSubj1/NIFTI/brainstemSsLabels.v12.FSvoxelSpace.nii'
PCFAddress = '/Users/kurtlab/Desktop/Image_registration/ChiariSubj1/NIFTI/PCFExtractLabel.nii'
CineAddress ="/Users/kurtlab/Desktop/Image_registration/ChiariSubj1/CineAllTimestep/"
TimeSteps = 21

## Load T1 and Cine images
# T1 images: whole brain and PCF mask
T1image = sitk.ReadImage(T1Address)
BSimage = sitk.ReadImage(BrainStemAddress)
PCFimage = sitk.ReadImage(PCFAddress)
# Cine images (stacked all time points)
Time = 0
CineImageNP = nib.load(CineAddress+"WholeVolume_Time1.nii").get_fdata()
CineImageShape = CineImageNP.shape


## Registration
# Transformation method:
# Translation to Rigid (3D)
# Rotation to Rigid (3D)
# Rigid to similarity (3D)
# Similarity to Affine
# BSpline transformation
# Displacement Field

# Images and resampling:
# set different output directory

## Reading and Writing


