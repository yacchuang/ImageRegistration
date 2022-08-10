import SimpleITK as sitk
import nibabel as nib
import numpy as np
import os
from tkinter import tk
import gui
import registration_gui as rgui

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

T1imageNP = nib.load(T1Address).get_fdata()
T1imageShape = T1imageNP.shape


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
        
   
    
## Registration

fixed = sitk.ReadImage(CineAddress+"WholeVolume_Time1.nii", sitk.sitkFloat32)
moving = sitk.ReadImage(T1Address, sitk.sitkFloat32) 

# # View images on GUI
# window = tk.Tk()
# Cine_window = [80, 216]
# T1_window = [256, 256]

# gui.MultiImageDisplay(image_list = [fixed, moving],
#                       title_list = ['fixed', 'moving'], figure_size = (8, 4), window_level_list=[Cine_window, T1_window])


# Transformation method: Align the centers of the two volumes
initial_transform = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)

registration_method = sitk.ImageRegistrationMethod()
    
# Translation to Rigid (3D)
# Rotation to Rigid (3D)
# Rigid to similarity (3D)
# Similarity to Affine
# BSpline transformation
# Displacement Field

# Images and resampling:
# set different output directory

## Reading and Writing


