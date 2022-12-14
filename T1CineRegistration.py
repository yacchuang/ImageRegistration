import SimpleITK as sitk
import os
from volumentations import *
# import gui
# import registration_gui as rgui

## Data path
T1Address = '/Users/kurtlab/Desktop/Image_registration/ChiariSubj1/NIFTI/T1_reoriented.nii'
BrainStemAddress = '/Users/kurtlab/Desktop/Image_registration/ChiariSubj1/NIFTI/brainstemSsLabels.v12.FSvoxelSpace.nii'
PCFAddress = '/Users/kurtlab/Desktop/Image_registration/ChiariSubj1/NIFTI/PCFExtractLabel.nii'
CineAddress ="/Users/kurtlab/Desktop/Image_registration/ChiariSubj1/CineAllTimestep/"
TimeSteps = 21

OUTPUT_DIR = '/Users/kurtlab/Desktop/Image_registration/ChiariSubj1/RegisteredITK/'


   
## Registration

# View images on GUI
# window = tk.Tk()
# Cine_window = [80, 216]
# T1_window = [256, 256]

# gui.MultiImageDisplay(image_list = [fixed, moving],
#                       title_list = ['fixed', 'moving'], figure_size = (8, 4), window_level_list=[Cine_window, T1_window])


fixed = sitk.ReadImage(CineAddress+"WholeVolume_Time1.nii", sitk.sitkFloat32)
moving = sitk.ReadImage(T1Address, sitk.sitkFloat32) 

# Transformation method: Align the centers of the two volumes
initial_transform = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)

registration_method = sitk.ImageRegistrationMethod()
# Similarity metric settings.
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)

registration_method.SetInterpolator(sitk.sitkLinear)

# Optimizer settings.
registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
registration_method.SetOptimizerScalesFromPhysicalShift()

# Setup for the multi-resolution framework.            
registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# Set the initial moving and optimized transforms.
optimized_transform = sitk.Euler3DTransform()    
registration_method.SetMovingInitialTransform(initial_transform)
registration_method.SetInitialTransform(optimized_transform, inPlace=False)

# # Don't optimize in-place, we would possibly like to run this cell multiple times.
# registration_method.SetInitialTransform(initial_transform, inPlace=False)

# # Connect all of the observers so that we can perform plotting during registration.
# registration_method.AddCommand(sitk.sitkStartEvent, rgui.start_plot)
# registration_method.AddCommand(sitk.sitkEndEvent, rgui.end_plot)
# registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, rgui.update_multires_iterations) 
# registration_method.AddCommand(sitk.sitkIterationEvent, lambda: rgui.plot_values(registration_method))

final_transform_v4 = sitk.CompositeTransform([registration_method.Execute(fixed, moving), initial_transform])   # ITKv4 Coordinate Systems
# final_transform = registration_method.Execute(fixed, moving)   # classic registration doesn't work

# Always check the reason optimization terminated.
print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    

## Reading and Writing
moving_resampled = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
sitk.WriteImage(moving_resampled, os.path.join(OUTPUT_DIR, 'T1_resampled.nii'))
sitk.WriteTransform(final_transform, os.path.join(OUTPUT_DIR, 'Cine_2_mr_T1.tfm'))


