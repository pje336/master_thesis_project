import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import compare_images

import json
from Evaluation.lap_model.slice_viewer_flow import slice_viewer



jac_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/elastix/outputs/121/0_to_20/"






jac = sitk.ReadImage(jac_path + "spatialJacobian.nii")
jac = sitk.GetArrayFromImage(jac)

print(np.shape(jac))
print(jac[40,128,128])
print(jac)

print(np.size(jac[jac < 0]) / np.size(jac))
