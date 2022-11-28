import SimpleITK as sitk
from inspect import getmembers, isfunction
import json
import itk
import numpy as np
import matplotlib.pyplot as plt
import torch

def transform_flow_to_unit_flow(flow):
    z, x, y, _ = flow.shape
    flow[:, :, :, 0] = flow[:, :, :, 0] / (x-1)*2
    flow[:, :, :, 1] = flow[:, :, :, 1] / (y-1)*2
    flow[:, :, :, 2] = flow[:, :, :, 2] / (z-1)*2

    return flow


root_path_data = "C:/Users/pje33/Desktop/4d-lung/manifest-1665386976937/4D-Lung/"
with open(root_path_data + "scan_dictionary.json", 'r') as file:
    ct_path_dict = json.load(file)

patient_id = "101"
scan_id = "10-21-1997-NA-p4-86157"
m_phase = "90"
f_phase = "30"

output_folder = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/elastix/outputs/102/0_to_20"
prediction_path = output_folder + '/deformationField.nii'
transform_path = output_folder + '{}_to_{}/TransformParameters0.txt'.format(m_phase,f_phase)
predicted_file = sitk.ReadImage(prediction_path)
predicted_image = sitk.GetArrayFromImage(predicted_file)

# plt.imshow(predicted_image[0])
# plt.show()
# print(predicted_image)
# print(np.shape(predicted_image))
# print(np.amax(predicted_image))
# print(np.amin(predicted_image))
#
# unit_flow = transform_flow_to_unit_flow(torch.tensor(predicted_image))
# print(unit_flow.max())
# print(unit_flow.min())

print(predicted_file)

#
# file_path = root_path_data + ct_path_dict[patient_id][scan_id][m_phase]
#
# index = file_path[::-1].find("/")
# file_path_f = file_path[:-index] + "/{}_256.nii".format(f_phase)
# file_path_m = file_path[:-index] + "/{}_256.nii".format(m_phase)
#
#
#
# #
# m_files = sitk.ReadImage(file_path_m)
# m_image = sitk.GetArrayFromImage(m_files)
#
# transform = sitk.ReadTransform(transform_path)
# toDisplacementFilter  = sitk.TransformToDisplacementFieldFilter()
# toDisplacementFilter.SetReferenceImage(m_files)
# displacementField = toDisplacementFilter.Execute(transform)
# itk.transformread(transform_path)