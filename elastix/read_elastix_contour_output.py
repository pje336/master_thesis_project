import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import compare_images

import json
from Evaluation.lap_model.slice_viewer_flow import slice_viewer
import torch


from Evaluation.lap_model.contour_viewer import contour_viewer

root_path_data = "C:/Users/pje33/Desktop/4d-lung/manifest-1665386976937/4D-Lung/"
with open(root_path_data + "scan_dictionary.json", 'r') as file:
    ct_path_dict = json.load(file)

patient_id = "102"
scan_id = "03-25-1998-NA-p4-57341"
m_phase = "20"
f_phase = "50"

output_folder = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/elastix/outputs/102/"
prediction_path = output_folder + '{}_to_{}/'.format(m_phase,f_phase)

# Read the full CT images
predicted_file = sitk.ReadImage(prediction_path+"/result.nii")
predicted_image = np.array(sitk.GetArrayFromImage(predicted_file))
predicted_image += 1000
predicted_image = np.divide(predicted_image,4000)

file_path = root_path_data + ct_path_dict[patient_id][scan_id][m_phase]
index = file_path[::-1].find("/")
file_path_f = file_path[:-index] + "/{}_256.nii".format(f_phase)
file_path_m = file_path[:-index] + "/{}_256.nii".format(m_phase)

m_files = sitk.ReadImage(file_path_m)
m_image = np.array(sitk.GetArrayFromImage(m_files))
m_image += 1000
m_image = np.divide(predicted_image,4000)
print(m_files.GetSize())


f_files = sitk.ReadImage(file_path_f)
f_image = np.array(sitk.GetArrayFromImage(f_files))
f_image += 1000
f_image = np.divide(predicted_image,4000)



# import the contours
roi_names = ["Rlung","Llung"]

combined_fixed_contour = np.zeros(( 80, 256, 256))
combined_moving_contour = np.zeros(( 80, 256, 256))
combined_warped_contour = np.zeros(( 80, 256, 256))

for roi_index, roi_name in enumerate(roi_names):
    file_path_m = file_path[:-index] + "contour_{}_c{}.nii".format(roi_name,m_phase)
    print(file_path_m)
    print(prediction_path)
    m_files = sitk.ReadImage(file_path_m)
    m_contour = np.array(sitk.GetArrayFromImage(m_files))
    combined_moving_contour += m_contour * (roi_index + 1)
    print(np.amax(m_contour, axis=(1, 2)))
    print("-->",m_files.GetSize())

    predicted_contour_file = sitk.ReadImage(prediction_path + "/{}".format(roi_name)+ "/result.nii")
    predicted_contour_image = np.array(sitk.GetArrayFromImage(predicted_contour_file))
    print(predicted_contour_image.shape)
    print(np.amax(predicted_contour_image, axis = (1,2)))
    combined_warped_contour += predicted_contour_image * (roi_index + 1)


plt.imshow(combined_warped_contour[-2])
plt.show()
print(np.amax(combined_warped_contour))
title = ["Moving image", "predicted image", "target image"]
contour_viewer([m_image[::-1], predicted_image[::-1], f_image[::-1],
                combined_moving_contour, combined_warped_contour, combined_fixed_contour], title,
               roi_names=roi_names)
# contour_viewer([combined_fixed_contour,combined_fixed_contour,combined_fixed_contour,
#                 combined_moving_contour, combined_warped_contour, combined_fixed_contour], title,
#                roi_names=roi_names)