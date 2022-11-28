import SimpleITK as sitk
import os
import json
import torch
import numpy as np
import h5py

patients = [100,101,102, 103, 104, 105, 106]
patients = [127,121,122, 123, 124, 125, 126]
phases = [0,10,20,30,40,50,60,70,80,90]
root_path_model = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/elastix/outputs/"
# root_path_CT_data = "C:/Users/pje33/Desktop/4d-lung/manifest-1665386976937/4D-Lung/"
root_path_CT_data = "C:/Users/pje33/Downloads/4D_CT_lyon_512/"

root_path_data = root_path_CT_data
root_path_contour = root_path_CT_data
with open(root_path_data + "scan_dictionary.json", 'r') as file:
    ct_path_dict = json.load(file)
# with open(root_path_contour + "contour_dictionary.json", 'r') as file:
#     contour_dict = json.load(file)


for patient_id in patients:
    # output_folder = root_path_model + "/{}/".format(patient_id)
    # print(output_folder)

    for scan_id in ct_path_dict[str(patient_id)].keys():
        file_path = root_path_data + ct_path_dict[str(patient_id)][scan_id]["0"]
        index = file_path[::-1].find("/")
        # f = h5py.File(file_path[:-index]  + "CT_dataset_all_phases.hdf5", "w")

        for f_phase in phases:

            # Load the fixed image
            file_path = root_path_data + ct_path_dict[str(patient_id)][scan_id][str(f_phase)]
            index = file_path[::-1].find("/")
            path_fixed_tensor = file_path[:-index] + "{}_256.nii".format(f_phase)
            print(path_fixed_tensor)
            print(file_path[:-index])
            fixed_file = sitk.ReadImage(path_fixed_tensor)
            fixed_image = np.array(sitk.GetArrayFromImage(fixed_file))
            print(fixed_image.shape)
            fixed_image = np.add(fixed_image, 1000)
            # f.create_dataset(str(f_phase), data = fixed_image)
