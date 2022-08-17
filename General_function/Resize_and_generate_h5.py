"""
This script can be used to resize dicom files from the original size to a set size and store it as hdf5.
This is only done in x and y.
"""

import os

import numpy as np
import pydicom
import h5py

from CT_path_dict.ct_path_dict import ct_path_dict
import matplotlib.pyplot as plt

dimensions_resize = [256, 256]

root_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-256/"
root_path_resize = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-{}-h5/".format(str(dimensions_resize[0]))

for patient_id in ct_path_dict.keys():
    for scan_id in ct_path_dict[patient_id].keys():
        filepath_0 = ct_path_dict[patient_id][scan_id]["0"]
        filepath_scan = root_path_resize + filepath_0[:-filepath_0[::-1].find("/")]

        if not os.path.exists(filepath_scan):
            os.makedirs(filepath_scan)
            print("Directory ", filepath_scan, " Created ")
        else:
            print("Directory ", filepath_scan, " already exists")

        f = h5py.File(filepath_scan + "CT_dataset_all_phases.hdf5", "w")

        for phase in ct_path_dict[patient_id][scan_id].keys():
            filepath = ct_path_dict[patient_id][scan_id][phase]
            # scan all files in the directory
            full_path, dirs, files = next(os.walk(root_path + filepath + "/"))

            data = pydicom.dcmread(full_path + files[0])

            # Setup a empty array with correct shape (z,x,y)
            ct_data = np.zeros((int(data.ImagesInAcquisition), int(data.Rows), int(data.Columns)), dtype=np.int16)

            # Iterate over files in folder.
            for z, file in enumerate(files):
                ct_data[z, :, :] = pydicom.dcmread(full_path + file).pixel_array

            f.create_dataset(phase, data = ct_data)
        #





