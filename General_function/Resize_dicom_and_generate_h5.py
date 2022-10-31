"""
This script can be used to resize dicom files from the original size to a set size and store it as hdf5.
This is only done in x and y.
"""

import os
import json
import numpy as np
import pydicom
import h5py
from scipy.ndimage import zoom

import matplotlib.pyplot as plt

dimensions_resize = [256, 256]
desired_resolution = [3,0.9766*2, 0.9766*2]

root_path = "C://Users//pje33//Downloads//4D_CT_lyon_512//"
root_path_resize = "C://Users//pje33//Downloads//4D-Lung-{}-h5_resampled//".format(str(dimensions_resize[0]))

with open(root_path + "scan_dictionary.json", 'r') as file:
    ct_path_dict = json.load(file)

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
            ct_data = np.zeros((len(files), int(data.Rows), int(data.Columns)), dtype=np.int16)

            # Iterate over files in folder.
            for z, file in enumerate(files):
                if z == 0:

                    spacing = pydicom.dcmread(full_path + file).PixelSpacing
                    slice_thickness = pydicom.dcmread(full_path + file).SliceThickness
                    print(spacing, slice_thickness)
                ct_data[z, :, :] = pydicom.dcmread(full_path + file).pixel_array
            ct_data_zoomed = zoom(ct_data, (slice_thickness/desired_resolution[0], spacing[0]/desired_resolution[1], spacing[1]/desired_resolution[2]))

            zoomed_shape = np.shape(ct_data_zoomed)

            print(np.shape(ct_data_zoomed))
            if zoomed_shape[1] < dimensions_resize[0]: # if it is to small pad it
                print("padding")
                padding = dimensions_resize[0]  - zoomed_shape[1]
                ct_data_zoomed = np.pad(ct_data_zoomed,((0,0),(padding//2,padding//2 + padding%2),(padding//2,padding//2 + padding%2)))
            elif zoomed_shape[1] > dimensions_resize[0]:  # else crop
                print("cropping")
                index = zoomed_shape[1] - dimensions_resize[0]
                ct_data_zoomed = ct_data_zoomed[:,index//2:-index//2,index//2:-index//2]
            print(np.shape(ct_data_zoomed))

            print(np.shape(ct_data))
            f.create_dataset(phase, data = ct_data_zoomed)
        #





