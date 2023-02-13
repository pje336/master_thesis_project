"""
This script can be used to resize 4dct dicom files from the original size to a set size.
This is only done in 2d (x-y).
The resized scans are then stored as h5 files.
"""

import os
import json
import numpy as np
import pydicom
from scipy.ndimage import zoom
from skimage.transform import resize,downscale_local_mean

import matplotlib.pyplot as plt

dimensions_resize = [256, 256]

root_path = "C://Users//pje33//Downloads//4D_CT_lyon_512//"
root_path_resize = "C://Users//pje33//Downloads//4D_CT_lyon_{}_zoomed//".format(str(dimensions_resize[0]))




with open(root_path + "scan_dictionary.json", 'r') as file:
    ct_path_dict = json.load(file)

for patient_id in ct_path_dict.keys():
    for scan_id in ct_path_dict[patient_id].keys():
        for phase in ct_path_dict[patient_id][scan_id].keys():
            filepath = ct_path_dict[patient_id][scan_id][phase]
            # scan all files in the directory
            full_path, dirs, files = next(os.walk(root_path + filepath + "/"))

            # Make a new dirs in the root_path_256 folder
            dirName = root_path_resize + filepath
            if not os.path.exists(dirName):
                os.makedirs(dirName)
                print("Directory ", dirName, " Created ")
            else:
                print("Directory ", dirName, " already exists")

            # Iterate over files in folder.
            for z, file in enumerate(files):
                # open file and get pixel data.
                data_original = pydicom.dcmread(full_path + file)
                print(data_original)
                data_pixels = np.array(data_original.pixel_array, np.float)
                # resize the pixeldata and scale it to 16bit int.
                data_resized = np.array(downscale_local_mean(data_pixels,(2,2)),dtype=np.uint16)
                # Change the pixeldata to the resized one and update the shape and save it.
                data_original.PixelData = data_resized.tobytes()
                data_original.Rows, data_original.Columns = np.shape(data_resized)
                # save the file in the new directory.
                data_original.save_as(dirName + "/" + file)


