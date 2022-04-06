import os

import numpy as np
import pydicom
from skimage.transform import resize

from ct_path_dict import ct_path_dict

root_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-512/"
root_path_256 = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-256/"
dimensions_original = [0, -1, 0, -1, 0, 0]
dimensions_resize = [256, 256]

for patient_id in ct_path_dict.keys():
    for scan_id in ct_path_dict[patient_id].keys():
        for phase in ct_path_dict[patient_id][scan_id].keys():
            filepath = ct_path_dict[patient_id][scan_id][phase]
            # scan all files in the directory
            full_path, dirs, files = next(os.walk(root_path + filepath + "/"))

            # Make a new dirs in the root_path_256 folder
            dirName = root_path_256 + filepath
            if not os.path.exists(dirName):
                os.makedirs(dirName)
                print("Directory ", dirName, " Created ")
            else:
                print("Directory ", dirName, " already exists")

            # Iterate over files in folder.
            for z, file in enumerate(files):
                # open file and get pixel data.
                data_original = pydicom.dcmread(full_path + file)
                data_pixels = data_original.pixel_array
                # resize the pixeldata and scale it to 16bit int.
                data_resized = np.array(65534 * resize(data_pixels, dimensions_resize, anti_aliasing=True),
                                        dtype=np.uint16)

                # Change the pixeldata to the resized one and update the shape and save it.
                data_original.PixelData = data_resized.tobytes()
                data_original.Rows, data_original.Columns = np.shape(data_resized)

                data_original.save_as(dirName + "/" + file)
