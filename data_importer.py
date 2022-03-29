import os

import numpy as np
import pydicom


def import_CT_data(path_drive, path_CT):
    """

    :param path_drive: string with path to folders.
    :param path_CT: string with the specific folder which holds CT data.
    :return: CT_data 3d numpy array of shape [z,x,y]
    """
    # Get all files and paths.
    full_path, dirs, files = next(os.walk(path_drive + path_CT))
    # read a single file to get the dimensions.
    data = pydicom.dcmread(full_path + files[0])

    # Setup a empty array with correct shape (z,x,y)
    CT_data = np.zeros((int(data.ImagesInAcquisition), int(data.Rows), int(data.Columns)))

    # Iterate over files axis.
    for z, file in enumerate(files):
        CT_data[z, :, :] = pydicom.dcmread(full_path + file).pixel_array

    # normalise data
    CT_data /= np.max(CT_data)
    return CT_data
