import os

import numpy as np
import pydicom


def import_CT_data(path_drive, path_CT):
    """

    :param path_drive: string with path to folders.
    :param path_CT: string with the specific folder which holds CT data.
    :return: CT_data 3d numpy array of shape [z,x,y]
    """
    full_path, dirs, files = next(os.walk(path_drive + path_CT))

    data = pydicom.dcmread(full_path + files[0])
    CT_data = np.zeros((int(data.ImagesInAcquisition), int(data.Rows), int(data.Columns)))

    for i, file in enumerate(files):
        CT_data[i, :, :] = pydicom.dcmread(full_path + file).pixel_array
    return CT_data
