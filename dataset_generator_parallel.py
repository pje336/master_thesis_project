import os

import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset


class ct_dataset(Dataset):
    """
    Dataset class with fixed and moving tensor pairs each of shape [1,z,x,y]
    """

    def __init__(self, root_path, ct_path_dict, scans_keys, dimensions):
        """
        :param root_path: [string] Root path to 4DCT folders.
        :param ct_path_dict: [dict] dictionary with all file paths to the CT data.
        :param scans_keys: [array]: array with keys for each scan. e.g: [[patient_id,scan_id,[f_phase,m_phase]],...]
        :param dimensions: [1d array] array with dimensions to crop the image [z_min,z_max,x_min,x_max,y_min,y_max]
        """
        self.root_path = root_path
        self.ct_path_dict = ct_path_dict
        self.scans_keys = scans_keys
        self.dimensions = dimensions

    def __len__(self):
        return len(self.scans_keys)

    def __getitem__(self, index):
        patient_id, scan_id, phases = self.scans_keys[index]
        _fixed = read_ct_data_file(self.root_path, self.ct_path_dict[patient_id][scan_id][phases[0]], self.dimensions)
        _moving = read_ct_data_file(self.root_path, self.ct_path_dict[patient_id][scan_id][phases[1]], self.dimensions)
        return _fixed, _moving

    def shape(self):
        return [self.dimensions[1] - self.dimensions[0], self.dimensions[3] - self.dimensions[2],
                self.dimensions[5] - self.dimensions[4]]


def read_ct_data_file(root_path, filepath, dimensions):
    """
    Imports all CT data by reading all dcm files in the given directory and normalises the data.
    :param root_path: [string] Root path to 4DCT folders.
    :param filepath: [string] String with the specific folder which holds CT data.
    :param dimensions: [1d array] array with dimensions to crop the image [z_min,z_max,x_min,x_max,y_min,y_max]
    :return: ct_data_tensor: [4d tensor] 4d pytorch tensor of shape [1,z,x,y] with normalised CT data.
    """
    # Get all files and paths.
    full_path, dirs, files = next(os.walk(root_path + filepath + "/"))
    # read a single file to get the dimensions.
    data = pydicom.dcmread(full_path + files[0])

    # Setup a empty array with correct shape (z,x,y)
    ct_data = np.zeros((int(data.ImagesInAcquisition), int(data.Rows), int(data.Columns)))

    # Iterate over files axis.
    for z, file in enumerate(files):
        ct_data[z, :, :] = pydicom.dcmread(full_path + file).pixel_array

    # normalise data
    ct_data_cropped = ct_data[dimensions[0]:dimensions[1], dimensions[2]:dimensions[3], dimensions[4]:dimensions[5]]
    ct_data_cropped /= np.max(ct_data_cropped)
    ct_data_tensor = torch.tensor(ct_data_cropped, dtype=torch.float)

    return ct_data_tensor[None, ...]
