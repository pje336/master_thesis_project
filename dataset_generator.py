import os

import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset


class ct_dataset(Dataset):
    """
    Dataset class with fixed and moving tensor pairs of shape [len(phases),1,z,x,y]
    """

    def __init__(self, fixed, moving):
        self.fixed = fixed
        self.moving = moving

    def __len__(self):
        return len(self.fixed)

    def __getitem__(self, index):
        _fixed = self.fixed[index]
        _moving = self.moving[index]

        return _fixed, _moving

    def shape(self):
        return self.fixed.shape


def read_CT_data_file(root_path, filepath):
    """
    Imports all CT data by reading all dcm files in the given directory and normalises the data.
    :param root_path: [string] Root path to 4DCT folders.
    :param filepath: [string] String with the specific folder which holds CT data.
    :return: CT_data: [3d array] 3d numpy array of shape [z,x,y] with normalised CT data.
    """
    # Get all files and paths.
    full_path, dirs, files = next(os.walk(root_path + filepath + "/"))
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


def CT_dataset(patient_id, scan_id, phases, root_path, ct_path_dict):
    """
    Generates a dataset of 4d array of CT data for a specific patient_id and scan_id for given phases.
    It return a 4d array with shape [len(phases),z,x,y].
    :param patient_id: [int] patient number
    :param scan_id: [string] ID of the scan
    :param phases: [array of ints] array with the phases to put in the array
    :param root_path: [string] root path to 4DCT folders.
    :param ct_path_dict: [dict] dictionary with all filepaths to the CT data.
    :return: ct_data: array[len(phases),z,x,y] array with CT data for all the phases.
    """
    for i, phase in enumerate(phases):

        try:  # try to tetrieve the filepath from dict.
            ct_filepath = ct_path_dict[str(patient_id)][scan_id][str(phase)]
        except KeyError:
            print("!Error: Key not found in the filepath dictionary.")
            print("key: [{}][{}][{}]".format(str(patient_id), scan_id, str(phase)))
            return

        if i == 0:  # make the initial array on the first run.
            ct_data = np.zeros(((len(phases),) + np.shape(read_CT_data_file(root_path, ct_filepath))))

        ct_data[i] = read_CT_data_file(root_path, ct_filepath)

    return ct_data


def dataset_generator(patient_id, scan_id, m_phases, f_phases, root_path, ct_path_dict, z_max):
    """
    Generates dataset object with pairs of CT data of moving images and fixed image.
    :param patient_id: [int] patient number
    :param scan_id: [string] ID of the scan
    :param m_phases: [array of ints] array with the phases to put in the array
    :param f_phases: [int] fixed phase
    :param root_path: [string] root path to 4DCT folders.
    :param ct_path_dict: [dict] dictionary with all filepaths to the CT data.
    :param z_max: [int] max z value for the CT volume. (Change this)
    :return ct_dataset: return a dataset class with fixed and moving tensor pairs.
    """

    if len(m_phases) != len(f_phases):
        raise Exception("Length of m_phases array does not match f_phases array.")

    moving_tensor = torch.tensor(CT_dataset(patient_id, scan_id, m_phases, root_path, ct_path_dict)[:, :z_max, ...],
                                 dtype=torch.float)
    fixed_tensor = torch.tensor(CT_dataset(patient_id, scan_id, f_phases, root_path, ct_path_dict)[:, :z_max, ...],
                                dtype=torch.float)

    # Add additional axis to tensor, and make a dataset object from ct_dataset class.
    return ct_dataset(fixed_tensor[:, None, ...], moving_tensor[:, None, ...])
