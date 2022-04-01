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

        try:  # try to retrieve the filepath from dict.
            ct_filepath = ct_path_dict[str(patient_id)][scan_id][str(phase)]
        except KeyError:
            print("!Error: Key not found in the filepath dictionary.")
            print("key: [{}][{}][{}]".format(str(patient_id), scan_id, str(phase)))
            return

        if i == 0:  # make the initial array on the first run.
            ct_data = np.zeros(((len(phases),) + np.shape(read_CT_data_file(root_path, ct_filepath))))

        ct_data[i] = read_CT_data_file(root_path, ct_filepath)

    return ct_data


def dataset_generator(patient_id, scan_id, m_phases, f_phases, root_path, ct_path_dict, relative_size, pad):
    """
    Generates dataset object with pairs of CT data of moving images and fixed image.
    Applies padding or removes some slices from the z-axis
    :param patient_id: [int] patient number
    :param scan_id: [string] ID of the scan
    :param m_phases: [array of ints] array with the phases to put in the array
    :param f_phases: [int] fixed phase
    :param root_path: [string] root path to 4DCT folders.
    :param ct_path_dict: [dict] dictionary with all filepaths to the CT data.
    :param relative_size: [int] max z value for the CT volume. (Change this)
    :param pad: [boolean] Add padding instead of reducing the size.
    :return ct_dataset: return a dataset class with fixed and moving tensor pairs.
    """

    if len(m_phases) != len(f_phases):
        raise Exception("Length of m_phases array does not match f_phases array.")

    moving_tensor = torch.tensor(CT_dataset(patient_id, scan_id, m_phases, root_path, ct_path_dict),
                                 dtype=torch.float)
    fixed_tensor = torch.tensor(CT_dataset(patient_id, scan_id, f_phases, root_path, ct_path_dict),
                                dtype=torch.float)

    # Adjust the size of the z-axis.
    if fixed_tensor.shape[1] % (2 ** relative_size) == 0:
        # if the z-axis length is a multiple of the relative_size
        # return it as a dataset.
        return ct_dataset(fixed_tensor[:, None, ...], moving_tensor[:, None, ...])

    elif fixed_tensor.shape[1] < (2 ** relative_size) or pad:
        # if the z-axis length is smaller than the relative volume
        # or when 'pad' is true. apply padding.
        print("Applying padding for  the Z-axis")

        if fixed_tensor.shape[1] < (2 ** relative_size):  # if the z-axis is smaller than relative volume
            under_size = (2 ** relative_size) - fixed_tensor.shape[1]
        else:
            new_size = (fixed_tensor.shape[1] // (2 ** relative_size) + 1) * (2 ** relative_size)
            under_size = new_size - fixed_tensor.shape[1]

        if under_size % 2 != 0:  # for odd under_size
            under_size += 1
            odd = True
        else:
            odd = False

        # Apply padding
        moving_tensor = torch.nn.functional.pad(moving_tensor, (0, 0, 0, 0, under_size // 2, under_size // 2))
        fixed_tensor = torch.nn.functional.pad(fixed_tensor, (0, 0, 0, 0, under_size // 2, under_size // 2))

        if odd:  # for odd under_size, remove the last slice of zeros
            return ct_dataset(fixed_tensor[:, None, :-1, ...],
                              moving_tensor[:, None, :-1, ...])
        else:
            return ct_dataset(fixed_tensor[:, None, ...],
                              moving_tensor[:, None, ...])

    else:
        print("reducing the size in the Z-axis")
        oversize = fixed_tensor.shape[1] % (2 ** relative_size)
        # Add additional axis to tensor, reduce the z-axis  and make a dataset object from ct_dataset class.
        return ct_dataset(fixed_tensor[:, None, oversize // 2:-(oversize - oversize // 2), ...],
                          moving_tensor[:, None, oversize // 2:-(oversize - oversize // 2), ...])
