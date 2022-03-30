import os

import numpy as np
import pydicom
import torch


def import_CT_data(root_path, filepath):
    """
    Imports all CT data from dcm files in the given directory and normalises the data.
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
            ct_data = np.zeros(((len(phases),) + np.shape(import_CT_data(root_path, ct_filepath))))

        ct_data[i] = import_CT_data(root_path, ct_filepath)
    return ct_data


def dataset_generator(patient_id, scan_id, m_phases, f_phase, root_path, ct_path_dict):
    """
    Generates tensors with CT data of moving images and fixed image. 
    Tensors have shape [len(m_phases),1,z,x,y]

    :param patient_id: [int] patient number
    :param scan_id: [string] ID of the scan
    :param m_phases: [array of ints] array with the phases to put in the array
    :param f_phase: [int] fixed phase
    :param phases: [array of ints] array with the phases to put in the array
    :param root_path: [string] root path to 4DCT folders.
    :param ct_path_dict: [dict] dictionary with all filepaths to the CT data.

    :return moving_tensor: torch.tensor shape [len(m_phases),1,z,x,y] tensor with CT data of all moving images.
    :return fixed_tensor: torch.tensor shape [len(m_phases),1,z,x,y] tensor with repeated CT data of fixed image.
    """

    moving_ct_data = CT_dataset(patient_id, scan_id, m_phases, root_path, ct_path_dict)
    # repeat fixed image data to match the moving image array shape.
    fixed_ct_data = np.repeat(CT_dataset(patient_id, scan_id, [f_phase], root_path, ct_path_dict), len(m_phases),
                              axis=0)

    # transform to torch.tensor
    moving_tensor = torch.tensor(np.expand_dims(moving_ct_data, axis=1), dtype=torch.float)
    fixed_tensor = torch.tensor(np.expand_dims(fixed_ct_data, axis=1), dtype=torch.float)

    return moving_tensor, fixed_tensor
