import os
from random import randint

import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader


class ct_dataset(Dataset):
    """
    Dataset class with fixed and moving tensor pairs each of shape [1,z,x,y]
    """

    def __init__(self, root_path, ct_path_dict, scan_keys, dimensions, shift):
        """
        :param root_path: [string] Root path to 4DCT folders.
        :param ct_path_dict: [dict] dictionary with all file paths to the dicom files..
        :param scan_keys: [array]: array with keys for each scan. e.g: [[patient_id,scan_id,[f_phase,m_phase]],...]
        :param dimensions: [1d array] array with dimensions to crop the image [z_min,z_max,x_min,x_max,y_min,y_max]
        :param shift: [1d array] array with max up and down shift [x_down,x_up,y_down,y_up] (can be zeros)
        """
        self.root_path = root_path
        self.ct_path_dict = ct_path_dict
        self.scans_keys = scan_keys
        self.dimensions = dimensions
        self.shift = shift

    def __len__(self):
        return len(self.scans_keys)

    def __getitem__(self, index):
        patient_id, scan_id, phases = self.scans_keys[index]

        # Get scan data.
        _fixed = read_ct_data_file(self.root_path, self.ct_path_dict[patient_id][scan_id][phases[0]], self.dimensions)
        _moving = read_ct_data_file(self.root_path, self.ct_path_dict[patient_id][scan_id][phases[1]], self.dimensions)

        # Apply shift for data augmentation
        # determine the random shift in x and y directions.
        x_shift = randint(-self.shift[0], self.shift[1])
        y_shift = randint(-self.shift[2], self.shift[3])

        # This tuple is y,x because pad function is weird. See documentation of torch.nn.functional.pad
        pad = (abs(y_shift), abs(y_shift), abs(x_shift), abs(x_shift))

        # Apply padding, rolling and then cropping to get the shifted image.
        # Note: The dims in roll are switched again, it is all super vague why this is.
        _fixed = torch.nn.functional.pad(_fixed[0], pad).roll(shifts=(x_shift, y_shift), dims=(2, 1))[None, :,
                 pad[2]:self.dimensions[3] + pad[2],
                 pad[0]:self.dimensions[5] + pad[0]]
        _moving = torch.nn.functional.pad(_moving[0], pad).roll(shifts=(x_shift, y_shift), dims=(2, 1))[None, :,
                  pad[2]:self.dimensions[3] + pad[2],
                  pad[0]:self.dimensions[5] + pad[0]]

        return _fixed, _moving

    def shape(self):
        return [self.dimensions[1] - self.dimensions[0], self.dimensions[3] - self.dimensions[2],
                self.dimensions[5] - self.dimensions[4]]


def generate_dataset(scan_keys, root_path, ct_path_dict, dimensions, shift, batch_size, shuffle=True):
    """ Generates dataset objects with  batches pairs of fixed_tensor, moving_tensor based on scan_keys.
    It generates a ct_dataset instance and then makes a dataloader object with the given batch_size.
    The data pairs are read on the fly when iterating and therefore do not have high memory use.

    First checks if all the scans files from scan_keys are present, otherwise an error is raised.



    Args:
        scan_keys: [array]: array with keys for each scan. e.g: [[patient_id,scan_id,[f_phase,m_phase]],...]
        root_path: [string] Root path to 4DCT folders.
        ct_path_dict: [dict] dictionary with all file paths to the dicom files..
        dimensions: [1d array] array with dimensions to crop the image [z_min,z_max,x_min,x_max,y_min,y_max]
        shift: [1d array] array with max up and down shift [x_down,x_up,y_down,y_up] (can be zeros)
        batch_size: [int] number of samples in each batch
        shuffle:[Boolean] shuffle the samples when generating batches (True by default)

    Returns: A dataset from DataLoader-object which can be used to load batches with pairs of fixed_tensor, moving_tensor tensors.
            One could iterate over the dataset as the following:
            "for fixed_tensor, moving_tensor in  dataset: "
            Shapes:
            fixed_tensor: [Batch_size,1,z,x,y]
            moving_tensor: [Batch_size,1,z,x,y]



    Raises:
      ValueError: If the root_path doesn't end with a "/"
      ValueError: If the length of scan_keys is zero.
      ValueError: If none of the dicom files from scan_keys can be read
      ValueError: If some of the dicom files from scan_keys can be read

    """

    # Checking the root_path and the len of the scan_keys array.
    if root_path[-1] != "/":
        raise ValueError("root_path should end with '/'")
    if len(scan_keys) == 0:
        raise ValueError("Scan_keys has no objects.")

    # Checking if the files for scan_keys excist. Else, raise an ValueError
    keys_not_found = scan_key_checker(scan_keys, root_path, ct_path_dict)
    print(len(keys_not_found), len(scan_keys))
    if len(keys_not_found) == len(scan_keys):
        raise ValueError("None of the dicom"
                         " files found, is the root path correct?")
    elif len(keys_not_found) != 0:
        raise ValueError("The dicom files for the following keys where not found:", keys_not_found)

    data = ct_dataset(root_path, ct_path_dict, scan_keys, dimensions, shift)
    return DataLoader(data, batch_size, shuffle)


def read_ct_data_file(root_path, filepath, dimensions):
    """
    Imports all CT data by reading all dcm files in the given directory and normalises the data.
    :param root_path: [string] Root path to 4DCT folders.
    :param filepath: [string] String with the specific folder whichdat  holds CT data.
    :param dimensions: [1d array] array with dimensions to crop the image [z_min,z_max,x_min,x_max,y_min,y_max]
    :return: ct_data_tensor: [4d tensor] 4d pytorch tensor of shape [1,z,x,y] with normalised CT data.
    """
    # Get all files and paths.
    full_path, dirs, files = next(os.walk(root_path + filepath + "/"))
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


def scan_key_generator(dictionary, patient_ids=None, scan_list=None):
    """
    Generates an array keys for each scan. This can be the a specific scan, or a specific patient or all scans.
    :param dictionary: [dict] dictionary with all file paths to the dicom files..
    :param patient_ids: array with patients (If None, then all patients)
    :param scan_list: array with specific scan of patient (If None, then all scans of patient)
    :return scan_keys: [array]: array with keys for each scan. e.g: [[patient_id,scan_id,[f_phase,m_phase]],...]

    """
    scan_keys = []
    if patient_ids is None:
        patient_ids = dictionary.keys()

    for patient_id in patient_ids:
        if scan_list is None:
            scan_ids = dictionary[patient_id].keys()
        else:
            scan_ids = scan_list

        for scan_id in scan_ids:

            for m_phase in dictionary[patient_id][scan_id].keys():

                for f_phase in dictionary[patient_id][scan_id].keys():
                    scan_keys.append([patient_id, scan_id, [f_phase, m_phase]])
    return scan_keys


def scan_key_checker(scan_keys, root_path, dictionary):
    """Checks if the dicom files for each scan key exists.
    If it does not it will be added to keys_not_found not found array.


    Args:
        scan_keys: [array]: array with keys for each scan. e.g: [[patient_id,scan_id,[f_phase,m_phase]],...]
        root_path: [string] Root path to 4DCT folders.
        dictionary: [dict] dictionary with all file paths to the dicom files.*.

    Returns:
        keys_not_found: Array with scan keys of which the dicom files where not found.

    """
    keys_not_found = []

    for i, [patient, scan, phases] in enumerate(scan_keys):

        for phase in phases:
            filepath = dictionary[patient][scan][phase]

            try:
                next(os.walk(root_path + filepath + "/"))
            except:
                keys_not_found.append(scan_keys[i])
                break  # we do not need to check the other phase anymore

    return keys_not_found
