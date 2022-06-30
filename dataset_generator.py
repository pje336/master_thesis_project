import os
from random import randint

import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
import h5py

class SampleDataset(Dataset):
    """Dataset class with fixed and moving tensor pairs each of shape [1,z,x,y]. Inherited from torch.utils.data.Dataset
        Generates a dataset class with fixed and moving tensor pairs.
        The images can be augmented via a random shift in x and y direction by the shift attribute.


       Attributes:
            root_path: [string] Root path to 4DCT folders.
            ct_path_dict: [dict] dictionary with all file paths to the dicom files..
            scan_keys: [array]: array with keys for each scan. e.g: [[patient_id,scan_id,[f_phase,m_phase]],...]
            dimensions: [1d array] array with dimensions to crop the image [z_min,z_max,x_min,x_max,y_min,y_max]
            shift: [1d array] array with max up and down shift [x_down,x_up,y_down,y_up] (can be zeros)
    """

    def __init__(self, root_path, ct_path_dict, scan_keys, dimensions, shift):
        self.root_path = root_path
        self.ct_path_dict = ct_path_dict
        self.scans_keys = scan_keys
        self.dimensions = dimensions
        self.shift = shift

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.scans_keys)

    def __getitem__(self, index):
        """Gets the fixed and moving tensor by reading the dicom files and applying data augmention"""
        patient_id, scan_id, phases = self.scans_keys[index]

        # Get scan data using dicom.
        # _fixed = read_ct_data_file_dicom(self.root_path, self.ct_path_dict[patient_id][scan_id][phases[0]], self.dimensions)
        # _moving = read_ct_data_file_dicom(self.root_path, self.ct_path_dict[patient_id][scan_id][phases[1]], self.dimensions)

        _fixed = read_ct_data_file_h5(self.root_path, self.ct_path_dict[patient_id][scan_id], phases[0], self.dimensions)
        _moving = read_ct_data_file_h5(self.root_path, self.ct_path_dict[patient_id][scan_id], phases[1], self.dimensions)



        # Apply shift for data augmentation
        # determine the random shift in x and y directions.
        x_shift = randint(-self.shift[0], self.shift[1])
        y_shift = randint(-self.shift[2], self.shift[3])

        if x_shift == 0 and y_shift == 0:  # if it doesn't move return tensors immediately.
            return _fixed, _moving, [self.scans_keys[index]]

        # This tuple is y,x because pad function is weird. See documentation of torch.nn.functional.pad
        pad = [abs(y_shift), abs(y_shift), abs(x_shift), abs(x_shift)]

        # Apply padding, rolling and then cropping to get the shifted image.
        # Note: The dims in roll are switched again, it is all super vague why this is.

        _fixed = torch.nn.functional.pad(_fixed[0], pad).roll(shifts=(x_shift, y_shift), dims=(2, 1))
        _moving = torch.nn.functional.pad(_moving[0], pad).roll(shifts=(x_shift, y_shift), dims=(2, 1))

        # if one of the padding is zero, then replace the value with None for correct slicing of array.
        if pad[0] == 0:
            return _fixed[None, :, pad[2]:-pad[2]], \
                   _moving[None, :, pad[2]:-pad[2]], \
                   [self.scans_keys[index]]
        if pad[2] == 0:
            return _fixed[None, :, :, pad[0]:-pad[0]], \
                   _moving[None, :, :, pad[0]:-pad[0]], \
                   [self.scans_keys[index]]

        return _fixed[None, :, pad[2]:-pad[2], pad[0]:-pad[0]], \
               _moving[None, :, pad[2]:-pad[2], pad[0]:-pad[0]], \
               [self.scans_keys[index]]

    def shape(self):
        """ Return the shape of the fixed/moving tensor. """
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
    keys_not_found, keys_to_short = scan_key_checker(scan_keys, root_path, ct_path_dict, dimensions[1])

    if len(keys_not_found) == len(scan_keys):
        raise ValueError("None of the dicom"
                         " files found, is the root path correct?")
    elif len(keys_not_found) != 0:
        raise ValueError("The dicom files for the following keys where not found:", keys_not_found)

    # remove keys which do not have enough slices in z direction.
    for key in keys_to_short:
        index = scan_keys.index(key)
        scan_keys.pop(index)

    data = SampleDataset(root_path, ct_path_dict, scan_keys, dimensions, shift)
    return DataLoader(data, batch_size, shuffle)


def read_ct_data_file_dicom(root_path, filepath, dimensions):
    """Imports  CT data by reading all dicom files in the given directory and normalises the data.


    Args:
        root_path: [string] Root path to 4DCT folders.
        filepath: [string] String with the specific folder whichdat  holds CT data.
        dimensions: [1d array] array with dimensions to crop the image [z_min,z_max,x_min,x_max,y_min,y_max]

    Returns:
        ct_data_tensor: [4d tensor] 4d pytorch tensor of shape [1,z,x,y] with normalised CT data.

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
    ct_data_cropped /= 4000
    ct_data_tensor = torch.tensor(ct_data_cropped, dtype=torch.float)

    return ct_data_tensor[None, ...]



def read_ct_data_file_h5(root_path, filepath, phase, dimensions):
    """Imports  CT data by reading all dicom files in the given directory and normalises the data.


    Args:
        root_path: [string] Root path to 4DCT folders.
        filepath: [string] String with the specific folder whichdat  holds CT data.
        phase: [string] string with the phase to read
        dimensions: [1d array] array with dimensions to crop the image [z_min,z_max,x_min,x_max,y_min,y_max]


    Returns:
        ct_data_tensor: [4d tensor] 4d pytorch tensor of shape [1,z,x,y] with normalised CT data.

    """
    # Get all files and paths.
    dataset = h5py.File(root_path + filepath + "/CT_dataset_all_phases.hdf5", "r")
    ct_data_cropped = dataset[phase][dimensions[0]:dimensions[1], dimensions[2]:dimensions[3], dimensions[4]:dimensions[5]]

    # normalise data
    ct_data_tensor = torch.tensor(ct_data_cropped, dtype=torch.float)
    ct_data_tensor /= 4000

    return ct_data_tensor[None, ...]

def scan_key_generator(dictionary, patient_ids=None, scan_list=None):
    """Generates an array with arrays containing keys for the dictionary.
    One could get the keys for all scans in the dictionary or for a specific patient or a specific scan of a patient.

    Args:
        dictionary: [dict] dictionary with all file paths to the dicom files..
        patient_ids: array with patients (If None, then all patients)
        scan_list: array with specific scan of patient (If None, then all scans of patient)

    Returns:
            scan_keys: [array]: array with arrays of keys for each scan. e.g: [[patient_id,scan_id,[f_phase,m_phase]],...]

    """
    phases = ["0","10","20","30","40","50","60","70","80","90"]
    scan_keys = []
    if patient_ids is None:
        patient_ids = dictionary.keys()

    for patient_id in patient_ids:
        if scan_list is None:
            scan_ids = dictionary[patient_id].keys()
        else:
            scan_ids = scan_list

        for scan_id in scan_ids:

            for m_phase in phases:

                for f_phase in phases:
                    scan_keys.append([patient_id, scan_id, [m_phase, f_phase]])
    return scan_keys


def scan_key_checker(scan_keys, root_path, dictionary, min_z_length):
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
    keys_to_short = []

    for i, [patient, scan, phases] in enumerate(scan_keys):
        filepath = dictionary[patient][scan]

        try:
            dataset = h5py.File(root_path + filepath + "/CT_dataset_all_phases.hdf5", "r")
            dataset

            # check if there are enough slices in the z-direction.
            if dataset["0"].shape[0] < min_z_length:
                keys_to_short.append(scan_keys[i])
                break  # we do not need to check the other phase anymore

        except:
            keys_not_found.append(scan_keys[i])
            break  # we do not need to check the other phase anymore


    return keys_not_found, keys_to_short
