# This file was taken and adapted from the github repository https://github.com/KeremTurgutlu/dicom-contour


import math
import operator
import os
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
import scipy.ndimage as scn
from scipy.sparse import csc_matrix


def get_contour_file(path):
    """
    Get contour file from a given path by searching for ROIContourSequence
    inside dicom data structure.
    More information on ROIContourSequence available here:
    http://dicom.nema.org/medical/dicom/2016c/output/chtml/part03/sect_C.8.8.6.html

    Inputs:
            path (str): path of the the directory that has DICOM files in it, e.g. folder of a single patient
    Return:
        contour_file (str): name of the file with the contour
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    # get .dcm contour file
    fpaths = [path + f for f in os.listdir(path) if '.dcm' in f]
    n = 0
    contour_file = None
    for fpath in fpaths:
        f = dicom.read_file(fpath)
        if 'ROIContourSequence' in dir(f):
            contour_file = fpath.split('/')[-1]
            n += 1
    if n > 1: warnings.warn("There are multiple contour files, returning the last one!")
    if contour_file is None: print("No contour file found in directory")
    return contour_file


def get_roi_names(contour_data):
    """
    This function will return the names of different contour data,
    e.g. different contours from different experts and returns the name of each.
    Inputs:
        contour_data (dicom.dataset.FileDataset): contour dataset, read by dicom.read_file
    Returns:
        roi_seq_names (list): names of the
    """
    roi_seq_names = [roi_seq.ROIName for roi_seq in list(contour_data.StructureSetROISequence)]
    return roi_seq_names


def coord2pixels(contour_dataset, path, uid_dict):
    """
    Given a contour dataset (a DICOM class) and path that has .dcm files of
    corresponding images. This function will return img_arr and contour_arr (2d image and contour pixels)
    Inputs
        contour_dataset: DICOM dataset class that is identified as (3006, 0016)  Contour Image Sequence
        path: string that tells the path of all DICOM images
        uid_dict: dictionary to convert SOPInstanceUID to filenames
    Return
        img_arr: 2d np.array of image with pixel intensities
        contour_arr: 2d np.array of contour with 0 and 1 labels
    """

    contour_coord = contour_dataset.ContourData

    # x, y, z coordinates of the contour in mm
    x0 = contour_coord[len(contour_coord) - 3]
    y0 = contour_coord[len(contour_coord) - 2]
    z0 = contour_coord[len(contour_coord) - 1]
    coord = []
    for i in range(0, len(contour_coord), 3):
        x = contour_coord[i]
        y = contour_coord[i + 1]
        z = contour_coord[i + 2]
        l = math.sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0))
        l = math.ceil(l * 2) + 1
        for j in range(1, l + 1):
            coord.append([(x - x0) * j / l + x0, (y - y0) * j / l + y0, (z - z0) * j / l + z0])
        x0 = x
        y0 = y
        z0 = z

    # extract the image id corresponding to given countour
    # read that dicom file (assumes filename = sopinstanceuid.dcm)
    SOPInstanceUID = contour_dataset.ContourImageSequence[0].ReferencedSOPInstanceUID
    img_ID = uid_dict[SOPInstanceUID]

    img = dicom.read_file(path + img_ID)
    img_arr = img.pixel_array

    # physical distance between the center of each pixel
    x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])

    # this is the center of the upper left voxel
    origin_x, origin_y, _ = img.ImagePositionPatient

    # y, x is how it's mapped
    pixel_coords = [(np.round((y - origin_y) / y_spacing), np.round((x - origin_x) / x_spacing)) for x, y, _ in coord]

    # get contour data for the image
    rows = []
    cols = []
    for i, j in list(set(pixel_coords)):
        rows.append(i)
        cols.append(j)
    contour_arr = csc_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int8,
                             shape=(img_arr.shape[0], img_arr.shape[1])).toarray()

    return img_arr, contour_arr, img_ID


def cfile2pixels(contour_path, path, uid_dict, ROIContourSeq=0):
    """
    Given a contour file and path of related images return pixel arrays for contours
    and their corresponding images.
    Inputs
        file: filepath of contour
        path: path that has contour and image files
        ROIContourSeq: tells which sequence of contouring to use default 0 (RTV)
        uid_dict: dictionary to convert SOPInstanceUID to filenames
    Return
        contour_iamge_arrays: A list which have pairs of img_arr and contour_arr for a given contour file
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    f = dicom.read_file(contour_path)
    # index 0 means that we are getting RTV information
    RTV = f.ROIContourSequence[ROIContourSeq]
    # get contour datasets in a list
    contours = [contour for contour in RTV.ContourSequence]
    # print(contours[0].ContourData)
    img_contour_arrays = [coord2pixels(cdata, path, uid_dict) for cdata in
                          contours]  # list of img_arr, contour_arr, im_id

    # debug: there are multiple contours for the same image indepently
    # sum contour arrays and generate new img_contour_arrays
    contour_dict = defaultdict(int)
    for im_arr, cntr_arr, im_id in img_contour_arrays:
        contour_dict[im_id] += cntr_arr
    image_dict = {}
    for im_arr, cntr_arr, im_id in img_contour_arrays:
        image_dict[im_id] = im_arr
    img_contour_arrays = [(image_dict[k], contour_dict[k], k) for k in image_dict]

    return img_contour_arrays




def cfile2points(contour_path, path, uid_dict, ROIContourSeq=0):
    """
    Given a contour file and path of related images return points of the contour
    Inputs
        file: filepath of contour
        path: path that has contour and image files
        ROIContourSeq: tells which sequence of contouring to use default 0 (RTV)
        uid_dict: dictionary to convert SOPInstanceUID to filenames
    Return
        contour_points: A array of shape [N, 3] of all the contour points.
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    f = dicom.read_file(contour_path)
    # index 0 means that we are getting RTV information
    RTV = f.ROIContourSequence[ROIContourSeq]
    # get contour datasets in a list
    contours = [contour for contour in RTV.ContourSequence]
    # print(contours[0].ContourData)
    contour_points = []

    for contour in contours:
        contour_points += contour.ContourData
    contour_points = np.array(contour_points).reshape(-1,3)

    return contour_points


def get_points(path_images, path_contour, index, filled=True):
    """
    Function to generate an array with the contour points
    Inputs:
        path (str): path of the the directory that has DICOM files in it
        contour_file: structure file
        index (int): index of the structure
     Return
        contour_points: A array of shape [N, 3] of all the contour points.
    """
    uid_dict = generate_uid_dict(path_images)
    # handle `/` missing
    if path_images[-1] != '/': path_images += '/'
    if path_contour[-1] != '/': path_contour += '/'
    # get slice orders
    ordered_slices = slice_order(path_images, uid_dict)
    # get contour file
    contour_file = path_contour + get_contour_file(path_contour)

    # img_arr, contour_arr, img_fname
    contour_points = cfile2points(contour_file, path_images, uid_dict, index)
    return contour_points



def plot2dcontour(img_arr, contour_arr, figsize=(20, 20)):
    """
    Shows 2d MR img with contour
    Inputs
        img_arr: 2d np.array image array with pixel intensities
        contour_arr: 2d np.array contour array with pixels of 1 and 0
    """

    masked_contour_arr = np.ma.masked_where(contour_arr == 0, contour_arr)
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.imshow(img_arr, cmap='gray', interpolation='none')
    plt.subplot(1, 2, 2)
    plt.imshow(img_arr, cmap='gray', interpolation='none')
    plt.imshow(masked_contour_arr, cmap='cool', interpolation='none', alpha=0.7)
    plt.show()


def slice_order(path, uid_dict):
    """
    Takes path of directory that has the DICOM images and returns
    a ordered list that has ordered filenames
    Inputs
        path: path that has .dcm images
        uid_dict: dictionary to convert SOPInstanceUID to filenames
    Returns
        ordered_slices: ordered tuples of filename and z-position
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    slices = []
    for s in os.listdir(path):
        try:
            f = dicom.read_file(path + '/' + s)
            f.pixel_array  # to ensure not to read contour file
            assert f.Modality != 'RTDOSE'
            slices.append(f)
        except:
            continue

    slice_dict = {uid_dict[s.SOPInstanceUID]: s.ImagePositionPatient[-1] for s in slices}
    ordered_slices = sorted(slice_dict.items(), key=operator.itemgetter(1))
    return ordered_slices


def get_contour_dict(contour_path, path_images, index, uid_dict):
    """
    Returns a dictionary as k: img fname, v: [corresponding img_arr, corresponding contour_arr]
    Inputs:
        contour_path: filepath to .dcm contour file.
        path: path which has image files
        uid_dict: dictionary to convert SOPInstanceUID to filenames
    Returns:
        contour_dict: dictionary with 2d np.arrays
    """
    # handle `/` missing
    if path_images[-1] != '/': path_images += '/'
    # img_arr, contour_arr, img_fname
    contour_list = cfile2pixels(contour_path, path_images, uid_dict, index)

    contour_dict = {}
    for img_arr, contour_arr, img_id in contour_list:
        contour_dict[img_id] = [img_arr, contour_arr]

    return contour_dict


def get_data(path_images, path_contour, index):
    """
    Generate image array and contour array
    Inputs:
        path (str): path of the the directory that has DICOM files in it
        contour_file: structure file
        index (int): index of the structure
    """
    # uid_dict: dictionary to convert SOPInstanceUID to filenames
    uid_dict = generate_uid_dict(path_images)

    images = []
    contours = []
    # handle `/` missing
    if path_images[-1] != '/': path_images += '/'
    if path_contour[-1] != '/': path_contour += '/'

    # get contour file
    contour_file = path_contour + get_contour_file(path_contour)
    # get slice orders
    ordered_slices = slice_order(path_images, uid_dict)
    # get contour dict
    contour_dict = get_contour_dict(contour_file, path_images, index,  uid_dict)

    for k,v in ordered_slices:
        # get data from contour dict
        if k in contour_dict:
            images.append(contour_dict[k][0])
            contours.append(contour_dict[k][1])
        # get data from dicom.read_file
        else:
            img_arr = dicom.read_file(path_images + k).pixel_array
            contour_arr = np.zeros_like(img_arr)
            images.append(img_arr)
            contours.append(contour_arr)

    return np.array(images), np.array(contours)


def get_mask(path_images, path_contour, index, filled=True):
    """
    Generate image array and contour array
    Inputs:
        path (str): path of the the directory that has DICOM files in it
        contour_file: structure file
        index (int): index of the structure
    """
    contours = []
    uid_dict = generate_uid_dict(path_images)
    # handle `/` missing
    if path_images[-1] != '/': path_images += '/'
    if path_contour[-1] != '/': path_contour += '/'
    # get slice orders
    ordered_slices = slice_order(path_images, uid_dict)
    # get contour file
    contour_file = path_contour + get_contour_file(path_contour)

    # get contour dict
    contour_dict = get_contour_dict(contour_file, path_images, index, uid_dict)
    for k, v in ordered_slices:
        # get data from contour dict
        if k in contour_dict:
            y = contour_dict[k][1]
            # y = scn.binary_fill_holes(y) make mask from contour
            contours.append(y)
        # get data from dicom.read_file
        else:
            img_arr = dicom.read_file(path_images + k).pixel_array
            contour_arr = np.zeros_like(img_arr)
            contours.append(contour_arr)

    return np.array(contours)


def create_image_mask_files(path, contour_file, index, img_format='png'):
    """
    Create image and corresponding mask files under to folders '/images' and '/masks'
    in the parent directory of path.

    Inputs:
        path (str): path of the the directory that has DICOM files in it, e.g. folder of a single patient
        index (int): index of the desired ROISequence
        img_format (str): image format to save by, png by default
    """
    # Extract Arrays from DICOM
    X, Y = get_data(path, contour_file, index)
    Y = np.array([scn.binary_fill_holes(y) if y.max() == 1 else y for y in Y])

    # Create images and masks folders
    new_path = '/'.join(path.split('/')[:-2])
    os.makedirs(new_path + '/images/', exist_ok=True)
    os.makedirs(new_path + '/masks/', exist_ok=True)
    for i in range(len(X)):
        plt.imsave(new_path + f'/images/image_{i}.{img_format}', X[i])
        plt.imsave(new_path + f'/masks/mask_{i}.{img_format}', Y[i])


def generate_uid_dict(path):
    """
    Generate a dictionary to convert SOPInstanceUID to filenames
    Args:
        path: path of the directory that has DICOM files in it

    Returns: uid_dict: dictionary to convert SOPInstanceUID to filenames

    """
    full_path, dirs, files = next(os.walk(path))
    uid_dict = {}
    for file in files:
        data = dicom.dcmread(path + '/' + file)
        uid_dict[data.SOPInstanceUID] = file
    return uid_dict
