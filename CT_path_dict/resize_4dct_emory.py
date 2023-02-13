"""
script to resize the 4DCT scans from the emory univserity and store them as h5 files.
https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html
"""


import scipy.io
import csv
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import h5py
import os

patients = [6,7,8,9,10]
phases = [0,10,20,30,40,50,60,70,80,90]
id = [130,131,132,133,134]
full_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-512/"
path_h5 = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-256-h5/"


for i, patient in enumerate(patients):
    mat = scipy.io.loadmat(full_path + 'case_{}.mat'.format(patient))
    filepath_scan = path_h5 + "{}_HM10395/scan_{}_1".format(id[i],id[i])
    if not os.path.exists(filepath_scan):
        os.makedirs(filepath_scan)
        print("Directory ", filepath_scan, " Created ")
    else:
        print("Directory ", filepath_scan, " already exists")
    print(filepath_scan)
    f = h5py.File(filepath_scan + "CT_dataset_all_phases.hdf5", "w")
    for phase in phases:
        ct_data = mat["case{}_T{}".format(patient,str(phase).zfill(2))].astype(np.float)

        ct_data_zoomed = zoom(ct_data, (1/2,1/2,2.5/3))
        ct_data_zoomed = np.transpose(ct_data_zoomed,(2,0,1))
        print(np.shape(ct_data_zoomed))
        # print(ct_data_zoomed)
        f.create_dataset(str(phase), data=ct_data_zoomed)






