import pydicom
import matplotlib.pyplot as plt
import numpy as np
from data_importer import import_CT_data
from slice_viewer import slice_viewer

#.\venv\Scripts\activate

path_drive = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/"
path_CT_0 = "4D_lung_CT/4D-Lung/107_HM10395/05-26-1999-p4-39328/1.000000-P4P107S300I00003 Gated 0.0A-97958/"
path_CT_90 = "4D_lung_CT/4D-Lung/107_HM10395/05-26-1999-p4-39328/1.000000-P4P107S300I00012 Gated 90.0A-89848/"

CT_data_0 = import_CT_data(path_drive, path_CT_0)
CT_data_90 = import_CT_data(path_drive, path_CT_90)


slice_viewer([CT_data_90,CT_data_90])


