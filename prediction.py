import numpy as np
from data_importer import import_CT_data
from slice_viewer import slice_viewer
import torch

model_path = "./saved_models/voxelmorph_model_timestamp_2022_03_24_14_53_16.pth"
path_drive = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung/"
path_CT_0  = "115_HM10395/05-25-2000-p4-95872/1.000000-P4P115S302I00003 Gated 0.0A-96968/"
path_CT_50 = "110_HM10395/10-14-1999-p4-65942/1.000000-P4P110S300I00008 Gated 50.0A-33388/"

CT_data_0 = import_CT_data(path_drive, path_CT_0)
CT_data50 = import_CT_data(path_drive, path_CT_50)
print(np.shape(CT_data_0))
# model = torch.load(model_path)
#
# x_data = torch.tensor(CT_data_0[np.newaxis,np.newaxis,  ...], dtype=torch.float)
# y_data = torch.tensor(CT_data_90[np.newaxis,np.newaxis, ...], dtype=torch.float)
#
# prediction, pos_flow = model(x_data, y_data)
#
# predicted_CT = prediction[0,0,...].detach().numpy()
# np.save("predicted_CT.npy",prediction[0,0,...].detach().numpy())
#
# predicted_CT_epoch_1 = np.load("predicted_CT_epoch_1.npy")
# predicted_CT_epoch_2 = np.load("predicted_CT_epoch_2.npy")
# predicted_CT_epoch_3 = np.load("predicted_CT_epoch_3.npy")
#
# slice_viewer([CT_data_90,predicted_CT_epoch_1,predicted_CT_epoch_2,predicted_CT_epoch_3])
