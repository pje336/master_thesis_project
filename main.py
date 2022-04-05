from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

import voxelmorph as vxm
from ct_path_dict import ct_path_dict
from dataset_generator_parallel import ct_dataset
from voxelmorph_model import train_model


#.\venv\Scripts\activate
# dataset parameters
root_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung/"
patient_id = "114"
scan_id = "05-05-2000-p4-38187"
f_phase = "50"
dimensions = [0,80,0,512,0,512]

#generate an array with scan_keys
scan_keys = []
for patient_id in ct_path_dict.keys():
    for scan_id in ct_path_dict[patient_id].keys():
        for m_phase in ct_path_dict[patient_id][scan_id].keys():
            scan_keys.append([patient_id,scan_id,[f_phase,m_phase]])





nb_features = [
    [16, 32, 32, 32],
    [ 32, 32, 32, 32]]  # number of features of encoder and decoder
relative_size = len(nb_features[0]) - 1

# Network parameters for voxelmorph
learning_rate = 1e-3
batch_size = 2
epochs = 1
losses = [vxm.losses.NCC().loss, vxm.losses.Grad('l2').loss]
loss_weights = [1, 0.01]

# Obtain CT dataset
data = ct_dataset(root_path,ct_path_dict, scan_keys, dimensions)
data_shape = data.shape()
dataset = DataLoader(data, 1, shuffle=True)
print("Shape of dataset:", data_shape)

train = True
save_model = False
model = vxm.networks.VxmDense(data_shape, nb_features, int_steps=0)
if torch.cuda.is_available():
    model.cuda()


# train model
if train:

    trained_model = train_model(model, dataset, epochs, batch_size, learning_rate, losses, loss_weights)
    if save_model:
        torch.save(trained_model,
                   "./saved_models/voxelmorph_model_timestamp_{}.pth".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
    print("model saved")

