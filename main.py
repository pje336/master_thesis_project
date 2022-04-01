from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

import voxelmorph as vxm
from ct_path_dict import ct_path_dict
from dataset_generator import dataset_generator
from voxelmorph_model import train

# dataset parameters
root_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung/"
patient_id = 107
scan_id = "06-02-1999-p4-89680"
m_phases = [0, 30, 10, 20]
f_phase = [50, 50, 50, 50]
pad = True



nb_features = [
    [16, 32, 32, 32, 32],
    [32, 32, 32, 32, 32]]  # number of features of encoder and decoder
relative_size = len(nb_features[0])-1

# Network parameters for voxelmorph
learning_rate = 1e-3
batch_size = 2
epochs = 3
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
loss_weights = [1, 0.01]

# Obtain CT dataset
data = dataset_generator(patient_id, scan_id, m_phases, f_phase, root_path, ct_path_dict, relative_size, pad)
data_shape = np.array(data.shape()[-3:])
dataset = DataLoader(data, 1, shuffle=True)
print("Shape of dataset:",data_shape)


# train model
model = vxm.networks.VxmDense(data_shape, nb_features, int_steps=0)
trained_model = train(model, dataset, epochs, batch_size, learning_rate, losses, loss_weights)
save_model = False
if save_model:
    torch.save(trained_model,
               "./saved_models/voxelmorph_model_timestamp_{}.pth".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
    print("model saved")
