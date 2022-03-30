import numpy as np
from slice_viewer import slice_viewer
import voxelmorph as vxm
import torch
from datetime import datetime
from dataset_generator import dataset_generator
from ct_path_dict import ct_path_dict

# data parameters
root_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung/"
patient_id = 107
scan_id = "06-02-1999-p4-89680"
m_phases = [0, 30]
f_phase = 50
z_max = 80

# Obtain data
dataset = dataset_generator(patient_id, scan_id, m_phases, f_phase, root_path, ct_path_dict)
vol_shape = np.array(dataset[:, :, :z_max, :, :].shape)[-3:]  # input shape

# Network parameters for voxelmorph

nb_features = [
    [16, 32, 32, 32],
    [32, 32, 32, 16]]  # number of features of encoder and decoder

losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
loss_weights = [1, 0.01]

vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)
learning_rate = 1e-3


def train(epochs):
    torch.backends.cudnn.deterministic = True
    vxm_model.train()
    optimizer = torch.optim.Adam(vxm_model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print("epoch {} of {}".format(epoch + 1, epochs))

        # iterate over all image pairs in the dataset.
        for index in range(dataset.shape[0]):
            fixed_tensor = dataset[None, None, index, 0, ...]
            moving_tensor = dataset[None, None, index, 1, ...]

            prediction, pos_flow = vxm_model(fixed_tensor, moving_tensor)
            np.save("predicted_CT_epoch_{}.npy".format(epoch + 1), prediction[0, 0, ...].detach().numpy())
            print(np.shape(prediction))

            print("calculate losses")
            loss_function = vxm.losses.MSE()
            loss = loss_function.loss(moving_tensor, prediction)
            print("loss:", loss)
            optimizer.zero_grad()
            loss.backward()
            print("done")
            optimizer.step()
    torch.save(vxm_model,
               "./saved_models/voxelmorph_model_timestamp_{}.pth".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
    print("model saved")


train(3)
