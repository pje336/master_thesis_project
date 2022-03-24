import numpy as np
from data_importer import import_CT_data
from slice_viewer import slice_viewer
import voxelmorph as vxm
import torch
from datetime import datetime





path_drive = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/"
path_CT_0 = "4D_lung_CT/4D-Lung/107_HM10395/05-26-1999-p4-39328/1.000000-P4P107S300I00003 Gated 0.0A-97958/"
path_CT_90 = "4D_lung_CT/4D-Lung/107_HM10395/05-26-1999-p4-39328/1.000000-P4P107S300I00012 Gated 90.0A-89848/"

CT_data_0 = import_CT_data(path_drive, path_CT_0)[:80]
CT_data_90 = import_CT_data(path_drive, path_CT_90)[:80]




# network parameters for voxelmorph
vol_shape = np.array(np.shape(CT_data_0)) #input shape
nb_features = [
    [16, 32, 32, 32],
    [ 32, 32, 32, 16]
] # number of features of encoder and decoder

losses = vxm.losses.Grad('l2').loss
loss_weights = [1, 0.01]

vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)


learning_rate = 1e-3
optimizer = torch.optim.Adam(vxm_model.parameters(), lr=learning_rate)

x_data = torch.tensor(CT_data_0[np.newaxis,np.newaxis,  ...], dtype=torch.float)
y_data = torch.tensor(CT_data_90[np.newaxis,np.newaxis, ...], dtype=torch.float)
print(np.shape(x_data))
# print(vxm_model)
def train(epochs):
    for epoch in range(epochs):
        print("epoch {} of {}".format(epoch+1 ,epochs))
        vxm_model.train()
        # for x_data, y_data:

        prediction, pos_flow = vxm_model(x_data,y_data)
        np.save("predicted_CT_epoch_{}.npy".format(epoch+1),prediction[0,0,...].detach().numpy())
        print(np.shape(prediction))

        print("calculate losses")
        loss_function = vxm.losses.MSE()
        loss = loss_function.loss(y_data,prediction)
        print("loss:",loss)
        optimizer.zero_grad()
        loss.backward()
        print("done")
        optimizer.step()
    torch.save(vxm_model, "./saved_models/voxelmorph_model_timestamp_{}.pth".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
    print("model saved")

train(3)







