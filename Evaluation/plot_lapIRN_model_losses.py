import glob
import numpy as np
import matplotlib.pyplot as plt

trained_model_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/saved_models/"
model_names = ["training_2022_08_17_08_15_24","training_2022_08_16_11_04_43"]
losses = []
for m, model in enumerate(model_names):
    losses.append([])
    model_dict_filename = glob.glob(trained_model_path + model + "/lossLDR_*_stagelvl*.npy")
    for i, file_name in enumerate(model_dict_filename):
        losses[m].append(np.load(file_name).mean(axis = 1))


for loss in losses:
    loss_array = np.array(loss)
    plt.plot(loss_array[:,0], ".")
plt.show()