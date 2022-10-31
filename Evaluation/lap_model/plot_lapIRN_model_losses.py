"""
Script to plot the training and validation losses from the .npy files
Change the model_names array for the folders with these files.
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
# from LapIRN_model.Code.Test_cLapIRN import *


trained_model_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/saved_models/lap_model_ncc_test/"

model_names = ["training_2022_09_14_13_49_39_5_res_blocks_32_filters",
               "training_2022_09_14_13_32_17_7_res_blocks_32_filters",
               "training_2022_09_14_13_38_18_9_res_blocks_32_filters",
               "training_2022_09_14_13_53_03_5_res_blocks_48_filters",
               "training_2022_09_14_13_52_27_7_res_blocks_48_filters",
               "training_2022_09_14_13_51_47_9_res_blocks_48_filters",
               "training_2022_09_14_14_10_37_5_res_blocks_80_filters",
               "training_2022_09_14_14_11_23_7_res_blocks_80_filters",
               "training_2022_09_14_14_12_14_9_res_blocks_80_filters"]
model_names = ["4_layers_deep"]

training_losses = []
validation_losses = []


model_tags = []
for model in model_names:
    model_tags.append(model[29:].replace("_"," "))


symbol = [".",".",".","x","x","x","s","s","s"]


for m, model in enumerate(model_names):

    training_losses.append([])
    validation_losses.append([])
    training_loss_files = glob.glob(trained_model_path + model + "/training_*.npy")
    training_loss_files.sort()
    validation_loss_files = glob.glob(trained_model_path + model + "/validation_*.npy")
    validation_loss_files.sort()
    x = np.arange(81,105)
    for i, file_name in enumerate(training_loss_files):
        training_losses[m].append(np.load(file_name).mean(axis = 1))
    for i, file_name in enumerate(validation_loss_files):
        validation_losses[m].append(np.load(file_name).mean(axis = 1))
    plt.figure(1)
    plt.plot(np.array(training_losses[m])[:, 0], symbol[m], label = model_tags[m])
    plt.figure(2)
    plt.plot(np.array(validation_losses[m])[:, 0], symbol[m] , label =model_tags[m])



plt.figure(1)
plt.axvline(x=40)
plt.axvline(x=80)
plt.axvline(x=120)
plt.xlabel("epoch")
plt.ylabel("Training loss")
plt.legend()
plt.figure(2)
plt.axvline(x=40)
plt.axvline(x=80)
plt.axvline(x=120)
plt.xlabel("epoch")
plt.ylabel("validation loss")
plt.legend()

plt.show()