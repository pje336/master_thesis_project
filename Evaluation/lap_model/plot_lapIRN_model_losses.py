import glob
import numpy as np
import matplotlib.pyplot as plt
# from LapIRN_model.Code.Test_cLapIRN import *


trained_model_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/saved_models/lap_model_ncc_test/"
model_names = ["training_2022_08_31_07_49_02", "training_2022_08_31_07_49_02"]
              # ,"training_2022_08_22_08_12_12",
              # "training_2022_08_22_08_11_00",
              # "training_2022_08_22_08_10_03",
              # "training_2022_08_22_08_09_36",
              # "training_2022_08_17_08_15_24"]

model_names = ["training_2022_09_09_06_28_29_number_res_blocks_5_10", "training_2022_09_09_08_19_49_number_res_blocks_7_10"
               ,"training_2022_09_09_10_17_53_number_res_blocks_9_10", "training_2022_09_09_10_17_53_number_res_blocks_5_4"]
model_names = ["training_2022_09_14_13_32_17_7_res_blocks_32_filters/losses"]
model_names = ["training_2022_09_14_13_49_39_5_res_blocks_32_filters",
               "training_2022_09_14_13_32_17_7_res_blocks_32_filters",
               "training_2022_09_14_13_38_18_9_res_blocks_32_filters",
               "training_2022_09_14_13_53_03_5_res_blocks_48_filters",
               "training_2022_09_14_13_52_27_7_res_blocks_48_filters",
               "training_2022_09_14_13_51_47_9_res_blocks_48_filters",
               "training_2022_09_14_14_10_37_5_res_blocks_80_filters",
               "training_2022_09_14_14_11_23_7_res_blocks_80_filters",
               "training_2022_09_14_14_12_14_9_res_blocks_80_filters"]
model_names = ["training_2022_09_26_15_20_36_7_res_blocks_32_filters",
"training_2022_09_27_07_35_11_9_res_blocks_32_filters" ,
"training_2022_09_27_07_36_12_9_res_blocks_48_filters" ,
"training_2022_09_27_07_37_01_7_res_blocks_48_filters" ,
"training_2022_09_27_07_37_45_5_res_blocks_48_filters" ,
"training_2022_09_27_07_39_13_5_res_blocks_80_filters" ,
"training_2022_09_27_07_39_52_7_res_blocks_80_filters" ,
"training_2022_09_27_07_40_57_9_res_blocks_80_filters"]

training_losses = []
validation_losses = []





symbol = [".",".",".","x","x","x","s","s","s"]


for m, model in enumerate(model_names):

    training_losses.append([])
    validation_losses.append([])
    training_loss_files = glob.glob(trained_model_path + model + "/training_*.npy")
    training_loss_files.sort()
    validation_loss_files = glob.glob(trained_model_path + model + "/validation_*.npy")
    validation_loss_files.sort()

    for i, file_name in enumerate(training_loss_files):
        training_losses[m].append(np.load(file_name).mean(axis = 1))
    for i, file_name in enumerate(validation_loss_files):
        validation_losses[m].append(np.load(file_name).mean(axis = 1))
    plt.figure(1)
    plt.plot(np.array(training_losses[m])[:, 0], symbol[m], label = str(model[-23:]))
    plt.figure(2)
    plt.plot(np.array(validation_losses[m])[:, 0], symbol[m] , label = str(model[-23:]))



plt.figure(1)
plt.axvline(x=40)
plt.axvline(x=80)
plt.axvline(x=120)
plt.xlabel("epoch")
plt.ylabel("Training loss")
# plt.legend()
plt.figure(2)
plt.axvline(x=40)
plt.axvline(x=80)
plt.axvline(x=120)
plt.xlabel("epoch")
plt.ylabel("validation loss")
# plt.legend()



plt.show()