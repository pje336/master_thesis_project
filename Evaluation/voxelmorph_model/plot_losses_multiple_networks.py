from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np

trained_model_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/saved_models/voxelmorph_grad_value_test/"

trained_model_folder = ["training_2022_09_26_09_04_30_voxelmorph_hardtan_10_grad_value_1",
                        "training_2022_09_26_09_04_30_voxelmorph_hardtan_10_grad_value_10",
                        "training_2022_09_26_09_05_41_voxelmorph_hardtan_10_grad_value_100",
                        "training_2022_09_26_09_05_58_voxelmorph_hardtan_10_grad_value_500",
                        "training_2022_09_26_09_06_39_voxelmorph_hardtan_10_grad_value_1000"]

fig, ax1 = plt.subplots()





# symbol = ["x", "o", "*", "s",".", "x", "s" ,"o", "*", "o",".", "x", "s" ]
symbol = ["o","*", "x", "s", "." ]
for i, scan in enumerate(trained_model_folder):
    print(scan)
    training_loss_str = open(trained_model_path + scan + "/training_loss.txt").read()[1:-1].replace("\n"," ").replace("  "," ").split(", ")
    training_loss = np.array([float(x) for x in training_loss_str])

    x = np.arange(1, len(training_loss)+1 , 1)

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training loss')
    ax1.plot(x, training_loss / 2500, symbol[i], label = trained_model_folder[i])
    ax1.tick_params(axis='y')

plt.legend(loc='upper right')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

# __________________________
fig, ax1 = plt.subplots()
for i, scan in enumerate(trained_model_folder):
    print(scan)
    validation_loss_str = open(trained_model_path + scan + "/validation_loss.txt").read()[1:-1].replace("\n"," ").replace("  "," ").split(", ")
    validation_loss = np.array([float(x) for x in validation_loss_str])

    # if scan == "training_2022_05_05_13_16_57":
    #     training_loss.pop(6)
    x = np.arange(1, len(validation_loss) + 1, 1)

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation loss')
    ax1.plot(x, validation_loss / 2500, symbol[i], label = trained_model_folder[i])
    ax1.tick_params(axis='y')
    plt.legend(loc='upper right')

# plt.legend(loc='lower center')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()



# # __________________________
# fig, ax1 = plt.subplots()
#
# validation_loss = [0.07581225, 0.07582858, 0.07679947, 0.06304156, 0.06355391, 0.06694065,
#                    0.06474733, 0.06635991, 0.06987813, 0.06392489, 0.07312243, 0.07082445, 0.07322995, 0.0711169  ]
#
#
# x = np.array([-1, 0,2,4,6,8,10,12,14,16,18,20,22,25] ) + 1
#
#
# color = 'tab:red'
# ax1.set_xlabel('Epoch')
# ax1.set_ylabel('Cumulative MSE')
# ax1.plot(x, validation_loss, symbol[i], label = labels[i])
# ax1.tick_params(axis='y')
# plt.show()