import numpy as np
import matplotlib.pyplot as plt

model_path = 'C:\\Users\\pje33\\Google Drive\\Sync\\TU_Delft\\MEP\\saved_models\\training_2022_08_16_11_04_43'

for level in range(3,4):
    filename = "lossLDR_OASIS_NCC_unit_disp_add_fea7_reg01_10_testing_stagelvl{}.npy".format(level)

    losses = np.load(model_path + "\\" + filename)[:,:-2]

    plt.plot(-losses[0][800:])
plt.yscale("log")
plt.show()
