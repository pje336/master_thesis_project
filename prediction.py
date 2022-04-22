import numpy as np
import torch

import voxelmorph
from CT_path_dict.ct_path_dict import ct_path_dict
from dataset_generator import scan_key_generator, generate_dataset


def read_model_parameters_from_file(model_path:str, filename:str="training_parameters.txt"):
    """
    Read the model parameters from the file training_parameters.txt and convert the values into variables.
    Args:
        model_path: [string] path to the folder with trained model.
        filename: [string]  filename of file with training parameters, by default is "training_parameters.txt"

    Returns: learning_rate, epochs, batch_size, loss_weights, patient_id, scan_id_training, scan_id_validation, \
           validation_batches, nb_features, data_shape

    """
    text = open(model_path + filename).read()  # read the file
    exec(text[text.find("\n") + 1:], globals())  # execute the text to set the variable values.
    return learning_rate, epochs, batch_size, loss_weights, patient_id, scan_id_training, scan_id_validation, \
           validation_batches, nb_features, data_shape


# Filepaths for the CT data and the trained model.
root_path_data = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-256/"
trained_model_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/saved_models/training_2022_04_20_10_56_34/"
trained_model_dict_name = "voxelmorph_model_epoch_0.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Obtain training parameters.
learning_rate, epochs, batch_size, loss_weights, patient_id, scan_id_training, scan_id_validation, validation_batches, \
nb_features, data_shape = read_model_parameters_from_file(trained_model_path)

# Make the model and load the weights.
model = voxelmorph.networks.VxmDense(data_shape, nb_features, int_downsize=1, bidir=True)
model.load_state_dict(torch.load(trained_model_path + trained_model_dict_name))
model.to(device)
model.eval()

# Parameters for evaluation dataset
patient_id_evaluation = ["107"]
scan_id_evaluation = ["06-02-1999-p4-89680"]
batch_size = 1
dimensions = [0, data_shape[0], 0, data_shape[1], 0, data_shape[2]]
shift = [0, 0, 0, 0]

# Make an evaluation dataset.
evaluation_scan_keys = scan_key_generator(ct_path_dict, patient_id_evaluation, scan_id_evaluation)
evaluation_set = generate_dataset(evaluation_scan_keys, root_path_data, ct_path_dict, dimensions, shift, batch_size)

# Run through all the samples in the evaluation_set.
for moving_tensor, fixed_tensor in evaluation_set:
    moving_tensor = moving_tensor.to(device)
    fixed_tensor = fixed_tensor.to(device)
    prediction = model(moving_tensor, fixed_tensor)

    # Calculate jacobian for every voxel in deformation map.
    jac = voxelmorph.py.utils.jacobian_determinant(prediction[-1][0].permute(1, 2, 3, 0).cpu().detach().numpy())
    print("percentage of voxel with negative jacobian:", np.size(jac[jac < 0]) / np.size(jac) * 100)

    del prediction
    del fixed_tensor
    del moving_tensor
