from datetime import datetime, timezone
from random import sample

import voxelmorph
from CT_path_dict.ct_path_dict import ct_path_dict
from dataset_generator import *
from train_model import train_model
from write_parameters_to_file import *

# .\venv\Scripts\activate
# dataset parameters
root_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-256/"
dimensions = [0, 80, 0, 256, 0, 256]
data_shape = [dimensions[1]-dimensions[0],dimensions[3]-dimensions[2],dimensions[5]-dimensions[2]]
shift = [0, 0, 0, 0]

# Network parameters for voxelmorph
nb_features = [
    [16, 32, 32, 32],
    [32, 32, 32, 32, 32, 16, 16]]  # number of features of encoder and decoder
losses = [voxelmorph.losses.MSE().loss, voxelmorph.losses.MSE().loss, voxelmorph.losses.Grad('l2').loss]
loss_weights = [0.5, 0.5, 0.01]

# Training parameters
learning_rate = 1e-3
epochs = 50
batch_size = 2
int_downsize = 2

print("Shape of dataset:", data_shape)

train = False

model = voxelmorph.networks.VxmDense(data_shape, nb_features, int_downsize=int_downsize, bidir=True)
if torch.cuda.is_available():
    model.cuda()  # If possible move model to GPU.

# train model
if train:
    file_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/saved_models/training_{}/".format(
        datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S"))

    patient_id = ["107"]
    scan_id_training = ["06-02-1999-p4-89680"]

    # single patient, single scan
    training_scan_keys = scan_key_generator(ct_path_dict, patient_id, scan_id_training)

    scan_id_validation = ["05-26-1999-p4-39328"]
    validation_batches = 20
    validation_scan_keys = scan_key_generator(ct_path_dict, patient_id, scan_id_validation)
    random_validation_keys = sample(validation_scan_keys, validation_batches * batch_size)

    print("Number of training samples:", len(training_scan_keys))
    print("Number of validation samples:", len(random_validation_keys))

    # Generate datasets
    training_set = generate_dataset(training_scan_keys, root_path, ct_path_dict, dimensions, shift, batch_size)
    validation_set = generate_dataset(random_validation_keys, root_path, ct_path_dict, dimensions, shift, batch_size)

    training_parameters_string = training_parameters_to_string(learning_rate, epochs, batch_size, loss_weights,
                                                               patient_id, scan_id_training, scan_id_validation,
                                                               validation_batches, nb_features, data_shape, int_downsize
                                                               ,losses)

    write_string_to_file(file_path, "training_parameters.txt", training_parameters_string)

    trained_model, training_loss, validation_loss = train_model(model, training_set, validation_set,
                                                                epochs, learning_rate, losses, loss_weights, file_path)

