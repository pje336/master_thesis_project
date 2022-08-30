import json
from datetime import datetime, timezone
from random import sample, randint
from write_string_to_file import write_string_to_file, training_parameters_to_string
from Network_Functions.dataset_generator import *
from train_voxelmorph_model import train_model


os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


# .\venv\Scripts\activate
# dataset parameters
root_path_data = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-256-h5/"
root_path_saving =  "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/saved_models/"


dimensions = [0, 80, 0, 256, 0, 256]

data_shape = [dimensions[1] - dimensions[0], dimensions[3] - dimensions[2], dimensions[5] - dimensions[2]]
shift = [5, 5, 15, 0]
int_downsize = 1
dropout_rate = 0.3

# Network parameters for voxelmorph
nb_features = [
    [16, 32, 32, 32],
    [32, 32, 32, 32, 32, 16, 16]]  # number of features of encoder and decoder

NCC = True
MSE = False

if NCC:
  losses = [voxelmorph.losses.NCC().loss,voxelmorph.losses.NCC().loss, voxelmorph.losses.Grad('l2').loss]
  # loss_weights = [0.5, 0.5, 1]
  loss_weights = [2500, 2500, 1] # from oscar
if MSE:
  losses = [voxelmorph.losses.MSE().loss,voxelmorph.losses.MSE().loss, voxelmorph.losses.Grad('l2').loss]
  loss_weights = [0.5,0.5, 0.01]

# Training parameters
learning_rate = 1e-3
epochs = 8
batch_size = 1
batches_per_step = 10

print("Shape of dataset:", data_shape)
steps_per_epoch = 200


trained_model_folder = "None"
trained_epoch = 7
load_pretrained_model = False
tanh_value = None
print(tanh_value, flush=True)

model = voxelmorph.networks.VxmDense(data_shape, nb_features, int_downsize=int_downsize, bidir=True, dropout = dropout_rate, tanh = tanh_value)

if torch.cuda.is_available():
    model.cuda()


if load_pretrained_model:
   model.load_state_dict(torch.load(root_path_saving + trained_model_folder + "/voxelmorph_model_epoch_{}.pth".format(trained_epoch), map_location=device))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if load_pretrained_model:
   optimizer.load_state_dict(torch.load(root_path_saving + trained_model_folder + "/optimiser_epoch_{}.pth".format(trained_epoch), map_location=device))



with open(root_path_data + "scan_dictionary.json", 'r') as file:
    ct_path_dict = json.load(file)




train = True

# train model
if train:
    file_path = root_path_saving + "training_{}/".format(
        datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S"))
    print(file_path, flush=True)


    patient_id_training = ["107","109","110","111","112","113","114","115","116","117","118","119"]
    # patient_id_training = ["107"]

    # scan_id_training = ["06-02-1999-p4-89680"]
    scan_id_training = None

    # single patient, single scan
    training_scan_keys = scan_key_generator(ct_path_dict, patient_id_training)

    patient_id_validation = ["108"]
    # scan_id_validation = ["06-15-1999-p4-07025"]
    scan_id_validation = None

    validation_scan_keys = scan_key_generator(ct_path_dict, patient_id_validation, scan_id_validation)
    validation_batches = 20
    random_validation_keys = sample(validation_scan_keys, batch_size * validation_batches)
    random_validation_keys = [['108', '07-28-1999-p4-56648', ['10', '40']], ['108', '06-15-1999-p4-07025', ['90', '80']], ['108', '07-02-1999-p4-21843', ['10', '0']], ['108', '07-08-1999-p4-52902', ['80', '40']], ['108', '06-21-1999-p4-15001', ['50', '40']], ['108', '07-02-1999-p4-21843', ['20', '60']], ['108', '07-28-1999-p4-56648', ['0', '90']], ['108', '06-21-1999-p4-15001', ['60', '80']], ['108', '07-02-1999-p4-21843', ['80', '10']], ['108', '07-08-1999-p4-52902', ['10', '80']], ['108', '07-02-1999-p4-21843', ['0', '0']], ['108', '07-28-1999-p4-56648', ['20', '20']], ['108', '07-28-1999-p4-56648', ['0', '0']], ['108', '07-28-1999-p4-56648', ['0', '30']], ['108', '07-08-1999-p4-52902', ['90', '40']], ['108', '07-28-1999-p4-56648', ['90', '70']], ['108', '07-28-1999-p4-56648', ['50', '90']], ['108', '07-08-1999-p4-52902', ['40', '20']], ['108', '06-15-1999-p4-07025', ['0', '0']], ['108', '06-21-1999-p4-15001', ['20', '90']], ['108', '06-21-1999-p4-15001', ['90', '30']], ['108', '07-02-1999-p4-21843', ['40', '90']], ['108', '07-28-1999-p4-56648', ['20', '50']], ['108', '07-28-1999-p4-56648', ['70', '20']], ['108', '07-28-1999-p4-56648', ['80', '0']], ['108', '07-02-1999-p4-21843', ['10', '30']], ['108', '07-02-1999-p4-21843', ['0', '60']], ['108', '07-28-1999-p4-56648', ['30', '80']], ['108', '07-02-1999-p4-21843', ['70', '50']], ['108', '07-28-1999-p4-56648', ['0', '80']], ['108', '06-21-1999-p4-15001', ['40', '90']], ['108', '06-21-1999-p4-15001', ['80', '70']], ['108', '06-21-1999-p4-15001', ['0', '80']], ['108', '06-15-1999-p4-07025', ['90', '90']], ['108', '07-02-1999-p4-21843', ['50', '20']], ['108', '07-02-1999-p4-21843', ['40', '50']], ['108', '07-02-1999-p4-21843', ['90', '80']], ['108', '06-21-1999-p4-15001', ['50', '70']], ['108', '07-28-1999-p4-56648', ['0', '60']], ['108', '07-28-1999-p4-56648', ['70', '40']]]



    print("Number of training samples:", len(training_scan_keys, ), flush=True)
    print("Number of validation samples:", len(random_validation_keys), flush=True)
    print(random_validation_keys, flush=True)
    # Generate datasets
    training_set = generate_dataset(training_scan_keys, root_path_data, ct_path_dict, dimensions, shift, batch_size)
    validation_set = generate_dataset(random_validation_keys, root_path_data, ct_path_dict, dimensions, shift, batch_size)

    training_parameters_string = training_parameters_to_string(learning_rate, epochs, batch_size, batches_per_step, loss_weights,
                                                               validation_batches, nb_features, data_shape,
                                                               int_downsize, losses, dropout_rate,
                                                               patient_id_training, scan_id_training,
                                                               patient_id_validation, scan_id_validation)

    write_string_to_file(file_path, "training_parameters.txt", training_parameters_string)

    trained_model, training_loss, validation_loss = train_model(model, optimizer, training_set, validation_set, epochs, batch_size, losses, loss_weights,
                file_path, steps_per_epoch, batches_per_step)
