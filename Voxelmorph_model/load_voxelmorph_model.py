import torch
import glob
import Voxelmorph_model.voxelmorph as voxelmorph



def read_model_parameters_from_file(model_path: str, filename: str = "training_parameters.txt"):
    """
    Read the model parameters from the file training_parameters.txt and convert the values into variables.
    Args:
        model_path: [string] path to the folder with trained model.
        filename: [string]  filename of file with training parameters, by default is "training_parameters.txt"

    Returns: learning_rate, epochs, batch_size, loss_weights, patient_id, scan_id_training, scan_id_validation, \
           validation_batches, nb_features, data_shape

    """
    text = open(model_path + filename).read()  # read the file
    exec(text[text.find("\n") + 1:text.find("losses")], globals())  # execute the text to set the variable values.
    return learning_rate, epochs, batch_size, loss_weights, patient_id_training, scan_id_training, patient_id_validation, scan_id_validation, \
           validation_batches, nb_features, data_shape, int_downsize



def load_voxelmorph_model(trained_model_folder , model_name, epoch = None):
    learning_rate, epochs, batch_size, loss_weights, patient_id_training, scan_id_training, patient_id_validation, scan_id_validation, \
    validation_batches, nb_features, data_shape, int_downsize = read_model_parameters_from_file(
        trained_model_folder + model_name + "/")

    model = voxelmorph.torch.networks.VxmDense(data_shape, nb_features, int_downsize=int_downsize, bidir=True,
                                               tanh=True)

    if epoch is None:
        # Find the model_dict from the latest epoch.
        epoch = len(glob.glob(trained_model_folder + model_name + "/voxelmorph_model_epoch_*.pth")) - 1

    model_dict_filename = trained_model_folder + model_name + "/voxelmorph_model_epoch_{}.pth".format(epoch)
    print(model_dict_filename)

    #Load the statedict to the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_dict_filename,map_location=torch.device(device)))
    model.to(device)
    model.eval()
    return model






