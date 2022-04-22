def read_model_parameters_from_file(model_path, filename="training_parameters.txt"):
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


model_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/saved_models/training_2022_04_20_10_56_34/"

learning_rate, epochs, batch_size, loss_weights, patient_id, scan_id_training, scan_id_validation, validation_batches, \
nb_features, data_shape = read_model_parameters_from_file(model_path)
#
# x_data = torch.tensor(CT_data_0[np.newaxis,np.newaxis,  ...], dtype=torch.float)
# y_data = torch.tensor(CT_data_90[np.newaxis,np.newaxis, ...], dtype=torch.float)
#
# prediction, pos_flow = model(x_data, y_data)
#
# predicted_CT = prediction[0,0,...].detach().numpy()
# np.save("predicted_CT.npy",prediction[0,0,...].detach().numpy())
#
# predicted_CT_epoch_1 = np.load("predicted_CT_epoch_1.npy")
# predicted_CT_epoch_2 = np.load("predicted_CT_epoch_2.npy")
# predicted_CT_epoch_3 = np.load("predicted_CT_epoch_3.npy")
#
# slice_viewer([CT_data_90,predicted_CT_epoch_1,predicted_CT_epoch_2,predicted_CT_epoch_3])
