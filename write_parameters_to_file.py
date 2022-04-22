import os

def training_parameters_to_string(learning_rate, epochs, batch_size, loss_weights, patient_id, scan_id_training,
                                  scan_id_validation, validation_batches, nb_features, data_shape):
    """
    Function to generate a string with the important training parameters of the network.
    Returns: A string with the information.

    """
    text = \
        f"""Training parameters for VoxelMorph network
learning_rate = {learning_rate}
epochs = {epochs}
batch_size = {batch_size}
loss_weights = {loss_weights}
patient_id = {patient_id}
scan_id_training = {scan_id_training}
scan_id_validation = {scan_id_validation}
validation_batches = {validation_batches}
nb_features = {nb_features}
data_shape = {data_shape}"""
    return text


def write_string_to_file(file_path, file_name, text):
    """ Writes text to file with file_name in folder from file_path.
    NOTE: This function overwrites the current file if it exists.
    Args:
        file_path: file path where the file should be written
        file_name: Name of the file
        text: [string] String of text to be writen in file.
    """

    # If the folder does not excist, make it.
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Write the text to the file.
    with open(file_path + file_name, "w") as file:
        file.write(text)

    return None

