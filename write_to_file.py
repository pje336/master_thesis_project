import os


def training_parameters_to_string(learning_rate, epochs, batch_size, loss_weights, patient_id, scan_id_training,
                        scan_id_validation, validation_batches):
    text = \
        f"""Training parameters for VoxelMorph network
learning_rate = {learning_rate}
epochs = {epochs}
batch_size = {batch_size}
loss_weights = {loss_weights}

patient_id = {patient_id}
scan_id_training = {scan_id_training}
scan_id_validation = {scan_id_validation}
validation_batches = {validation_batches}"""
    return text


def write_file(file_path, file_name, text):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    with open(file_path + file_name, "w") as file:
        file.write(text)

    return None

