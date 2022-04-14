import torch


def train_model(vxm_model, train_dataset, validation_dataset, epochs, learning_rate, losses, loss_weights):
    """
    Training routine for voxelmorph model using dataset.
    For each epoch the model is trained on the train_dataset and then is validated on the validation_dataset.
    The average loss of each epoch is added to the epoch_training_loss and epoch_validation_loss arrays.
    The function return the trained model and the epoch_training_loss and epoch_validation_loss arrays.

    :param vxm_model: [torch model] Voxelmorph  model.
    :param train_dataset: DataLoader of ct_dataset class with fixed and moving image pairs for training.
    :param validation_dataset: DataLoader of ct_dataset class with fixed and moving image pairs for validation.
    :param epochs: [int] number of epochs
    :param learning_rate: [float] Set learning rate for optimiser.
    :param losses: [array] array of loss functions from voxelmorph
    :param loss_weights: [array] array with weight for each loss function
    :return: vxm_model: [torch model] Trained Voxelmorph model
    :return epoch_training_loss: [1d array] Array with the mean training loss for each epoch.
    :return epoch_validation_loss: [1d array] Array with the mean validation loss for each epoch.
    """

    torch.backends.cudnn.deterministic = True
    optimizer = torch.optim.Adam(vxm_model.parameters(), lr=learning_rate)
    epoch_training_loss = []
    epoch_validation_loss = []

    for epoch in range(epochs):
        vxm_model.train()
        epoch_loss = 0
        # Reset optimizer and loss each epoch.
        optimizer.zero_grad()
        loss = 0
        batch = 0

        # iterate over all image pairs in the dataset.

        for fixed_tensor, moving_tensor in train_dataset:
            batch += 1
            loss = 0

            if fixed_tensor.shape[2] < 80 or moving_tensor.shape[2] < 80:
                print("To small")
                continue

            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                moving_tensor = moving_tensor.to(device)
                fixed_tensor = fixed_tensor.to(device)

            # Obtain next image pare from iterator.
            prediction = vxm_model(moving_tensor, fixed_tensor)

            # Calculate loss for all the loss functions.
            for j, loss_function in enumerate(losses):
                loss += loss_function(fixed_tensor, prediction[j]) * loss_weights[j]
            print("epoch {} of {}, Batch: {} : Loss: {}".format(epoch + 1, epochs, batch, float(loss)))

            epoch_loss += float(loss)

            # Remove variables to create more space in ram.
            del moving_tensor
            del prediction
            torch.cuda.empty_cache()

            print("updating weights")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            del loss

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        epoch_training_loss.append(epoch_loss / len(train_dataset))

        # Here we will validate the model with data from the validation_dataset.
        with torch.no_grad():
            validation_batches = 0
            validation_loss = 0
            for fixed_tensor, moving_tensor in validation_dataset:
                validation_batches += 1
                loss = 0
                prediction = vxm_model(moving_tensor, fixed_tensor)

                # Calculate loss for all the loss functions.
                for j, loss_function in enumerate(losses):
                    loss += loss_function(fixed_tensor, prediction[j]) * loss_weights[j]
                validation_loss += float(loss)

            epoch_validation_loss.append(validation_loss / validation_batches)
            print("validation loss: {}".format(epoch_validation_loss[-1]))

    torch.cuda.empty_cache()
    return vxm_model, epoch_training_loss, epoch_validation_loss
