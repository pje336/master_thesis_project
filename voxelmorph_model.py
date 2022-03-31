import torch


def train(vxm_model, dataset, epochs, batch_size, learning_rate, losses, loss_weights):
    """
    Training routine for voxelmorph model using dataset.
    return trained model

    :param vxm_model: [torch model] Voxelmorph  model.
    :param dataset: DataLoader of ct_dataset class with fixed and moving image pairs.
    :param epochs: [int] number of epochs
    :param batch_size: [int] batch size before updating weights.
    :param learning_rate: [float] Set learning rate for optimiser.
    :param losses: [array] array of loss functions from voxelmorph
    :param loss_weights: [array] array with weight for each loss function
    :return: vxm_model: [torch model] trained Voxelmorph model
    """

    torch.backends.cudnn.deterministic = True
    vxm_model.train()
    optimizer = torch.optim.Adam(vxm_model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print("epoch {} of {}".format(epoch + 1, epochs))
        # Make a new iterator of the dataset each epoch
        data_iterator = iter(dataset)
        # Reset optimizer and loss each epoch.
        optimizer.zero_grad()
        loss = 0

        # iterate over all image pairs in the dataset.
        for i in range(len(data_iterator)):
            # Obtain next image pare from iterator.
            fixed_tensor, moving_tensor = data_iterator.next()
            prediction = vxm_model(moving_tensor, fixed_tensor)

            # Calculate loss for all the loss functions.
            for j, loss_function in enumerate(losses):
                loss += loss_function(fixed_tensor, prediction[j]) * loss_weights[j]
            print("loss:", loss)

            # After batch_size number of samples, update the weights.
            if (i + 1) % batch_size == 0:
                print("updating weights")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss = 0

        # Update weights with de last samples
        if loss != 0:
            loss.backward()
            optimizer.step()

    return vxm_model
