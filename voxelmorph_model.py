import torch
import numpy as np


def train_model(vxm_model, dataset, epochs, batch_size, learning_rate, losses, loss_weights):
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
    epocht_loss_array = []

    for epoch in range(epochs):
        epoch_loss = 0
        print("epoch {} of {}".format(epoch + 1, epochs))
        # Reset optimizer and loss each epoch.
        optimizer.zero_grad()
        loss = 0
        batch = 0

        # iterate over all image pairs in the dataset.

        for fixed_tensor, moving_tensor in dataset:
            batch += 1
            print("batch:", batch)
            loss = 0

            if fixed_tensor.shape[2] < 80 or moving_tensor.shape[2] < 80:
                print("To small")
                continue
            print(moving_tensor.shape)
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                moving_tensor = moving_tensor.to(device)
                fixed_tensor = fixed_tensor.to(device)

            # Obtain next image pare from iterator.
            prediction = vxm_model(moving_tensor, fixed_tensor)


            # Calculate loss for all the loss functions.
            print("calculating loss")
            for j, loss_function in enumerate(losses):
                loss += loss_function(fixed_tensor, prediction[j]) * loss_weights[j]
            print("loss:", loss)
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


        epocht_loss_array.append(epoch_loss/len(dataset))

    torch.cuda.empty_cache()
    return (vxm_model, epocht_loss_array)