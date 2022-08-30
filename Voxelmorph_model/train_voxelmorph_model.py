import torch
from datetime import datetime, timezone
from Network_Functions.write_parameters_to_file import write_string_to_file
import voxelmorph

def train_model(vxm_model, optimizer, train_dataset, validation_dataset, epochs, batch_size, losses, loss_weights,
                saving_file_path, steps_per_epoch,  batches_per_step):
    """
    Training routine for voxelmorph model using dataset.
    For each epoch the model is trained on the train_dataset and then is validated on the validation_dataset.
    The average loss of each epoch is added to the epoch_training_loss and epoch_validation_loss arrays.
    The function return the trained model and the epoch_training_loss and epoch_validation_loss arrays.


    Args:
        vxm_model: [torch model] Voxelmorph  model.
        optimizer: optimizer
        train_dataset: DataLoader of ct_dataset class with fixed and moving image pairs for training.
        validation_dataset: DataLoader of ct_dataset class with fixed and moving image pairs for validation.
        epochs: [int] number of epochs
        batch_size: [int] number of samples per batch
        losses: [array] array of loss functions from voxelmorph
        loss_weights: [array] array with weight for each loss function
        saving_file_path: [string] String with the filepath to the folder to save the model and text files.
        steps_per_epoch: [int] Batches per epoch before performing validation and saving of the model
        batches_per_step: [int] Batches to do before updating weights.

    Returns:
        vxm_model: [torch model] Trained Voxelmorph model
        epoch_training_loss: [1d array] Array with the mean training loss for each epoch.
        epoch_validation_loss: [1d array] Array with the mean validation loss for each epoch.

    """

    torch.backends.cudnn.deterministic = True


    epoch_training_loss = []
    epoch_validation_loss = []
    vxm_model.train()
    optimizer.zero_grad()
    print("Start training", flush=True )
    print(datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S"), flush=True)

    for epoch in range(epochs):

        batch = 0
        epoch_loss = 0
        loss_total = 0

        # iterate over all image pairs in the dataset.
        for fixed_tensor, moving_tensor, scan_id in train_dataset:
            batch += 1

            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                moving_tensor = moving_tensor.to(device)
                fixed_tensor = fixed_tensor.to(device)
            # try:
            if True:
                # Obtain next image pare from iterator.
                prediction = vxm_model(moving_tensor, fixed_tensor)

                # Calculate/ loss for all the loss functions.

                for j, loss_function in enumerate(losses):
                    if j == 0:
                      loss_total = loss_function(fixed_tensor, prediction[j]) * loss_weights[j]
                    else:
                      loss_total += loss_function(fixed_tensor, prediction[j]) * loss_weights[j]
                loss_total += voxelmorph.py.utils.neg_Jdet_loss(prediction[-1][0].permute((1,2,3,0))) * 350000

                # Normalise the loss over the number of batches per step.
                loss_total /= (batches_per_step * batch_size)
                # Calculate the gradients
                loss_total.backward()

                # After every batches_per_step update the weights of the network
                if batch > 0 and batch % batches_per_step == 0:
                    # print("updating weights")

                    optimizer.step()
                    optimizer.zero_grad()

                    del loss_total
            # except:
            #     print("problem with the following scan:" + str(scan_id), flush=True)

        # Save the model
        print("save model", flush=True )
        print(datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S"), flush=True)
        epoch_training_loss.append(epoch_loss/batch)
        torch.save(vxm_model.state_dict(), saving_file_path + "voxelmorph_model_epoch_{}.pth".format(epoch))
        torch.save(optimizer.state_dict(), saving_file_path + "optimiser_epoch_{}.pth".format(epoch))
        write_string_to_file(saving_file_path, "training_loss.txt", str(epoch_training_loss))

        # testing
        with torch.no_grad():
            validation_batches = 0
            validation_loss = 0
            for fixed_tensor, moving_tensor, _ in validation_dataset:
                if torch.cuda.is_available():
                    device = torch.device("cuda:0")
                    moving_tensor = moving_tensor.to(device)
                    fixed_tensor = fixed_tensor.to(device)
                validation_batches += 1
                loss_total = 0
                prediction = vxm_model(moving_tensor, fixed_tensor)

                # Calculate loss for all the loss functions.
                for j, loss_function in enumerate(losses):
                    loss_total += loss_function(fixed_tensor, prediction[j]) * loss_weights[j]
                loss_total += voxelmorph.py.utils.neg_Jdet_loss(prediction[-1][0].permute((1,2,3,0))) * 350000
                validation_loss += float(loss_total)

            epoch_validation_loss.append(validation_loss / validation_batches)
        write_string_to_file(saving_file_path, "validation_loss.txt", str(epoch_validation_loss))


    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vxm_model, epoch_training_loss, epoch_validation_loss
