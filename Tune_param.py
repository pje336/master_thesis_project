### This is for tuning !!!!!1
from ray import tune
from datetime import datetime, timezone
from random import sample
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

from CT_path_dict.ct_path_dict import ct_path_dict
from Network_Functions.dataset_generator import *
from Network_Functions.write_parameters_to_file import *
import torch


#
def train_cifar(config):

    # .\venv\Scripts\activate
    # dataset parameters
    root_path = "/content/"
    dimensions = [0, 80, 0, 256, 0, 256]
    data_shape = [80, 256, 256]
    shift = [0, 0, 0, 0]
    int_downsize = 2
    dropout_rate = config["dropout"]
    learning_rate = config["lr"]

    # Network parameters for voxelmorph
    nb_features = [
        [16, 32, 32, 32],
        [32, 32, 32, 32, 32, 16, 16]]  # number of features of encoder and decoder
    # losses = [voxelmorph.losses.NCC().loss,voxelmorph.losses.NCC().loss, voxelmorph.losses.Grad('l2').loss]
    losses = [Voxelmorph_model.voxelmorph.torch.losses.MSE().loss, Voxelmorph_model.voxelmorph.torch.losses.MSE().loss, Voxelmorph_model.voxelmorph.torch.losses.Grad('l2').loss]

    loss_weights = [0.5, 0.5, 0.01]

    # Training parameters

    epochs = 25
    batch_size = 2

    print("Shape of dataset:", data_shape)
    steps_per_epoch = 200

    train = False

    vxm_model = Voxelmorph_model.voxelmorph.networks.VxmDense(data_shape, nb_features, int_downsize=int_downsize, bidir=True,
                                                              dropout=dropout_rate)
    if torch.cuda.is_available():
        vxm_model.cuda()  # If possible move model to GPU.
    # model.load_state_dict(torch.load(trained_model_path + trained_model_dict_name, map_location=device))
    optimizer = torch.optim.Adam(vxm_model.parameters(), lr=learning_rate)



    # if checkpoint_dir:
    #   model_state, optimizer_state = torch.load(
    #   os.path.join(checkpoint_dir, "checkpoint"))
    #   vxm_model.load_state_dict(model_state)
    #   optimizer.load_state_dict(optimizer_state)
    # train model

    saving_file_path = "/content/drive/MyDrive/Sync/TU_Delft/MEP/saved_models/training_{}/".format(
        datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S"))

    patient_id_training = ["107",]

    # scan_id_training = ["06-02-1999-p4-89680"]
    scan_id_training = None

    # single patient, single scan
    training_scan_keys = scan_key_generator(ct_path_dict, patient_id_training)

    patient_id_validation = ["108"]
    scan_id_validation = ["06-15-1999-p4-07025"]

    validation_scan_keys = scan_key_generator(ct_path_dict, patient_id_validation, scan_id_validation)
    validation_batches = 20
    random_validation_keys = sample(validation_scan_keys, batch_size * validation_batches)

    # print("Number of training samples:", len(training_scan_keys))
    # print("Number of validation samples:", len(random_validation_keys))

    # Generate datasets
    training_set = generate_dataset(training_scan_keys, root_path, ct_path_dict, dimensions, shift, batch_size)
    validation_set = generate_dataset(random_validation_keys, root_path, ct_path_dict, dimensions, shift, batch_size)

    # #
    torch.use_deterministic_algorithms(True)
    epoch_training_loss = []
    epoch_validation_loss = []
    iterationss = int((epochs * steps_per_epoch) / len(training_set)) + 1
    vxm_model.train()

    epoch = 0
    batch = 0
    epoch_loss = 0

    for iteration in range(iterationss):

        # iterate over all image pairs in the dataset.
        for fixed_tensor, moving_tensor, _ in training_set:
            batch += 1

            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                moving_tensor = moving_tensor.to(device)
                fixed_tensor = fixed_tensor.to(device)

            # Obtain next image pare from iterator.
            prediction = vxm_model(moving_tensor, fixed_tensor)

            # Calculate loss for all the loss functions.
            loss_total = 0
            loss_array = []
            for j, loss_function in enumerate(losses):
                loss = loss_function(fixed_tensor, prediction[j]) * loss_weights[j]
                loss_array.append(float(loss))
                loss_total += loss
            print("epoch {} of {}, Batch: {} - Loss: {}".format(epoch + 1, epochs, batch, loss_array))

            epoch_loss += float(loss_total)

            # Remove variables to create more space in ram.
            del moving_tensor
            del prediction
            torch.cuda.empty_cache()

            # Apply back propagation
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            del loss_total

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if batch == steps_per_epoch:
                epoch_training_loss.append(epoch_loss / batch)
                print("Training loss: {}".format(epoch_training_loss[-1]))

                # Here we will validate the model with data from the validation_dataset.
                with torch.no_grad():
                    validation_batches = 0
                    validation_loss = 0
                    for fixed_tensor, moving_tensor, _ in validation_set:
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
                        validation_loss += float(loss_total)

                    epoch_validation_loss.append(validation_loss / validation_batches)
                    print("validation loss: {}".format(epoch_validation_loss[-1]))
                    tune.report(mean_loss = epoch_validation_loss[-1])

                # Save the model
                print("save model")
                print(saving_file_path)
                torch.save(vxm_model.state_dict(), saving_file_path + "voxelmorph_model_epoch_{}.pth".format(epoch))
                torch.save(optimizer.state_dict(), saving_file_path + "optimiser_epoch_{}.pth".format(epoch))
                write_string_to_file(saving_file_path, "training_loss.txt", str(epoch_training_loss))
                write_string_to_file(saving_file_path, "validation_loss.txt", str(epoch_validation_loss))
                vxm_model.train()

                batch = 0
                epoch_loss = 0
                epoch += 1

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vxm_model, epoch_training_loss, epoch_validation_loss
    # for step in range(10):
    #     tune.report(mean_loss=object(step, config["lr"], config["dropout"]))


def main(num_samples=10, max_num_epochs=10):
    config = {
        "dropout": tune.uniform(0.1,1),
        "lr": tune.loguniform(10**(-5), 10**(-2))
    }
        #
        # "conv_1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 7)),
        # "conv_2": tune.sample_from(lambda _: 2 ** np.random.randint(3, 7)),
        # "conv_3": tune.sample_from(lambda _: 2 ** np.random.randint(4, 7)),
        # "conv_4": tune.sample_from(lambda _: 2 ** np.random.randint(4, 7)),
        # "conv_5": tune.sample_from(lambda _: 2 ** np.random.randint(4, 7)),
        # "conv_6": tune.sample_from(lambda _: 2 ** np.random.randint(4, 7)),
        # "conv_7": tune.sample_from(lambda _: 2 ** np.random.randint(4, 7)),
        # "conv_8": tune.sample_from(lambda _: 2 ** np.random.randint(4, 7)),
        # "conv_9": tune.sample_from(lambda _: 2 ** np.random.randint(4, 7)),
        # "conv_10": tune.sample_from(lambda _: 2 ** np.random.randint(2, 7)),
        # "conv_11": tune.sample_from(lambda _: 2 ** np.random.randint(2, 7))


    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "training_iteration"])
    result = tune.run(
        partial(train_cifar),
        resources_per_trial={"cpu": 2},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    # best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    # best_trained_model.to(device)

    # best_checkpoint_dir = best_trial.checkpoint.value
    # model_state, optimizer_state = torch.load(os.path.join(
    #     best_checkpoint_dir, "checkpoint"))
    # best_trained_model.load_state_dict(model_state)

    # test_acc = test_accuracy(best_trained_model, device)
    # print("Best trial test set accuracy: {}".format(test_acc))


main(num_samples=5, max_num_epochs=5)