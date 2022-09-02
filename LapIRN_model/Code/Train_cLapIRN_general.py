import glob
import os
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone
import json

import numpy as np
import torch
from dataset_generator import *

from Functions import generate_grid, transform_unit_flow_to_flow_cuda, \
    generate_grid_unit
from miccai2021_model import Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1, \
    Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2, Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3, \
    SpatialTransform_unit, SpatialTransformNearest_unit, smoothloss, \
    neg_Jdet_loss, NCC, multi_resolution_NCC, Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl_general

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--epochs_lvl1", type=int,
                    dest="epochs_lvl1", default=50,
                    help="number of lvl1 epochs")
parser.add_argument("--epochs_lvl2", type=int,
                    dest="epochs_lvl2", default=30,
                    help="number of lvl2 epochs")
parser.add_argument("--epochs_lvl3", type=int,
                    dest="epochs_lvl3", default=30,
                    help="number of lvl3 epochs")
parser.add_argument("--antifold", type=float,
                    dest="antifold", default=0.,
                    help="Anti-fold loss: suggested range 1 to 10000")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=5000,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=7,  # default:8, 7 for stage
                    help="number of start channels")
# parser.add_argument("--datapath", type=str,
#                     dest="datapath",
#                     default='../Dataset/Brain_dataset/OASIS/crop_min_max/norm',
#                     help="data path for training images")
parser.add_argument("--freeze_step", type=int,
                    dest="freeze_step", default=1000,
                    help="Number of step to freeze the previous level")
opt = parser.parse_args()

lr = opt.lr
start_channel = opt.start_channel
antifold = opt.antifold
n_checkpoint = opt.checkpoint
# datapath = opt.datapath
freeze_step = opt.freeze_step

epochs_lvl1_min = 7
epochs_lvl2_min = 7
epochs_lvl3_min = 7
epochs_lvl1_max = opt.epochs_lvl1
epochs_lvl2_max = opt.epochs_lvl2
epochs_lvl3_max = opt.epochs_lvl3


def dataset_generators(batch_size):
    print("batch_size:", batch_size)
    dimensions = [0, 80, 0, 256, 0, 256]
    shift = [5, 5, 15, 0]
    root_path_data = "/scratch/thomasvanderme/4D-Lung-256-h5/"
    # root_path_data = "C:/users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-256-h5/"

    with open(root_path_data + "scan_dictionary.json", 'r') as file:
        ct_path_dict = json.load(file)

    patient_id_training = ["107", "109", "110", "111", "112", "113", "114", "115", "116", "117", "118", "119"]
    scan_id_training = ["06-02-1999-p4-89680"]
    patient_id_training = ["107"]

    # single patient, single scan
    training_scan_keys = scan_key_generator(ct_path_dict, patient_id_training, scan_id_training)

    # patient_id_validation = ["108"]
    # scan_id_validation = None
    #
    # validation_scan_keys = scan_key_generator(ct_path_dict, patient_id_validation, scan_id_validation)
    # validation_batches = 20
    # random_validation_keys = sample(validation_scan_keys, batch_size * validation_batches)
    random_validation_keys = [['108', '07-28-1999-p4-56648', ['10', '40']],
                              ['108', '06-15-1999-p4-07025', ['90', '80']],
                              ['108', '07-02-1999-p4-21843', ['10', '0']],
                              ['108', '07-08-1999-p4-52902', ['80', '40']],
                              ['108', '06-21-1999-p4-15001', ['50', '40']],
                              ['108', '07-02-1999-p4-21843', ['20', '60']],
                              ['108', '07-28-1999-p4-56648', ['0', '90']],
                              ['108', '06-21-1999-p4-15001', ['60', '80']],
                              ['108', '07-02-1999-p4-21843', ['80', '10']],
                              ['108', '07-08-1999-p4-52902', ['10', '80']],
                              ['108', '07-02-1999-p4-21843', ['0', '0']],
                              ['108', '07-28-1999-p4-56648', ['20', '20']],
                              ['108', '07-28-1999-p4-56648', ['0', '0']],
                              ['108', '07-28-1999-p4-56648', ['0', '30']],
                              ['108', '07-08-1999-p4-52902', ['90', '40']],
                              ['108', '07-28-1999-p4-56648', ['90', '70']],
                              ['108', '07-28-1999-p4-56648', ['50', '90']],
                              ['108', '07-08-1999-p4-52902', ['40', '20']],
                              ['108', '06-15-1999-p4-07025', ['0', '0']],
                              ['108', '06-21-1999-p4-15001', ['20', '90']],
                              ['108', '06-21-1999-p4-15001', ['90', '30']],
                              ['108', '07-02-1999-p4-21843', ['40', '90']],
                              ['108', '07-28-1999-p4-56648', ['20', '50']],
                              ['108', '07-28-1999-p4-56648', ['70', '20']],
                              ['108', '07-28-1999-p4-56648', ['80', '0']],
                              ['108', '07-02-1999-p4-21843', ['10', '30']],
                              ['108', '07-02-1999-p4-21843', ['0', '60']],
                              ['108', '07-28-1999-p4-56648', ['30', '80']],
                              ['108', '07-02-1999-p4-21843', ['70', '50']],
                              ['108', '07-28-1999-p4-56648', ['0', '80']],
                              ['108', '06-21-1999-p4-15001', ['40', '90']],
                              ['108', '06-21-1999-p4-15001', ['80', '70']],
                              ['108', '06-21-1999-p4-15001', ['0', '80']],
                              ['108', '06-15-1999-p4-07025', ['90', '90']],
                              ['108', '07-02-1999-p4-21843', ['50', '20']],
                              ['108', '07-02-1999-p4-21843', ['40', '50']],
                              ['108', '07-02-1999-p4-21843', ['90', '80']],
                              ['108', '06-21-1999-p4-15001', ['50', '70']],
                              ['108', '07-28-1999-p4-56648', ['0', '60']],
                              ['108', '07-28-1999-p4-56648', ['70', '40']]]

    print("Number of training samples:", len(training_scan_keys), flush=True)
    print("Number of validation samples:", len(random_validation_keys), flush=True)
    # print(random_validation_keys, flush=True)
    # Generate datasets
    training_generator = generate_dataset(training_scan_keys, root_path_data, ct_path_dict, dimensions, shift,
                                          batch_size)
    validation_set = generate_dataset(random_validation_keys, root_path_data, ct_path_dict, dimensions, shift,
                                      batch_size=1)
    return training_generator, validation_set, len(training_scan_keys), len(random_validation_keys)


def train_routine(model, model_name, model_number, batch_size, epoch_min, epoch_max, image_shape, loss_criterium, freeze_step=None):
    print("starting training routine", flush = True)
    loss_similarity = NCC(win=3)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    grid = generate_grid(image_shape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training_generator, validation_set, number_of_training_samples, number_of_validation_samples = dataset_generators(
        batch_size)

    mean_old_loss = -10
    relative_loss = -1
    epoch = 0

    while (epoch < epoch_max and (relative_loss > loss_criterium or relative_loss < 0)) or epoch <= epoch_min:
        step_training = 0
        step_validation = 0
        loss_training = np.zeros((4, int(number_of_training_samples / batch_size)))
        loss_validation = np.zeros((4, int(number_of_validation_samples)))

        for X, Y, scan_id in training_generator:
            # print(scan_id, flush=True)

            X = X.cuda().float()
            Y = Y.cuda().float()
            reg_code = torch.rand(1, dtype=X.dtype, device=X.device).unsqueeze(dim=0)

            # (F_X_Y, X_Y, Y_4x, F_xy, _ )= model(X, Y, reg_code)
            prediction = model(X, Y, reg_code)
            F_X_Y = prediction[0]
            X_Y = prediction[1]
            Y_4x = prediction[2]

            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid)

            _, _, x, y, z = F_X_Y.shape
            norm_vector = torch.zeros((1, 3, 1, 1, 1), dtype=F_X_Y.dtype, device=F_X_Y.device)
            norm_vector[0, 0, 0, 0, 0] = (z - 1)
            norm_vector[0, 1, 0, 0, 0] = (y - 1)
            norm_vector[0, 2, 0, 0, 0] = (x - 1)
            loss_regulation = loss_smooth(F_X_Y * norm_vector)

            smo_weight = reg_code * max_smooth
            loss = loss_multiNCC + antifold * loss_Jacobian + smo_weight * loss_regulation

            loss_training[:, step_training] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])

            del loss_multiNCC
            del loss_Jacobian
            del smo_weight
            torch.cuda.empty_cache()

            optimizer.zero_grad()  # clear gradients for this training step_
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            del loss
            del prediction
            del F_X_Y
            del X_Y
            del Y_4x
            torch.cuda.empty_cache()

            # with lr 1e-3 + with bias
            step_training += 1
            if step_training % 100 == 0:
                print(step_training, flush=True)

            if freeze_step is not None and epoch == 0 and step_training == freeze_step // batch_size:
                model.unfreeze_model_previous()

        with torch.no_grad():
            for X, Y, scan_id in validation_set:
                X = X.cuda().float()
                Y = Y.cuda().float()
                reg_code = torch.rand(1, dtype=X.dtype, device=X.device).unsqueeze(dim=0)

                # (F_X_Y, X_Y, Y_4x, F_xy, _) = model(X, Y, reg_code)
                prediction = model(X, Y, reg_code)
                F_X_Y = prediction[0]
                X_Y = prediction[1]
                Y_4x = prediction[2]

                loss_multiNCC = loss_similarity(X_Y, Y_4x)

                F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

                loss_Jacobian = loss_Jdet(F_X_Y_norm, grid)

                _, _, x, y, z = F_X_Y.shape
                norm_vector = torch.zeros((1, 3, 1, 1, 1), dtype=F_X_Y.dtype, device=F_X_Y.device)
                norm_vector[0, 0, 0, 0, 0] = (z - 1)
                norm_vector[0, 1, 0, 0, 0] = (y - 1)
                norm_vector[0, 2, 0, 0, 0] = (x - 1)
                loss_regulation = loss_smooth(F_X_Y * norm_vector)

                smo_weight = reg_code * max_smooth
                loss = loss_multiNCC + antifold * loss_Jacobian + smo_weight * loss_regulation

                loss_validation[:, step_validation] = np.array(
                    [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
                step_validation += 1


                del loss_multiNCC
                del loss_Jacobian
                del smo_weight
                del loss
                del prediction
                del F_X_Y
                del X_Y
                del Y_4x
                torch.cuda.empty_cache()

            relative_loss = (loss_validation[0, :].mean() - mean_old_loss) / mean_old_loss
            mean_old_loss = loss_validation[0, :].mean()
            print("mean epoch validation loss:", mean_old_loss, flush=True)
            print("relative loss:", relative_loss, flush=True)

        print("Epoch {} of {}".format(epoch + 1, epoch_max))
        print(datetime.now(), flush=True)
        modelname = model_dir + '/' + model_name + "stagelvl{}_".format(model_number) + str(epoch) + '.pth'
        torch.save(model.state_dict(), modelname)
        np.save(model_dir + '/training_loss' + model_name + "stagelvl{}_".format(model_number) + str(epoch) + '.npy',
                loss_training)
        np.save(model_dir + '/validation_loss' + model_name + "stagelvl{}_".format(model_number) + str(epoch) + '.npy',
                loss_validation)
        epoch += 1

    return model



imgshape = (80, 256, 256)
imgshape_4 = (imgshape[0] / 4, imgshape[1] / 4, imgshape[2] / 4)
imgshape_2 = (imgshape[0] / 2, imgshape[1] / 2, imgshape[2] / 2)

range_flow = 0.4
max_smooth = 10.
start_t = datetime.now()

model_name = "test_"
root_path_saving = '/scratch/thomasvanderme/saved_models/'
# root_path_saving = "C:/users/pje33/Google Drive/Sync/TU_Delft/MEP/saved_models/"


model_dir = root_path_saving + "training_{}_{}".format(
    datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S"), model_name)
# model_dir = root_path_saving + "test_folder"
print(model_dir, flush=True)

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
loss_criterium = 0.05

model_lvl1 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True,
                                                                imgshape=imgshape_4,
                                                                range_flow=range_flow).cuda()
print("Train model 1", flush=True)

model_lvl1_trained = train_routine(model_lvl1, model_name, model_number = 1, batch_size = 20, epoch_min = 0, epoch_max = 1, image_shape=imgshape_4,
              loss_criterium=loss_criterium, freeze_step=None)
del model_lvl1
torch.cuda.empty_cache()
model_lvl2 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl_general(2, 3, start_channel, is_train=True,
                                                                     imgshape=imgshape_2,
                                                                     range_flow=range_flow,
                                                                     model_previous=model_lvl1_trained).cuda()

print("Train model 2", flush=True)
model_lvl2_trained = train_routine(model_lvl2, model_name, model_number = 2, batch_size = 2, epoch_min = 0, epoch_max = 1, image_shape=imgshape_2,
              loss_criterium=loss_criterium, freeze_step=60)
del model_lvl2
del model_lvl1_trained
torch.cuda.empty_cache()

model_lvl3 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl_general(2, 3, start_channel, is_train=True,
                                                                     imgshape=imgshape,
                                                                     range_flow=range_flow,
                                                                     model_previous=model_lvl2_trained).cuda()
del model_lvl2_trained
torch.cuda.empty_cache()
print("Train model 3", flush=True)

model_lvl3_trained = train_routine(model_lvl3, model_name, model_number = 3, batch_size = 2, epoch_min = 0, epoch_max = 1, image_shape=imgshape,
              loss_criterium=loss_criterium, freeze_step=20)

del model_lvl3

model_lvl4 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl_general(2, 3, start_channel, is_train=True,
                                                                     imgshape=imgshape_2,
                                                                     range_flow=range_flow,
                                                                     model_previous=model_lvl3_trained).cuda()
del model_lvl3_trained
torch.cuda.empty_cache()
print("Train model 5", flush=True)

model_lvl4_trained = train_routine(model_lvl4, model_name, model_number = 4, batch_size = 1, epoch_min = 0, epoch_max = 1, image_shape=imgshape_2,
              loss_criterium=loss_criterium, freeze_step=60)

del model_lvl3_trained
torch.cuda.empty_cache()

model_lvl5 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl_general(2, 3, start_channel, is_train=True,
                                                                     imgshape=imgshape,
                                                                     range_flow=range_flow,
                                                                     model_previous=model_lvl4_trained).cuda()
print("Train model 5", flush=True)

model_lvl5_trained = train_routine(model_lvl5, model_name, model_number = 5, batch_size = 1, epoch_min = 0, epoch_max = 1, image_shape=imgshape,
              loss_criterium=loss_criterium, freeze_step=20)

del model_lvl4_trained
torch.cuda.empty_cache()

# # time
end_t = datetime.now()
total_t = end_t - start_t
print("Time: ", total_t.total_seconds())
