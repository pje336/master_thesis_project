import glob
import os
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone
import json

import numpy as np
from Network_Functions.dataset_generator import *

from Functions import generate_grid, transform_unit_flow_to_flow_cuda, \
    generate_grid_unit
from miccai2021_model import Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1, \
    Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2, Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3, \
    SpatialTransform_unit, SpatialTransformNearest_unit, smoothloss, \
    neg_Jdet_loss, NCC, multi_resolution_NCC

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--epochs_lvl1", type=int,
                    dest="epochs_lvl1", default=1,
                    help="number of lvl1 epochs")
parser.add_argument("--epochs_lvl2", type=int,
                    dest="epochs_lvl2", default=1,
                    help="number of lvl2 epochs")
parser.add_argument("--epochs_lvl3", type=int,
                    dest="epochs_lvl3", default=1,
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
batch_size = 4
epochs_lvl1 = opt.epochs_lvl1
epochs_lvl2 = opt.epochs_lvl2
epochs_lvl3 = opt.epochs_lvl3

model_name = "LDR_model"


def dataset_generators(batch_size):
    print("batch_size:", batch_size)
    dimensions = [0, 80, 0, 256, 0, 256]
    shift = [5, 5, 15, 0]
    root_path_data = "/scratch/thomasvanderme/4D-Lung-256-h5/"
    # root_path_data = "C:/users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-256-h5/"


    with open(root_path_data + "scan_dictionary.json", 'r') as file:
        ct_path_dict = json.load(file)

    patient_id_training = ["107", "109", "110", "111", "112", "113", "114", "115", "116", "117", "118", "119"]
    scan_id_training = None

    # single patient, single scan
    training_scan_keys = scan_key_generator(ct_path_dict, patient_id_training)

    # patient_id_validation = ["108"]
    # scan_id_validation = None
    #
    # validation_scan_keys = scan_key_generator(ct_path_dict, patient_id_validation, scan_id_validation)
    # validation_batches = 20
    # random_validation_keys = sample(validation_scan_keys, batch_size * validation_batches)
    random_validation_keys = [['108', '07-28-1999-p4-56648', ['10', '40']],
                              ['108', '06-15-1999-p4-07025', ['90', '80']], ['108', '07-02-1999-p4-21843', ['10', '0']],
                              ['108', '07-08-1999-p4-52902', ['80', '40']],
                              ['108', '06-21-1999-p4-15001', ['50', '40']],
                              ['108', '07-02-1999-p4-21843', ['20', '60']], ['108', '07-28-1999-p4-56648', ['0', '90']],
                              ['108', '06-21-1999-p4-15001', ['60', '80']],
                              ['108', '07-02-1999-p4-21843', ['80', '10']],
                              ['108', '07-08-1999-p4-52902', ['10', '80']], ['108', '07-02-1999-p4-21843', ['0', '0']],
                              ['108', '07-28-1999-p4-56648', ['20', '20']], ['108', '07-28-1999-p4-56648', ['0', '0']],
                              ['108', '07-28-1999-p4-56648', ['0', '30']], ['108', '07-08-1999-p4-52902', ['90', '40']],
                              ['108', '07-28-1999-p4-56648', ['90', '70']],
                              ['108', '07-28-1999-p4-56648', ['50', '90']],
                              ['108', '07-08-1999-p4-52902', ['40', '20']], ['108', '06-15-1999-p4-07025', ['0', '0']],
                              ['108', '06-21-1999-p4-15001', ['20', '90']],
                              ['108', '06-21-1999-p4-15001', ['90', '30']],
                              ['108', '07-02-1999-p4-21843', ['40', '90']],
                              ['108', '07-28-1999-p4-56648', ['20', '50']],
                              ['108', '07-28-1999-p4-56648', ['70', '20']], ['108', '07-28-1999-p4-56648', ['80', '0']],
                              ['108', '07-02-1999-p4-21843', ['10', '30']], ['108', '07-02-1999-p4-21843', ['0', '60']],
                              ['108', '07-28-1999-p4-56648', ['30', '80']],
                              ['108', '07-02-1999-p4-21843', ['70', '50']], ['108', '07-28-1999-p4-56648', ['0', '80']],
                              ['108', '06-21-1999-p4-15001', ['40', '90']],
                              ['108', '06-21-1999-p4-15001', ['80', '70']], ['108', '06-21-1999-p4-15001', ['0', '80']],
                              ['108', '06-15-1999-p4-07025', ['90', '90']],
                              ['108', '07-02-1999-p4-21843', ['50', '20']],
                              ['108', '07-02-1999-p4-21843', ['40', '50']],
                              ['108', '07-02-1999-p4-21843', ['90', '80']],
                              ['108', '06-21-1999-p4-15001', ['50', '70']], ['108', '07-28-1999-p4-56648', ['0', '60']],
                              ['108', '07-28-1999-p4-56648', ['70', '40']]]

    print("Number of training samples:", len(training_scan_keys), flush=True)
    print("Number of validation samples:", len(random_validation_keys), flush=True)
    # print(random_validation_keys, flush=True)
    # Generate datasets
    training_generator = generate_dataset(training_scan_keys, root_path_data, ct_path_dict, dimensions, shift,
                                          batch_size)
    validation_set = generate_dataset(random_validation_keys, root_path_data, ct_path_dict, dimensions, shift,
                                      batch_size=10)
    return training_generator, len(training_scan_keys)


root_path_saving = '/scratch/thomasvanderme/saved_models/'
# root_path_saving = "C:/users/pje33/Google Drive/Sync/TU_Delft/MEP/saved_models/"

model_dir = root_path_saving + "training_{}".format(
    datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S"))
print(model_dir, flush=True)


if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

def train_lvl1(model_dir, batch_size):
    print("Training lvl1...")
    model = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True,
                                                                    imgshape=imgshape_4,
                                                                    range_flow=range_flow).cuda()

    loss_similarity = NCC(win=3)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # OASIS
    # names = sorted(glob.glob(datapath + '/*.nii'))

    grid_4 = generate_grid(imgshape_4)
    grid_4 = torch.from_numpy(np.reshape(grid_4, (1,) + grid_4.shape)).cuda().float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # edited by Thomas:
    # add training_dataset
    training_generator, number_of_samples = dataset_generators(batch_size)

    lossall = np.zeros((4, int(epochs_lvl1 * number_of_samples / batch_size) + 2))
    print(np.shape(lossall))

    epoch = 0
    step = 0

    load_model = False
    if load_model is True:
        model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
        print("Loading weight: ", model_path)
        step = 3000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
        lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    while epoch < epochs_lvl1:
        for X, Y, scan_id in training_generator:
            # print(scan_id, flush=True)

            X = X.cuda().float()
            Y = Y.cuda().float()
            reg_code = torch.rand(1, dtype=X.dtype, device=X.device).unsqueeze(dim=0)

            F_X_Y, X_Y, Y_4x, F_xy, _ = model(X, Y, reg_code)

            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_4)

            _, _, x, y, z = F_X_Y.shape
            norm_vector = torch.zeros((1, 3, 1, 1, 1), dtype=F_X_Y.dtype, device=F_X_Y.device)
            norm_vector[0, 0, 0, 0, 0] = (z - 1)
            norm_vector[0, 1, 0, 0, 0] = (y - 1)
            norm_vector[0, 2, 0, 0, 0] = (x - 1)
            loss_regulation = loss_smooth(F_X_Y * norm_vector)

            smo_weight = reg_code * max_smooth
            loss = loss_multiNCC + antifold * loss_Jacobian + smo_weight * loss_regulation

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f} -reg_c "{5:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item(),
                    reg_code[0].item()))
            sys.stdout.flush()

            del loss_multiNCC
            del loss_Jacobian
            del smo_weight
            torch.cuda.empty_cache()

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            del loss

            # with lr 1e-3 + with bias


            step += 1
            print(step)

        print("Epoch {} of {}".format(epoch, epochs_lvl1))
        print(datetime.now(), flush=True)
        modelname = model_dir + '/' + model_name + "stagelvl1_" + str(epoch) + '.pth'
        torch.save(model.state_dict(), modelname)
        np.save(model_dir + '/loss' + model_name + "stagelvl1_" + str(epoch) + '.npy', lossall)
        epoch += 1


def train_lvl2(model_dir, batch_size):
    print("Training lvl2...")
    model_lvl1 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True,
                                                                         imgshape=imgshape_4,
                                                                         range_flow=range_flow).cuda()

    print(model_dir + model_name, flush=True)
    model_path = sorted(glob.glob(model_dir +"/" + model_name + "stagelvl1_*pth"))[-1]
    model_lvl1.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl1...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl1.parameters():
        param.requires_grad = False

    model = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2(2, 3, start_channel, is_train=True,
                                                                    imgshape=imgshape_2,
                                                                    range_flow=range_flow, model_lvl1=model_lvl1).cuda()

    loss_similarity = multi_resolution_NCC(win=5, scale=2)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # OASIS
    # names = sorted(glob.glob(datapath + '/*.nii'))

    grid_2 = generate_grid(imgshape_2)
    grid_2 = torch.from_numpy(np.reshape(grid_2, (1,) + grid_2.shape)).cuda().float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    training_generator, number_of_samples = dataset_generators(batch_size)

    lossall = np.zeros((4, int(epochs_lvl2 * number_of_samples / batch_size) + 2))
    epoch = 0
    step = 0
    load_model = False
    if load_model is True:
        model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
        print("Loading weight: ", model_path)
        step = 3000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
        lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    while epoch < epochs_lvl2:
        for X, Y, scan_id in training_generator:

            X = X.cuda().float()
            Y = Y.cuda().float()
            reg_code = torch.rand(1, dtype=X.dtype, device=X.device).unsqueeze(dim=0)

            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, _ = model(X, Y, reg_code)

            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_2)

            _, _, x, y, z = F_X_Y.shape
            norm_vector = torch.zeros((1, 3, 1, 1, 1), dtype=F_X_Y.dtype, device=F_X_Y.device)
            norm_vector[0, 0, 0, 0, 0] = (z - 1)
            norm_vector[0, 1, 0, 0, 0] = (y - 1)
            norm_vector[0, 2, 0, 0, 0] = (x - 1)
            loss_regulation = loss_smooth(F_X_Y * norm_vector)

            smo_weight = reg_code * max_smooth
            loss = loss_multiNCC + antifold * loss_Jacobian + smo_weight * loss_regulation

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f} -reg_c "{5:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item(),
                    reg_code[0].item()))
            sys.stdout.flush()

            del loss_multiNCC
            del loss_Jacobian
            del smo_weight
            torch.cuda.empty_cache()

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            del loss

            # with lr 1e-3 + with bias
            if step == freeze_step // batch_size:
                model.unfreeze_modellvl1()

            step += 1

        print("Epoch {} of {}".format(epoch, epochs_lvl2))
        print(datetime.now(),flush=True)
        modelname = model_dir + '/' + model_name + "stagelvl2_" + str(epoch) + '.pth'
        torch.save(model.state_dict(), modelname)
        np.save(model_dir + '/loss' + model_name + "stagelvl2_" + str(epoch) + '.npy', lossall)
        epoch += 1


def train_lvl3(model_dir, batch_size):
    print("Training lvl3...")
    model_lvl1 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True,
                                                                         imgshape=imgshape_4,
                                                                         range_flow=range_flow).cuda()
    model_lvl2 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2(2, 3, start_channel, is_train=True,
                                                                         imgshape=imgshape_2,
                                                                         range_flow=range_flow,
                                                                         model_lvl1=model_lvl1).cuda()

    model_path = sorted(glob.glob(model_dir +"/"+ model_name + "stagelvl2_*.pth"))[-1]
    model_lvl2.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl2...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl2.parameters():
        param.requires_grad = False

    model = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3(2, 3, start_channel, is_train=True,
                                                                    imgshape=imgshape,
                                                                    range_flow=range_flow, model_lvl2=model_lvl2).cuda()

    loss_similarity = multi_resolution_NCC(win=7, scale=3)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().cuda()
    transform_nearest = SpatialTransformNearest_unit().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # OASIS
    # names = sorted(glob.glob(datapath + '/*.nii'))

    grid = generate_grid(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    grid_unit = generate_grid_unit(imgshape)
    grid_unit = torch.from_numpy(np.reshape(grid_unit, (1,) + grid_unit.shape)).cuda().float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    training_generator, number_of_samples = dataset_generators(batch_size)
    lossall = np.zeros((4, int(epochs_lvl3 * number_of_samples / batch_size) + 2))
    epoch = 0
    step = 0
    load_model = False
    if load_model is True:
        model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
        print("Loading weight: ", model_path)
        step = 3000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
        lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    while epoch < epochs_lvl3:
        for X, Y, scan_id in training_generator:

            X = X.cuda().float()
            Y = Y.cuda().float()
            reg_code = torch.rand(1, dtype=X.dtype, device=X.device).unsqueeze(dim=0)

            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y, reg_code)

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

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f} -reg_c "{5:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item(),
                    reg_code[0].item()))
            sys.stdout.flush()

            del loss_multiNCC
            del loss_Jacobian
            del smo_weight
            torch.cuda.empty_cache()

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            del loss

            # with lr 1e-3 + with bias

            if step == freeze_step // batch_size:
                model.unfreeze_modellvl2()

            step += 1

        print("Epoch {} of {}".format(epoch, epochs_lvl3))
        print(datetime.now(), flush=True)
        modelname = model_dir + '/' + model_name + "stagelvl3_" + str(epoch) + '.pth'
        torch.save(model.state_dict(), modelname)
        np.save(model_dir + '/loss' + model_name + "stagelvl3_" + str(epoch) + '.npy', lossall)
        epoch += 1


imgshape = (80, 256, 256)
imgshape_4 = (imgshape[0] / 4, imgshape[1] / 4, imgshape[2] / 4)
imgshape_2 = (imgshape[0] / 2, imgshape[1] / 2, imgshape[2] / 2)

range_flow = 0.4
max_smooth = 10.
start_t = datetime.now()
train_lvl1(model_dir, batch_size=20)
torch.cuda.empty_cache()
train_lvl2(model_dir, batch_size=10)
torch.cuda.empty_cache()
train_lvl3(model_dir, batch_size=2)
torch.cuda.empty_cache()

# # time
end_t = datetime.now()
total_t = end_t - start_t
print("Time: ", total_t.total_seconds())
