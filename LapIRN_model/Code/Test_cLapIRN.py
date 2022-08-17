from argparse import ArgumentParser

import numpy as np
from Network_Functions.dataset_generator import *
import json
from Evaluation.slice_viewer_flow import slice_viewer




from Functions import generate_grid_unit, transform_unit_flow_to_flow
from miccai2021_model import Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1, \
    Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2, Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3, \
    SpatialTransform_unit, SpatialTransformNearest_unit


parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    dest="modelpath", default='training_2022_08_16_11_04_43',
                    help="Trained model path")
parser.add_argument("--savepath", type=str,
                    dest="savepath", default='../Result',
                    help="path for saving images")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=7,
                    help="number of start channels")

parser.add_argument("--reg_input", type=float,
                    dest="reg_input", default=0.4,
                    help="Normalized smoothness regularization (within [0,1])")
opt = parser.parse_args()
epoch = 15000
opt.modelpath = "C:/users/pje33/Google Drive/Sync/TU_Delft/MEP/saved_models/" + opt.modelpath + "/LDR_OASIS_NCC_unit_disp_add_fea7_reg01_10_testing_stagelvl3_{}.pth".format(epoch)

# .\venv\Scripts\activate

start_channel = opt.start_channel
reg_input = opt.reg_input


def dataset_generators(batch_size):
    print("batch_size:", batch_size)
    dimensions = [0, 80, 0, 256, 0, 256]
    shift = [5, 5, 15, 0]
    root_path_data = "C:/users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-256-h5/"

    with open(root_path_data + "scan_dictionary.json", 'r') as file:
        ct_path_dict = json.load(file)

    patient_id_training = ["107", "109", "110", "111", "112", "113", "114", "115", "116", "117", "118", "119"]
    # patient_id_training = ["107"]

    # scan_id_training = ["06-02-1999-p4-89680"]
    scan_id_training = None

    # single patient, single scan
    training_scan_keys = scan_key_generator(ct_path_dict, patient_id_training)

    patient_id_validation = ["108"]
    # scan_id_validation = ["06-15-1999-p4-07025"]
    scan_id_validation = None

    validation_scan_keys = scan_key_generator(ct_path_dict, patient_id_validation, scan_id_validation)
    validation_batches = 20
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
    print(random_validation_keys, flush=True)
    # Generate datasets
    training_generator = generate_dataset(training_scan_keys, root_path_data, ct_path_dict, dimensions, shift,
                                          batch_size)

    return training_generator, len(training_scan_keys)


def test():
    print("Current reg_input: ", str(reg_input))

    model_lvl1 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                                                         range_flow=range_flow)
    model_lvl2 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                                                         range_flow=range_flow, model_lvl1=model_lvl1)

    model = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3(2, 3, start_channel, is_train=False, imgshape=imgshape,
                                                                    range_flow=range_flow, model_lvl2=model_lvl2)


    transform = SpatialTransform_unit().cuda()
    transform_nearest = SpatialTransformNearest_unit()

    model.load_state_dict(torch.load(opt.modelpath,map_location=torch.device('cpu')))
    model.eval()
    transform.eval()

    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).float()

    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset, _ = dataset_generators(batch_size = 1)

    for fixed_img, moving_img, scan_id in dataset:
    # for (fixed_img, moving_img, scan_id) in dataset:
        print(scan_id)
        #
        # fixed_img = torch.from_numpy(fixed_img).float().to(device).unsqueeze(dim=0)
        # moving_img = torch.from_numpy(moving_img).float().to(device).unsqueeze(dim=0)

        with torch.no_grad():
            reg_code = torch.tensor([reg_input], dtype=fixed_img.dtype, device=fixed_img.device).unsqueeze(dim=0)

            F_X_Y = model(moving_img, fixed_img, reg_code)

            image = transform(moving_img, F_X_Y.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]

            F_X_Y_cpu = F_X_Y.data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
            flow = transform_unit_flow_to_flow(F_X_Y_cpu)

            source_array = moving_img[0, 0].detach().numpy()
            target = fixed_img[0, 0].detach().numpy()
            slice_viewer([source_array,image, target],["source_array","prediction", "target"] , flow_field=flow)
            print(np.shape(image))



if __name__ == '__main__':
    imgshape = (80, 256, 256)
    imgshape_4 = (imgshape[0] / 4, imgshape[1] / 4, imgshape[2] / 4)
    imgshape_2 = (imgshape[0] / 2, imgshape[1] / 2, imgshape[2] / 2)

    range_flow = 0.4
    test()
