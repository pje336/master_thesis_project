import torch
import glob
import Voxelmorph_model.voxelmorph as voxelmorph
from LapIRN_model.Code.Functions import generate_grid, transform_unit_flow_to_flow_cuda, \
    generate_grid_unit
from LapIRN_model.Code.miccai2021_model_temp import Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1, \
    SpatialTransform_unit, SpatialTransformNearest_unit, smoothloss, \
    neg_Jdet_loss, NCC, multi_resolution_NCC, Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl_general
from argparse import ArgumentParser
from datetime import datetime
import sparse
import json
from Network_Functions.dataset_generator import *
import numpy as np
import csv
from surface_distance.metrics import *
from skimage.util import compare_images
from Evaluation.lap_model.slice_viewer_flow import slice_viewer
from Evaluation.lap_model.contour_viewer import contour_viewer
import torch.nn as nn
import torch.nn.functional as nnf

import SimpleITK as sitk

parser = ArgumentParser()

parser.add_argument("--root_path_model", type=str,
                    dest="root_path_model",
                    default='../',
                    help="data path for stored model")

parser.add_argument("--root_path_CT_data", type=str,
                    dest="root_path_CT_data",
                    default='C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT',
                    help="data path for training images")





opt = parser.parse_args()
root_path_model = opt.root_path_model
root_path_CT_data = opt.root_path_CT_data

root_path_model = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/elastix/outputs"
root_path_CT_data = "C:/Users/pje33/Desktop/4d-lung/manifest-1665386976937/4D-Lung/"

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


def transform_flow_to_unit_flow(flow):
    z, x, y, _ = flow.shape
    new_flow = torch.zeros(flow.shape)
    new_flow[:, :, :, 1] = flow[:, :, :, 0] / (x-1)*2
    new_flow[:, :, :, 2] = flow[:, :, :, 1] / (y-1)*2
    new_flow[:, :, :, 0] = flow[:, :, :, 2] / (z-1)*2

    return new_flow


def transform_flow(flow):
    new_flow = torch.zeros(flow.shape)
    new_flow[:, :, :, 1] = flow[:, :, :, 0]
    new_flow[:, :, :, 2] = flow[:, :, :, 1]
    new_flow[:, :, :, 0] = flow[:, :, :, 2]
    return new_flow


def deform_contour(flow_field, scan_key, root_path, z_shape, csv_path, moving_tensor, fixed_tensor, warped_tensor):
    def calculate_metrics(moving, fixed, csv_array):
        spacing_mm = [3, 1, 1]
        results_dict = compute_surface_distances(moving[0, 0].numpy().astype(bool), fixed[0, 0].numpy().astype(bool),
                                                 spacing_mm)

        csv_array.append(compute_average_surface_distance(results_dict))
        csv_array.append(compute_robust_hausdorff(results_dict, 100))
        csv_array.append(compute_surface_dice_at_tolerance(results_dict, 1))
        print(csv_array)
        return csv_array

    with open(root_path + "/contour_dictionary.json", 'r') as file:
        contour_dict = json.load(file)

    # Obtain scan information and the corresponding file paths.
    [[(patient_id), (scan_id), (f_phase, m_phase)]] = scan_key  # This ugly I know :/

    path_contour_moving = root_path + contour_dict[patient_id[0]][scan_id[0]][m_phase[0]]
    path_contour_fixed = root_path + contour_dict[patient_id[0]][scan_id[0]][f_phase[0]]

    roi_names = ['Esophagus', 'RLung', 'LLung', 'Tumor', 'LN', 'Vertebra', 'Carina', 'Heart', 'cord']
    # roi_names = ['LLung']

    # iterate over all the contours
    transformer_512 = SpatialTransformNearest_unit().to(device)
    grid_512 = generate_grid_unit([80, 512, 512])
    grid_512 = torch.from_numpy(np.reshape(grid_512, (1,) + grid_512.shape)).float().to(device)

    # Upsample the flowfield  from [1, 3, 80, 256, 256] to [1, 3, 80, 512, 512]
    scale_factor = (1, 2, 2)
    print(flow_field.shape)
    flow_field_upsampled = torch.nn.functional.interpolate(flow_field, scale_factor=scale_factor,
                                                           mode='trilinear', align_corners=True).type(torch.FloatTensor)

    # flow_field_upsampled = flow_field_upsampled.permute(0, 2, 3, 4, 1)
    del flow_field
    torch.cuda.empty_cache()
    combined_fixed_contour = torch.zeros((1, 1, 80, 512, 512))
    combined_moving_contour = torch.zeros((1, 1, 80, 512, 512))
    combined_warped_contour = torch.zeros((1, 1, 80, 512, 512))

    roi_names_used = []

    for roi_index, roi_name in enumerate(roi_names):
        print(roi_name)
        # Find the correct index for the specific roi.
        csv_array = [patient_id[0], scan_id[0], f_phase[0], m_phase[0]]
        try:
            contour_moving = sparse.load_npz(path_contour_moving + "/sparse_contour_{}.npz".format(roi_name)).todense()
            contour_moving = np.flip(contour_moving, axis=0).copy()
            contour_moving = torch.tensor(contour_moving[None, None, z_shape[0]:], dtype=torch.float)

            contour_fixed = sparse.load_npz(path_contour_fixed + "/sparse_contour_{}.npz".format(roi_name)).todense()
            contour_fixed = np.flip(contour_fixed, axis=0).copy()
            contour_fixed = torch.tensor(contour_fixed[None, None, z_shape[0]:], dtype=torch.float)
        except:
            print("The following ROI was not found:", roi_names[roi_index], flush=True)
            continue

        csv_array.append(roi_name)
        csv_array = calculate_metrics(contour_moving, contour_fixed, csv_array)
        print(contour_fixed.shape)
        print(combined_fixed_contour.shape)
        combined_fixed_contour += contour_fixed * (roi_index + 1)
        combined_moving_contour += contour_moving * (roi_index + 1)

        transformer_256 = SpatialTransformer((80, 512, 512),mode = "nearest")
        warped_contour = transformer_256(combined_moving_contour,flow_field_upsampled)
        # warped_contour = transformer_512(contour_moving, flow_field_upsampled, grid_512)
        print(warped_contour.shape)
        combined_warped_contour += warped_contour * (roi_index + 1) * 1.0

        csv_array = calculate_metrics(warped_contour, contour_fixed, csv_array)

        file = open(csv_path, 'a')
        writer = csv.writer(file)
        writer.writerow(csv_array)
        file.close()
        roi_names_used.append(roi_name)

    combined_moving_contour = combined_moving_contour[0, 0].detach().numpy()
    combined_fixed_contour = combined_fixed_contour[0, 0].detach().numpy()
    combined_warped_contour = combined_warped_contour[0, 0].detach().numpy()

    moving_tensor = torch.nn.functional.interpolate(moving_tensor, scale_factor=scale_factor,
                                                    mode='trilinear', align_corners=True).type(torch.FloatTensor)[
        0, 0].detach().numpy()
    warped_tensor = torch.nn.functional.interpolate(warped_tensor, scale_factor=scale_factor,
                                                    mode='trilinear', align_corners=True).type(torch.FloatTensor)[
        0, 0].detach().numpy()
    fixed_tensor = torch.nn.functional.interpolate(fixed_tensor, scale_factor=scale_factor,
                                                   mode='trilinear', align_corners=True).type(torch.FloatTensor)[
        0, 0].detach().numpy()

    title = ["Moving image", "predicted image", "target image"]
    contour_viewer([moving_tensor, warped_tensor, fixed_tensor,
                    combined_moving_contour, combined_warped_contour, combined_fixed_contour], title,
                   roi_names=roi_names_used)


def plot_prediction(moving_tensor, fixed_tensor, prediction, flowfield):
    prediction_array = prediction[0, 0].detach().numpy()
    source_array = moving_tensor[0, 0].detach().numpy()
    target_array = fixed_tensor[0, 0].detach().numpy()

    diff_ps = compare_images(prediction_array, source_array, method='diff')
    diff_pt = compare_images(prediction_array, target_array, method='diff')
    diff_ts = compare_images(target_array, source_array, method='diff')

    print(np.max(diff_ps), np.max(diff_pt), np.max(diff_ts))
    # titles = ["Fixed","Prediction","Moving","diff predict - moving","diff prediction - fixed","diff fixed - moving"]
    # slice_viewer([target_array,prediction_array,source_array,diff_ps,diff_pt,diff_ts], titles, (2,3) )
    titles = ["diff predict - moving", "diff prediction - fixed", "diff fixed - moving", "moving", "prediction",
              "Fixed"]

    slice_viewer([diff_ps, diff_pt, diff_ts, source_array, prediction_array, target_array], titles,
                 shape=(2, 4), flow_field=flowfield)


def MSE_loss(fixed_image, predicted_tensor):
    MSE = torch.nn.MSELoss()
    return float(MSE(fixed_image, predicted_tensor))

def MAE_loss(fixed_image, predicted_tensor):
    MAE = torch.nn.L1Loss()
    return float(MAE(fixed_image, predicted_tensor))


def jac_loss(jac):
    return np.size(jac[jac < 0]) / np.size(jac)


# .\venv\Scripts\activate
# cd C:\Users\pje33\GitHub\master_thesis_project\
# python -m elastix.Evaluate_networks_plot_all_contours.py --number_res_blocks 5 --number_of_res_filters 8 --root_path_model
# Filepaths for the CT data and the trained model.

# setting the datapaths
root_path_data = root_path_CT_data
root_path_contour = root_path_CT_data
with open(root_path_data + "/scan_dictionary.json", 'r') as file:
    ct_path_dict = json.load(file)
# with open(root_path_contour + "contour_dictionary.json", 'r') as file:
#     contour_dict = json.load(file)






# choose here what to show/calculate
calculate_MSE_and_jac = True
calculate_contours = False
show_difference = False
save_flowfield = False

# Make two CSV files for MSE/JAC and contour metrics
if calculate_MSE_and_jac:
    results_file_MSE_JAC = root_path_model + "/results_elastix_MAE_MSE_JAC.csv"
    print(results_file_MSE_JAC, flush=True)
    file = open(results_file_MSE_JAC, "a")
    file.close()

if calculate_contours:
    results_file_contours = root_path_model + "/results_model_elastix_contours.csv"
    print(results_file_contours, flush=True)
    file2 = open(results_file_contours, "a")
    file2.close()


patients = [100,101,102, 103, 104, 105, 106]
patients = [122, 123, 124, 125, 126, 127]
phases = [0,10,20,30,40,50,60,70,80,90]



# Run through all the samples in the evaluation_set.
for patient_id in patients:
    if patient_id > 120:
        root_path_CT_data = "C:/Users/pje33/Downloads/4D_CT_lyon_512/"
        root_path_data = root_path_CT_data
        with open(root_path_CT_data + "/scan_dictionary.json", 'r') as file:
            ct_path_dict = json.load(file)


    output_folder = root_path_model + "/{}/".format(patient_id)
    print(output_folder)



    for scan_id in ct_path_dict[str(patient_id)].keys():
        for f_phase in phases:

            # Load the fixed image
            file_path = root_path_data + ct_path_dict[str(patient_id)][scan_id][str(f_phase)]
            index = file_path[::-1].find("/")
            path_fixed_tensor = file_path[:-index] + "{}_256.nii".format(f_phase)
            fixed_file = sitk.ReadImage(path_fixed_tensor)
            fixed_image = sitk.GetArrayFromImage(fixed_file)
            fixed_tensor = torch.tensor(fixed_image).to(torch.float)
            # fixed_tensor = torch.add(fixed_tensor, 1000)
            # fixed_tensor = torch.div(fixed_tensor, 4000)




            for m_phase in phases:
                path_moving_tensor = file_path[:-index] + "{}_256.nii".format(m_phase)
                fixed_file = sitk.ReadImage(path_moving_tensor)
                moving_image = sitk.GetArrayFromImage(fixed_file)
                moving_tensor = torch.tensor(moving_image).to(torch.float)
                # moving_tensor = torch.add(moving_tensor, 1000)
                # moving_tensor = torch.div(moving_tensor, 4000)

                path_prediction = output_folder + '{}_to_{}/result.nii'.format(m_phase, f_phase)
                predicted_file = sitk.ReadImage(path_prediction)
                predicted_image = sitk.GetArrayFromImage(predicted_file)

                predicted_tensor = torch.tensor(predicted_image).to(torch.float)
                # predicted_tensor = torch.add(predicted_tensor,1000)
                # predicted_tensor = torch.div(predicted_tensor,4000)
                # print(path_fixed_tensor)
                print(path_prediction)



                # if show_difference:
                #     plot_prediction(moving_tensor, fixed_tensor, warped_tensor, F_X_Y)

                if calculate_MSE_and_jac:
                    # Calculate the MSE of the warped image
                    csv_array = [patient_id, scan_id, f_phase, m_phase]
                    csv_array.append(MSE_loss(moving_tensor, fixed_tensor))  # Calculate the baseline MSE.
                    csv_array.append(MSE_loss(predicted_tensor, fixed_tensor))
                    csv_array.append(MAE_loss(moving_tensor, fixed_tensor))  # Calculate the baseline MSE.
                    csv_array.append(MAE_loss(predicted_tensor, fixed_tensor))
                    # csv_array.append()

                    jac_det = sitk.ReadImage(output_folder + '{}_to_{}/spatialJacobian.nii'.format(m_phase, f_phase))
                    jac_det = sitk.GetArrayFromImage(jac_det)

                    csv_array.append(jac_loss(jac_det))

                    file = open(results_file_MSE_JAC, 'a')
                    writer = csv.writer(file)
                    writer.writerow(csv_array)
                    file.close()



                # print("All Good", flush=True)
                torch.cuda.empty_cache()
