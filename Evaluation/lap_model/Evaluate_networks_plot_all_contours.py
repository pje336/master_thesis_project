import torch
import glob
import Voxelmorph_model.voxelmorph as voxelmorph
from LapIRN_model.Code.Functions import generate_grid, transform_unit_flow_to_flow_cuda, \
    generate_grid_unit
from LapIRN_model.Code.miccai2021_model import Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1, \
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


parser = ArgumentParser()

parser.add_argument("--root_path_model", type=str,
                    dest="root_path_model",
                    default='../',
                    help="data path for stored model")


parser.add_argument("--root_path_CT_data", type=str,
                    dest="root_path_CT_data",
                    default='C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT',
                    help="data path for training images")

parser.add_argument("--number_res_blocks", type=int,
                    dest="number_res_blocks", default=9,
                    help="Number of number_res_blocks in level 1")

parser.add_argument("--number_of_res_filters", type=int,
                    dest="number_of_res_filters", default=10,
                    help="Number of number_of_res_filters in level 1")

opt = parser.parse_args()
root_path_model = opt.root_path_model
root_path_CT_data = opt.root_path_CT_data
number_of_res_blocks_lvl_1 = opt.number_res_blocks
number_of_res_filters_lvl_1 = opt.number_of_res_filters



def deform_contour(flow_field, scan_key, root_path, z_shape, csv_path, moving_tensor, fixed_tensor, warped_tensor):
    def calculate_metrics(moving, fixed, csv_array):
        spacing_mm = [3, 1, 1]
        results_dict = compute_surface_distances(moving[0, 0].numpy().astype(bool), fixed[0, 0].numpy().astype(bool),
                                                 spacing_mm)

        csv_array.append(compute_average_surface_distance(results_dict))
        csv_array.append(compute_robust_hausdorff(results_dict, 0))
        csv_array.append(compute_surface_dice_at_tolerance(results_dict, 0))

        return csv_array

    with open(root_path + "contour_dictionary.json", 'r') as file:
        contour_dict = json.load(file)

    # Obtain scan information and the corresponding file paths.
    [[(patient_id), (scan_id), (f_phase, m_phase)]] = scan_key  # This ugly I know :/

    path_contour_moving = root_path + contour_dict[patient_id[0]][scan_id[0]][m_phase[0]]
    path_contour_fixed = root_path + contour_dict[patient_id[0]][scan_id[0]][f_phase[0]]

    roi_names = ['Esophagus', 'RLung', 'LLung', 'Tumor', 'LN', 'Vertebra', 'Carina', 'Heart', 'cord']

    # iterate over all the contours
    transformer_512 = SpatialTransformNearest_unit().to(device)
    grid_512 = generate_grid_unit([80, 512, 512])
    grid_512 = torch.from_numpy(np.reshape(grid_512, (1,) + grid_512.shape)).float().to(device)

    # Upsample the flowfield  from [1, 3, 80, 256, 256] to [1, 3, 80, 512, 512]
    scale_factor = (1, 2, 2)
    flow_field_upsampled = torch.nn.functional.interpolate(flow_field, scale_factor=scale_factor,
                                                           mode='trilinear', align_corners=True).type(torch.FloatTensor)

    flow_field_upsampled = flow_field_upsampled.permute(0, 2, 3, 4, 1)
    del flow_field
    torch.cuda.empty_cache()
    combined_fixed_contour =torch.zeros((1, 1, 80, 512, 512))
    combined_moving_contour =torch.zeros((1,1, 80, 512, 512))
    combined_warped_contour =torch.zeros((1,1, 80, 512, 512))

    for roi_index, roi_name in enumerate(roi_names):
        # Find the correct index for the specific roi.
        csv_array = [patient_id[0], scan_id[0], f_phase[0], m_phase[0]]
        try:
            contour_moving = sparse.load_npz(path_contour_moving + "/sparse_contour_{}.npz".format(roi_name)).todense()
            contour_moving = np.flip(contour_moving,axis = 0).copy()
            contour_moving = torch.tensor(contour_moving[None, None, z_shape[0]:z_shape[1]], dtype=torch.float)

            contour_fixed = sparse.load_npz(path_contour_fixed + "/sparse_contour_{}.npz".format(roi_name)).todense()
            contour_fixed = np.flip(contour_fixed, axis=0).copy()
            contour_fixed = torch.tensor(contour_fixed[None, None, z_shape[0]:z_shape[1]], dtype=torch.float)
        except:
            print("The following ROI was not found:", roi_names[roi_index], flush=True)
            continue

        csv_array.append(roi_name)
        csv_array = calculate_metrics(contour_moving, contour_fixed, csv_array)
        combined_fixed_contour += contour_fixed * (roi_index + 1)
        combined_moving_contour += contour_moving * (roi_index + 1)





        warped_contour = transformer_512(contour_moving, flow_field_upsampled, grid_512)

        combined_warped_contour += warped_contour * (roi_index + 1)

        csv_array = calculate_metrics(warped_contour, contour_fixed, csv_array)

        file = open(csv_path, 'a')
        writer = csv.writer(file)
        writer.writerow(csv_array)
        file.close()




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
                    combined_moving_contour,  combined_warped_contour, combined_fixed_contour], title ,roi_names = roi_names)


def plot_prediction(moving_tensor, fixed_tensor, prediction, flowfield):
    prediction_array = prediction[0, 0].detach().numpy()
    source_array = moving_tensor[0, 0].detach().numpy()
    target_array = fixed_tensor[0, 0].detach().numpy()

    print(np.shape(prediction_array))
    print(np.shape(source_array))
    # print(source_array.shape())
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
    MSE = voxelmorph.torch.losses.MSE().loss
    return float(MSE(fixed_image, predicted_tensor))


def jac_loss(flowfield):
    flowfield_norm = transform_unit_flow_to_flow_cuda(flowfield.permute(0, 2, 3, 4, 1).clone())[0]
    jac = voxelmorph.py.utils.jacobian_determinant(flowfield_norm).cpu().detach().numpy()
    return np.size(jac[jac < 0]) / np.size(jac)


# .\venv\Scripts\activate
# cd C:\Users\pje33\GitHub\master_thesis_project\
# Filepaths for the CT data and the trained model.
# python -m Evaluation.Evaluate_networks_multi_networks.py

# setting the datapaths
root_path_data = root_path_CT_data + "/4D-Lung-256-h5/"
root_path_contour = root_path_CT_data + "/4D-Lung-512/"
with open(root_path_data + "scan_dictionary.json", 'r') as file:
    ct_path_dict = json.load(file)
with open(root_path_contour + "contour_dictionary.json", 'r') as file:
    contour_dict = json.load(file)



# Import the models
print(root_path_model)
model_path = sorted(glob.glob(root_path_model + "/*_entire_model.pth"))[-1]
print(model_path)

# Import models
model = torch.load(model_path)
model.eval()
reg_code = torch.tensor([[0.4]])

print("Models imported", flush=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters for evaluation dataset
patient_id_evaluation = ["107"]
batch_size = 1
dimensions = [0, 80, 0, 256, 0, 256]
shift = [0, 0, 0, 0]

# Make an evaluation dataset.
evaluation_scan_keys = scan_key_generator(ct_path_dict, patient_id_evaluation)
evaluation_set = generate_dataset(evaluation_scan_keys, root_path_data, ct_path_dict, dimensions, shift, batch_size,
                                  shuffle=False)

# choose here what to show/calculate
calculate_MSE_and_jac = False
calculate_contours = True
show_difference = True
save_flowfield = False

# Make two CSV files for MSE/JAC and contour metrics
if calculate_MSE_and_jac:
    results_file_MSE_JAC = root_path_model + "\\results_model_{}_res_blocks_{}_filters_MSE_JAC.csv".format(number_of_res_blocks_lvl_1,number_of_res_filters_lvl_1*8)
    print(results_file_MSE_JAC, flush=True)
    file = open(results_file_MSE_JAC, "a")
    file.close()

if calculate_contours:
    results_file_contours = root_path_model + "\\results_model_{}_res_blocks_{}_filters_contours.csv".format(number_of_res_blocks_lvl_1,number_of_res_filters_lvl_1*8)
    print(results_file_contours, flush=True)
    file2 = open(results_file_contours, "a")
    file2.close()



# Run through all the samples in the evaluation_set.
for fixed_tensor, moving_tensor, scan_key in evaluation_set:
    print(scan_key)
    [[(patient_id), (scan_id), (f_phase, m_phase)]] = scan_key  # This ugly I know :/
    if int(m_phase[0]) != 50:
        continue

    # Move the tensors to the GPU
    moving_tensor = moving_tensor.to(device)
    fixed_tensor = fixed_tensor.to(device)

    # predict the flowfield
    start_time = datetime.now()
    # with torch.no_grad():
    #     prediction = model(moving_tensor, fixed_tensor, reg_code)
    # F_X_Y = prediction[0]
    F_X_Y = torch.rand(1,3,80,256,256)
    jac_loss(F_X_Y)
    warped_tensor = prediction[1]
    flowfield = F_X_Y.permute(0, 2, 3, 4, 1)
    print(F_X_Y.shape)

    print(flowfield.max())

    print("prediction_time:", datetime.now() - start_time, flush=True)

    if show_difference:
        plot_prediction(moving_tensor, fixed_tensor, warped_tensor, flowfield)


    if calculate_MSE_and_jac:
        # Calculate the MSE of the warped image
        csv_array = [patient_id[0], scan_id[0], f_phase[0], m_phase[0]]
        csv_array.append(MSE_loss(moving_tensor, fixed_tensor)) # Calculate the baseline MSE.
        csv_array.append(MSE_loss(warped_tensor, fixed_tensor))
        csv_array.append(jac_loss(F_X_Y))

        file = open(results_file_MSE_JAC, 'a')
        writer = csv.writer(file)
        writer.writerow(csv_array)
        file.close()



    if calculate_contours:
        deform_contour(F_X_Y, scan_key, root_path_contour, [0, 80], results_file_contours,
                       moving_tensor, fixed_tensor, warped_tensor)

    if save_flowfield:
        name = "/predicted_flowfield_{}_{}_{}_{}.pth".format(patient_id[0], scan_id[0], f_phase[0], m_phase[0])
        torch.save(flowfield,root_path_model+name)
        print("flowfield_saved")

    del fixed_tensor, moving_tensor
    del warped_tensor
    del flowfield
    print("All Good", flush=True)
    torch.cuda.empty_cache()
