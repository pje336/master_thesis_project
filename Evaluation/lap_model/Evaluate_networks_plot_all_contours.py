"""
This is the main script to evaluate the performence of the Lap models
"""

import torch
import matplotlib.pyplot as plt
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
from surface_distance.metrics import compute_average_surface_distance,compute_robust_hausdorff,\
    compute_surface_dice_at_tolerance,compute_surface_distances
from skimage.util import compare_images
from Evaluation.lap_model.slice_viewer_flow import slice_viewer
from Evaluation.lap_model.contour_viewer import contour_viewer
from matplotlib import font_manager
import matplotlib
import neurite as ne
import tensorflow.keras.backend as K
import tensorflow as tf
from Evaluation.lap_model.plot_contour_layers import plot_contour_layers

##############
# This just for the fonts of the plots
x = font_manager.get_font("C:\\Users\\pje33\\AppData\\Local\\Microsoft\\Windows\\Fonts\\utopia-regular.ttf")
font_manager.fontManager.addfont("C:\\Users\\pje33\\AppData\\Local\\Microsoft\\Windows\\Fonts\\utopia-regular.ttf")
# https://stackoverflow.com/questions/27174425/how-to-add-a-string-as-the-artist-in-matplotlib-legend
class AnyObject(object):
    def __init__(self, text, color):
        self.my_text = text
        self.my_color = color
font = {'size': 4}
matplotlib.rc('font', **font)
plt.rcParams["font.family"] = "Utopia"
##############

parser = ArgumentParser()
parser.add_argument("--root_path_model", type=str,
                    dest="root_path_model",
                    default='C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/saved_models/paper_results/change_single_layer/5_res_blocks_32_filters',
                    help="data path for stored model")

parser.add_argument("--root_path_CT_data", type=str,
                    dest="root_path_CT_data",
                    default='C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT',
                    help="data path for training images")

parser.add_argument("--number_res_blocks", type=int,
                    dest="number_res_blocks", default=5,
                    help="Number of number_res_blocks in level 1")

parser.add_argument("--number_of_res_filters", type=int,
                    dest="number_of_res_filters", default=4,
                    help="Number of number_of_res_filters in level 1")

opt = parser.parse_args()
root_path_model = opt.root_path_model
root_path_CT_data = opt.root_path_CT_data
number_of_res_blocks_lvl_1 = opt.number_res_blocks
number_of_res_filters_lvl_1 = opt.number_of_res_filters


def deform_landmarks(flowfield, scan_key, point_path):
    def point_spatial_transformer(x, single=False, sdt_vol_resize=1):
        """
        Transforms surface points with a given deformation.
        Note that the displacement field that moves image A to image B will be "in the space of B".
        That is, `trf(p)` tells you "how to move data from A to get to location `p` in B".
        Therefore, that same displacement field will warp *landmarks* in B to A easily
        (that is, for any landmark `L(p)`, it can easily find the appropriate `trf(L(p))`
        via interpolation.
        TODO: needs documentation
        """

        # surface_points is a N x D or a N x (D+1) Tensor
        # trf is a *volshape x D Tensor
        surface_points, trf = x
        trf = trf * sdt_vol_resize
        surface_pts_D = surface_points.get_shape().as_list()[-1]
        trf_D = trf.get_shape().as_list()[-1]
        assert surface_pts_D in [trf_D, trf_D + 1]

        if surface_pts_D == trf_D + 1:
            li_surface_pts = K.expand_dims(surface_points[..., -1], -1)
            surface_points = surface_points[..., :-1]

        # just need to interpolate.
        # at each location determined by surface point, figure out the trf...
        # note: if surface_points are on the grid, gather_nd should work as well
        fn = lambda x: ne.utils.interpn(x[0], x[1])
        diff = tf.map_fn(fn, [trf, surface_points], fn_output_signature=tf.float32)
        ret = surface_points + diff

        if surface_pts_D == trf_D + 1:
            ret = tf.concat((ret, li_surface_pts), -1)

        return ret
    flowfield_norm = transform_unit_flow_to_flow_cuda(flowfield.clone().permute(0, 2, 3, 4, 1))

    [[(patient_id), (scan_id), (f_phase, m_phase)]] = scan_key  # This ugly I know :/
    csv_array = [patient_id[0], scan_id[0], f_phase[0], m_phase[0]]

    np_tensor = flowfield_norm.numpy()
    tf_flowfield = tf.convert_to_tensor(np_tensor, dtype=tf.float32)

    points_m = np.loadtxt(
        point_path + "/{}_HM10395/points/{}_voxel.pts.txt".format(patient_id[0], m_phase[0].zfill(2)))[None, :,
               [-1, 0, 1]]
    points_f = np.loadtxt(
        point_path + "/{}_HM10395/points/{}_voxel.pts.txt".format(patient_id[0], f_phase[0].zfill(2)))[None, :,
               [-1, 0, 1]]

    points_tensor_m = tf.convert_to_tensor(points_m, dtype=tf.float32)
    warped_points = point_spatial_transformer((points_tensor_m, tf_flowfield)).numpy()

    L2_warped = np.linalg.norm(points_f[0] - warped_points[0], axis=1)
    L2_orginal = np.linalg.norm(points_f[0] - points_m[0], axis=1)

    csv_array.append(L2_orginal.mean())
    csv_array.append(L2_orginal.std())
    csv_array.append(L2_warped.mean())
    csv_array.append(L2_warped.std())
    print(csv_array)

    return csv_array


def deform_contour(flow_field, scan_key, root_path, z_shape, csv_path, moving_tensor, fixed_tensor, warped_tensor):
    """
    Function to deform the contours using the predicted flowfield.
    It calculates metrics for these contours and plots these contours over the images.

    Args:
        flow_field: The flowfield tensor of shape  [1,3,80,256,256]
        scan_key: The scan key
        root_path: Rootpath to the contours
        z_shape: The shape of the z axes to crop the contour to.
        csv_path: path to the contour csv file to store the metrics in.
        moving_tensor: The moving tensor of shape [1,1,z,x,y] for background
        fixed_tensor: The fixed tensor of shape [1,1,z,x,y] for background
        warped_tensor: The predicted tensor of shape [1,1,z,x,y] for background

    Returns:

    """

    def calculate_metrics(contour_warped, contour_fixed, csv_array):
        """
        Function to calculate the metrics of the contours between the moving and the fixed image
        Metrics: Average surface distance, hausdorff, surface dice.
        Using surface_distance.metrics package.

        Args:
            contour_warped: The moving contour tensor of shape [1,1,z,x,y]
            contour_fixed: The fixed contour tensor of shape [1,1,z,x,y]
            csv_array: csv array to append the results to.

        Returns:

        """
        # Set the image spacing in mm
        spacing_mm = [3, 0.97, 0.97]
        results_dict = compute_surface_distances(contour_warped[0, 0].numpy().astype(bool), contour_fixed[0, 0].numpy().astype(bool),
                                                 spacing_mm)
        csv_array.append(compute_average_surface_distance(results_dict))
        csv_array.append(compute_robust_hausdorff(results_dict, 100))
        csv_array.append(compute_surface_dice_at_tolerance(results_dict, 1))
        return csv_array

    with open(root_path + "contour_dictionary.json", 'r') as file:
        contour_dict = json.load(file)

    # Obtain scan information and the corresponding file paths.
    [[(patient_id), (scan_id), (f_phase, m_phase)]] = scan_key  # This ugly I know :/

    path_contour_moving = root_path + contour_dict[patient_id[0]][scan_id[0]][m_phase[0]]
    path_contour_fixed = root_path + contour_dict[patient_id[0]][scan_id[0]][f_phase[0]]

    # Give the contour names which to be evaluated.
    roi_names = ['Esophagus', 'RLung', 'LLung', 'Tumor', 'LN', 'Vertebra', 'Carina', 'Heart', 'cord']
    roi_names = ['Esophagus', 'RLung', 'LLung', 'Tumor', 'Heart']

    # Upsample the flowfield  from [1, 3, 80, 256, 256] to [1, 3, 80, 512, 512]
    scale_factor = (1, 2, 2)
    flow_field_upsampled = torch.nn.functional.interpolate(flow_field, scale_factor=scale_factor,
                                                           mode='trilinear', align_corners=True).type(torch.FloatTensor)

    flow_field_upsampled = flow_field_upsampled.permute(0, 2, 3, 4, 1)
    del flow_field  # delete 256x256 flowfield
    torch.cuda.empty_cache()

    # Setup a spatial transformer with nearest neighbour mode.
    transformer_512 = SpatialTransformNearest_unit().to(device)
    grid_512 = generate_grid_unit([80, 512, 512])
    grid_512 = torch.from_numpy(np.reshape(grid_512, (1,) + grid_512.shape)).float().to(device)

    # make empty tensors to store the contours in.
    combined_fixed_contour = torch.zeros((1, 1, 80, 512, 512))
    combined_moving_contour = torch.zeros((1, 1, 80, 512, 512))
    combined_warped_contour = torch.zeros((1, 1, 80, 512, 512))

    roi_names_used = []

    # Iterate over all the roi organs
    for roi_index, roi_name in enumerate(roi_names):
        print(roi_name)
        csv_array = [patient_id[0], scan_id[0], f_phase[0], m_phase[0]]
        # Try to obtain the contours, if it fails continue to the next roi.
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
        csv_array = calculate_metrics(contour_moving, contour_fixed, csv_array)  # calculate baseline values

        # deform the contour.
        warped_contour = transformer_512(contour_moving, flow_field_upsampled, grid_512)
        csv_array = calculate_metrics(warped_contour, contour_fixed, csv_array)

        # add all the contours to the array for plotting
        combined_fixed_contour += contour_fixed * (roi_index + 1)
        combined_moving_contour += contour_moving * (roi_index + 1)
        combined_warped_contour += warped_contour * (roi_index + 1) * 1.0

        file = open(csv_path, 'a')
        writer = csv.writer(file)
        writer.writerow(csv_array)
        file.close()
        roi_names_used.append(roi_name)

    # convert the tensors to numpy arrays for plotting
    combined_moving_contour = combined_moving_contour[0, 0].detach().numpy()
    combined_fixed_contour = combined_fixed_contour[0, 0].detach().numpy()
    combined_warped_contour = combined_warped_contour[0, 0].detach().numpy()

    # upsample the CT tensors from 256 to 512 and make them numpy arrays.
    moving_tensor = torch.nn.functional.interpolate(moving_tensor, scale_factor=scale_factor,
                                                    mode='trilinear', align_corners=True).type(torch.FloatTensor)[
        0, 0].detach().numpy()
    warped_tensor = torch.nn.functional.interpolate(warped_tensor, scale_factor=scale_factor,
                                                    mode='trilinear', align_corners=True).type(torch.FloatTensor)[
        0, 0].detach().numpy()
    fixed_tensor = torch.nn.functional.interpolate(fixed_tensor, scale_factor=scale_factor,
                                                   mode='trilinear', align_corners=True).type(torch.FloatTensor)[
        0, 0].detach().numpy()

    # Plot and store the slices using the plot_contour_layers function.
    plot_contour_layers(moving_tensor, warped_tensor, fixed_tensor,
                        combined_moving_contour, combined_warped_contour, combined_fixed_contour,
                        transform_unit_flow_to_flow_cuda(flow_field_upsampled), roi_names)

    # View the contours in a window.
    title = ["Moving image", "predicted image", "target image"]
    contour_viewer([moving_tensor, warped_tensor, fixed_tensor,
                    combined_moving_contour, combined_warped_contour, combined_fixed_contour], title,
                   roi_names=roi_names_used)


def plot_prediction(moving_tensor, fixed_tensor, predicted_tensor, flowfield):
    """
    Function to plot the fixed,moving and predicted image and the flowfield using the slice_viewer function.
    Args:
        moving_tensor: The moving tensor of shape [1,1,z,x,y]
        fixed_tensor: The fixed tensor of shape [1,1,z,x,y]
        predicted_tensor: The predicted tensor of shape [1,1,z,x,y]
        flowfield: The flowfield tensor of shape [1,3,z,x,y]
    """
    flowfield_norm = transform_unit_flow_to_flow_cuda(flowfield.clone().permute(0, 2, 3, 4, 1))

    prediction_array = predicted_tensor[0, 0].detach().numpy()
    moving_array = moving_tensor[0, 0].detach().numpy()
    fixed_array = fixed_tensor[0, 0].detach().numpy()

    # Calulate the difference between the arrays
    diff_pm = compare_images(prediction_array, moving_array, method='diff')
    diff_pf = compare_images(prediction_array, fixed_array, method='diff')
    diff_fm = compare_images(fixed_array, moving_array, method='diff')
    # Print the max difference
    print(np.max(diff_pm), np.max(diff_pf), np.max(diff_fm))

    # Plot the images using slice_viewer
    titles = ["diff predict - moving", "diff prediction - fixed", "diff fixed - moving", "moving", "prediction",
              "Fixed"]
    slice_viewer([diff_pm, diff_pf, diff_fm, moving_array, prediction_array, fixed_array], titles,
                 shape=(2, 4), flow_field=flowfield_norm)


def MSE_loss(fixed_tensor, predicted_tensor):
    """
    Calculates the mean squared error in HU units
    Args:
        fixed_tensor: The fixed tensor of shape [1,1,z,x,y] in HU units
        predicted_tensor: The predicted tensor of shape [1,1,z,x,y] in HU units

    Returns:
        Mean squared error in HU units
    """
    MSE = torch.nn.MSELoss()
    return float(MSE(fixed_tensor, predicted_tensor))


def MAE_loss(fixed_tensor, predicted_tensor):
    """
    Calculates the mean absolute error in HU units
    Args:
        fixed_tensor: The fixed tensor of shape [1,1,z,x,y] in HU units
        predicted_tensor: The predicted tensor of shape [1,1,z,x,y] in HU units

    Returns:
        Mean absolute error in HU units
    """
    MAE = torch.nn.L1Loss()
    return float(MAE(fixed_tensor, predicted_tensor))


def jac_loss(flowfield):
    """
    Calculate the ratio of voxel in the flow field with the negative jacobian determinant.
    Args:
        flowfield: flow_field: The flowfield tensor of shape  [1,3,z,x,y]
    Returns:
        The ratio of voxel in the flow field with the negative jacobian determinant.
    """

    flowfield_norm = transform_unit_flow_to_flow_cuda(flowfield.permute(0, 2, 3, 4, 1).clone())[0]
    jac = voxelmorph.py.utils.jacobian_determinant(flowfield_norm).cpu().detach().numpy()
    return np.size(jac[jac < 0]) / np.size(jac)


##_____________________MAIN_PART______________________________

# .\venv\Scripts\activate
# cd C:\Users\pje33\GitHub\master_thesis_project\
# python -m Evaluation.lap_model.Evaluate_networks_plot_all_contours.py --number_res_blocks 5 --number_of_res_filters 8 --root_path_model "C:\Users\pje33\Google Drive\Sync\TU_Delft\MEP\saved_models\paper_results"
# Filepaths for the CT data and the trained model.

# setting the data paths
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=torch.device(device))
model.eval()
reg_code = torch.tensor([[0.4]])  # this is a wierd regulator value that need to be set
print("Models imported", flush=True)

# Parameters for evaluation dataset
patient_id_evaluation = ["121"]
batch_size = 1
dimensions = [-80, 0, 0, 256, 0, 256]
shift = [0, 0, 0, 0]

# Make an evaluation dataset.
evaluation_scan_keys = scan_key_generator(ct_path_dict, patient_id_evaluation)
evaluation_set = generate_dataset(evaluation_scan_keys, root_path_data, ct_path_dict, dimensions, shift, batch_size,
                                  shuffle=False, shift_z=False)

# choose here what to show/calculate
calculate_MSE_and_jac = False
calculate_contours = False
show_difference = False
save_flowfield = False
plot_layers = False
calculate_deform_landmarks = True

# Make two CSV files for MSE/JAC and contour metrics
if calculate_MSE_and_jac:
    results_file_MSE_JAC = root_path_model + "\\results_model_{}_res_blocks_{}_filters_MSE_MAE_JAC.csv".format(
        number_of_res_blocks_lvl_1, number_of_res_filters_lvl_1 * 8)
    print(results_file_MSE_JAC, flush=True)
    file = open(results_file_MSE_JAC, "a")
    file.close()

if calculate_contours:
    results_file_contours = root_path_model + "\\results_model_{}_res_blocks_{}_filters_contours.csv".format(
        number_of_res_blocks_lvl_1, number_of_res_filters_lvl_1 * 8)
    print(results_file_contours, flush=True)
    file2 = open(results_file_contours, "a")
    file2.close()

if calculate_deform_landmarks:
    results_file_landmark = root_path_model + "\\landmark_metrics_model_{}_res_blocks_{}_filters_contours.csv".format(
        number_of_res_blocks_lvl_1, number_of_res_filters_lvl_1 * 8)
    print(results_file_landmark, flush=True)
    file3 = open(results_file_landmark, "a")
    file3.close()

# Setup a spatial transformer.
transformer_256 = SpatialTransform_unit().to(device)
grid_256 = generate_grid_unit([80, 256, 256])
grid_256 = torch.from_numpy(np.reshape(grid_256, (1,) + grid_256.shape)).float().to(device)

# Run through all the samples in the evaluation_set.
for fixed_tensor, moving_tensor, scan_key, z_shape in evaluation_set:
    z_shape = [-80, -1]
    print(scan_key)
    [[(patient_id), (scan_id), (f_phase, m_phase)]] = scan_key  # Obtain sample data

    # Move the tensors to the device (cpu or gpu).
    moving_tensor = moving_tensor.to(device)
    fixed_tensor = fixed_tensor.to(device)

    # predict the flowfield
    start_time = datetime.now()
    with torch.no_grad():
        Flowfield = model(moving_tensor, fixed_tensor, reg_code)

    warped_tensor = transformer_256(moving_tensor, Flowfield.permute(0, 2, 3, 4, 1), grid_256)  # Warp the image

    if show_difference:
        plot_prediction(moving_tensor, fixed_tensor, warped_tensor, Flowfield)

    if calculate_MSE_and_jac:
        # Calculate the MSE of the warped image
        csv_array = [patient_id[0], scan_id[0], f_phase[0], m_phase[0]]

        warped_tensor = torch.mul(warped_tensor, 4000)
        warped_tensor = torch.add(warped_tensor, -1000)

        fixed_tensor = torch.mul(fixed_tensor, 4000)
        fixed_tensor = torch.add(fixed_tensor, -1000)

        moving_tensor = torch.mul(moving_tensor, 4000)
        moving_tensor = torch.add(moving_tensor, -1000)
        print(moving_tensor.max())
        print(moving_tensor.min())

        # Calculate the MSE of the warped image
        csv_array.append(MSE_loss(fixed_tensor, moving_tensor))
        csv_array.append(MSE_loss(fixed_tensor, warped_tensor))
        csv_array.append(MAE_loss(fixed_tensor, moving_tensor))
        csv_array.append(MAE_loss(fixed_tensor, warped_tensor))
        csv_array.append(jac_loss(Flowfield))

        # Write results to csv
        file = open(results_file_MSE_JAC, 'a')
        writer = csv.writer(file)
        writer.writerow(csv_array)
        file.close()

    if calculate_contours:
        if patient_id[0] != "101":
            print("Hellop")
            continue
        # Deform the contours and calculate metrics.
        deform_contour(Flowfield, scan_key, root_path_contour, [-80, 0], results_file_contours,
                       moving_tensor, fixed_tensor, warped_tensor)

    if save_flowfield:
        flowfield_save = transform_unit_flow_to_flow_cuda(Flowfield.permute(0, 2, 3, 4, 1).clone())
        # Save the predicted flow field. (large files)
        name = "/predicted_flowfield_{}_{}_{}_{}.pth".format(patient_id[0], scan_id[0], f_phase[0], m_phase[0])
        torch.save(flowfield_save, root_path_model + name)
        print("flowfield_saved")

    if calculate_deform_landmarks:
        csv_array_landmarks = deform_landmarks(Flowfield, scan_key, root_path_data)
        file = open(results_file_landmark, 'a')
        writer = csv.writer(file)
        writer.writerow(csv_array_landmarks)
        file.close()

    del fixed_tensor, moving_tensor
    del warped_tensor
    del Flowfield
    print("All Good", flush=True)
    torch.cuda.empty_cache()
