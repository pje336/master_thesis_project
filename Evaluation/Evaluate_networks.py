import torch

from Voxelmorph_model import voxelmorph
from contours.contour import *
from Voxelmorph_model.voxelmorph.torch.layers import SpatialTransformer
from Voxelmorph_model.load_voxelmorph_model import load_voxelmorph_model
from LapIRN_model.Code.Test_cLapIRN import *
from datetime import datetime
import sparse
from scipy.spatial.distance import directed_hausdorff
import json

def deform_contour(flow_field_array, scan_key, root_path, z_shape):
    def calculate_hausdorff_score(moving,fixed):
        score = []
        for z in range(len(moving)):
            score.append(directed_hausdorff(moving[z],fixed[z]))
        print(score)
        return score

    with open(root_path + "contour_dictionary.json", 'r') as file:
        contour_dict = json.load(file)
    with open(root_path + "scan_dictionary.json", 'r') as file:
        ct_path_dict = json.load(file)

    # Obtain scan information and the corresponding file paths.
    [[(patient_id), (scan_id), (f_phase, m_phase)]] = scan_key  # This ugly I know :/
    path_contour_moving = root_path + contour_dict[patient_id[0]][scan_id[0]][m_phase[0]]
    path_contour_fixed = root_path + contour_dict[patient_id[0]][scan_id[0]][f_phase[0]]


    roi_names = ['Esophagus', 'RLung', 'LLung', 'Tumor', 'LN', 'Vertebra', 'Carina', 'Heart', 'cord']

    # iterate over all the contours
    warped_contour = torch.zeros((len(flow_field_array) + 1, len(roi_names), z_shape[-1] - z_shape[0], 512, 512))
    hausdorff_score_warped = np.zeros((len(flow_field_array) + 1, len(roi_names)))
    transformer = SpatialTransformer((z_shape[-1] - z_shape[0], 512, 512), mode='nearest')

    initial_roi = True
    for roi_index, roi_name in enumerate(roi_names):
        # Find the correct index for the specific roi.
        try:


            contour_moving = sparse.load_npz(path_contour_moving + "/sparse_contour_{}.npz".format(roi_name)).todense()
            contour_moving = np.float64(contour_moving[z_shape[0]:z_shape[1]])
            contour_fixed = sparse.load_npz(path_contour_fixed + "/sparse_contour_{}.npz".format(roi_name)).todense()
            contour_fixed = np.float64(contour_fixed[z_shape[0]:z_shape[1]])
            contour_moving_tensor = torch.from_numpy(contour_moving[None, None, ...]).type(torch.FloatTensor)

            if initial_roi is True:
                flow_fields_upsampled = []
                for flow_field in flow_field_array:
                    # flow_field = torch.from_numpy(flow_field[None, ...]).permute(0, 4, 1, 2, 3)
                    scale_factor = tuple(np.array(contour_moving_tensor.shape)[-3:] / np.array(flow_field.shape)[-3:])
                    flow_fields_upsampled.append(torch.nn.functional.interpolate(flow_field, scale_factor=scale_factor,
                                                                                 mode='trilinear', align_corners=True).type(torch.FloatTensor))
                initial_roi = False
                del flow_field
                torch.cuda.empty_cache()


        except:
            print("The following ROI was not found:", roi_names[roi_index], flush=True)
            continue

        hausdorff_score_warped[0, roi_index] = np.mean(calculate_hausdorff_score(contour_moving, contour_fixed))


        for i, flow_field in enumerate(flow_fields_upsampled):
            # Apply transformation to get warped_contour
            warped_contour[i, roi_index] = transformer(contour_moving_tensor, flow_field)

            # Calculate the dice score between the warped mask  and the fixed mask.
            hausdorff_score_warped[i + 1, roi_index] = \
            np.mean(calculate_hausdorff_score(warped_contour[i, roi_index].detach().numpy()), contour_fixed)
    del flow_fields_upsampled
    torch.cuda.empty_cache()
    return hausdorff_score_warped



def deform_mask(flow_field_array, scan_key, root_path, z_shape):
    with open(root_path + "contour_dictionary.json", 'r') as file:
        contour_dict = json.load(file)
    with open(root_path + "scan_dictionary.json", 'r') as file:
        ct_path_dict = json.load(file)
    """
    Function to perform deformation of contours for a specific scan using the flow field of that scan.
    Args:
        flow_field: predicted flow field with shape [1,3,z,x,y]
        scan_key: [array]: array with key for  scan. e.g: [[(patient_id), (scan_id), (f_phase, m_phase)]]
        root_path: [string] Root path to 4DCT folders.
        ct_path_dict: [dict] dictionary with all file paths to the scan dicom files..
        contour_dict: [dict] dictionary with all file paths to the contour dicom file.
        z_shape: [array] min and max z-value [z_min,z_max]

    Returns: 4-d array with warped_contour [len(contours),z,x,y]

    """
    # Obtain scan information and the corresponding file paths.
    [[(patient_id), (scan_id), (f_phase, m_phase)]] = scan_key  # This ugly I know :/
    path_images_moving = root_path + ct_path_dict[patient_id[0]][scan_id[0]][m_phase[0]]
    path_contour_moving = root_path + contour_dict[patient_id[0]][scan_id[0]][m_phase[0]]
    path_images_fixed = root_path + ct_path_dict[patient_id[0]][scan_id[0]][f_phase[0]]
    path_contour_fixed = root_path + contour_dict[patient_id[0]][scan_id[0]][f_phase[0]]

    # obtain contour data.

    roi_names = ['Esophagus', 'RLung', 'LLung', 'Tumor', 'LN', 'Vertebra', 'Carina', 'Heart', 'cord']

    # iterate over all the contours
    warped_mask = torch.zeros((len(flow_field_array) + 1, len(roi_names), z_shape[-1] - z_shape[0], 512, 512))
    dice_score_warped = np.zeros((len(flow_field_array) + 1, len(roi_names)))
    transformer = SpatialTransformer((z_shape[-1] - z_shape[0], 512, 512), mode='nearest')

    initial_roi = True
    for roi_index, roi_name in enumerate(roi_names):
        # Find the correct index for the specific roi.
        try:


            mask_moving = sparse.load_npz(path_contour_moving + "/sparse_mask_{}.npz".format(roi_name)).todense()
            mask_moving = np.float64(mask_moving[z_shape[0]:z_shape[1]])
            mask_fixed = sparse.load_npz(path_contour_fixed + "/sparse_mask_{}.npz".format(roi_name)).todense()
            mask_fixed = np.float64(mask_fixed[z_shape[0]:z_shape[1]])
            mask_moving_tensor = torch.from_numpy(mask_moving[None, None, ...]).type(torch.FloatTensor)

            if initial_roi is True:
                flow_fields_upsampled = []
                for flow_field in flow_field_array:
                    flow_field = torch.from_numpy(flow_field[None, ...]).permute(0, 4, 1, 2, 3)
                    scale_factor = tuple(np.array(mask_moving_tensor.shape)[-3:] / np.array(flow_field.shape)[-3:])
                    flow_fields_upsampled.append(torch.nn.functional.interpolate(flow_field, scale_factor=scale_factor,
                                                                                 mode='trilinear', align_corners=True).type(torch.FloatTensor))
                initial_roi = False
                del flow_field
                torch.cuda.empty_cache()


        except:
            print("The following ROI was not found:", roi_names[roi_index], flush=True)
            continue

        dice_score_warped[0, roi_index] = voxelmorph.py.utils.dice(mask_moving, mask_fixed, 1)[0]

        for i, flow_field in enumerate(flow_fields_upsampled):
            # Apply transformation to get warped_contour
            warped_mask[i, roi_index] = transformer(mask_moving_tensor, flow_field)

            # Calculate the dice score between the warped mask  and the fixed mask.
            dice_score_warped[i + 1, roi_index] = \
            voxelmorph.py.utils.dice(warped_mask[i, roi_index].detach().numpy(), mask_fixed, 1)[
                0]
    del flow_fields_upsampled
    torch.cuda.empty_cache()
    return dice_score_warped


def plot_prediction(moving_tensor, fixed_tensor, prediction, flowfield):
    return None


#    prediction_array = prediction[0, 0].detach().numpy()
#    source_array = moving_tensor[0, 0].detach().numpy()
#    target_array = fixed_tensor[0, 0].detach().numpy()
#
#    diff_ps = compare_images(prediction_array, source_array, method='diff')
#    diff_pt = compare_images(prediction_array, target_array, method='diff')
#    diff_ts = compare_images(target_array, source_array, method='diff')
#    # slice_viewer([source_array,diff_ts, target_array],["source","diff", "target"])
#
#    print(np.max(diff_ps), np.max(diff_pt), np.max(diff_ts))
#    # titles = ["Fixed","Prediction","Moving","diff predict - moving","diff prediction - fixed","diff fixed - moving"]
#    # slice_viewer([target_array,prediction_array,source_array,diff_ps,diff_pt,diff_ts], titles, (2,3) )
#    titles = ["diff predict - moving", "diff prediction - fixed", "diff fixed - moving", "moving", "prediction",
#              "Fixed"]
#    slice_viewer([diff_ps, diff_pt, diff_ts, source_array, prediction_array, target_array], titles,
#                 shape=(2, 4), flow_field=flowfield)


def evaulation_metrics(predicted_tensor, fixed_image, flowfield, calculate_MSE, calculate_dice, calculate_jac,
                       show_difference):
    metrics = []
    items = []

    if calculate_MSE:
        MSE = voxelmorph.torch.losses.MSE().loss
        metrics.append(float(MSE(fixed_image, predicted_tensor)))
        items.append("MSE")

    if calculate_jac:
        # Calculate jacobian for every voxel in deformation map.
        # print(flowfield.shape)
        jac = voxelmorph.py.utils.jacobian_determinant(flowfield)
        # print("percentage of voxel with negative jacobian:", np.size(jac[jac < 0]) / np.size(jac) * 100)
        metrics.append(np.size(jac[jac < 0]) / np.size(jac))
        items.append("jac")

    if show_difference:
        plot_prediction(moving_tensor, fixed_image, predicted_tensor, flowfield)

    return metrics, items


# .\venv\Scripts\activate
# cd C:\Users\pje33\GitHub\master_thesis_project\
# Filepaths for the CT data and the trained model.
# python -m Evaluation.Evaluate_networks.py

root_path_data = "/scratch/thomasvanderme/4D-Lung-256-h5/"
root_path_contour = "/scratch/thomasvanderme/4D-Lung-512/"
trained_model_path = "/scratch/thomasvanderme/saved_models/"
#
root_path_data = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-256-h5/"
root_path_contour = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-512/"
trained_model_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/saved_models/"

with open(root_path_data + "scan_dictionary.json", 'r') as file:
    ct_path_dict = json.load(file)
with open(root_path_contour + "contour_dictionary.json", 'r') as file:
    contour_dict = json.load(file)

model_array_VM = []

# Import voxelmorph models
# voxelmorph_models = ["training_2022_07_02_07_48_00", "training_2022_07_02_07_48_17"]
voxelmorph_models = ["delftblue_NCC"]
#
for model_name in voxelmorph_models:
    model_array_VM.append(load_voxelmorph_model(trained_model_path, model_name))

model_array_lab = []

# Import voxelmorph models
lab_models = ["training_2022_08_22_08_13_45"]

for model_name in lab_models:
    model_array_lab.append(load_LapIRN_model(trained_model_path, model_name))
print(len(model_array_lab))
print("Models imported", flush=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters for evaluation dataset
patient_id_evaluation = ["107"]
# scan_id_evaluation = ["06-15-1999-p4-07025"]
scan_id_evaluation = None
batch_size = 1
dimensions = [0, 80, 0, 256, 0, 256]
shift = [0, 0, 0, 0]

# Make an evaluation dataset.
evaluation_scan_keys = scan_key_generator(ct_path_dict, patient_id_evaluation, scan_id_evaluation)
evaluation_set = generate_dataset(evaluation_scan_keys, root_path_data, ct_path_dict, dimensions, shift, batch_size,
                                  shuffle=False)

calculate_MSE = False
calculate_dice = True
calculate_jac = False
show_difference = False

counter = 1

z = 20
results_file = "/scratch/thomasvanderme/saved_models/results.txt"
results_file = "./results.txt"

initial_results_file = True
file = open(results_file, "a")
file.write(str(voxelmorph_models) + "\n")
file.write(str(lab_models) + "\n")
file.write(str(patient_id_evaluation) + "\n")
file.close()



# Run through all the samples in the evaluation_set.
for moving_tensor, fixed_tensor, scan_key in evaluation_set:
    flowfield_array = []

    file = open(results_file, "a")
    results = []
    print(scan_key, flush=True)
    file.write(str(scan_key) + "\n")

    moving_tensor = moving_tensor.to(device)
    fixed_tensor = fixed_tensor.to(device)
    if calculate_MSE:
        metrics, items = evaulation_metrics(moving_tensor, fixed_tensor, None, True, False,
                                            False, False)
        for metric in metrics:
            results.append(metric)

    for i, model in enumerate(model_array_VM):
        start_time = datetime.now()
        # VM model
        (predicted_image, _, flowfield) = model(moving_tensor, fixed_tensor)
        print("prediction_time:", datetime.now() - start_time, flush=True)
        # flowfield = flowfield.permute(0, 2, 3, 4,1)
        print("VM flow shape:",flowfield.shape)
        print(torch.max(flowfield),torch.min(flowfield))

        metrics, items = evaulation_metrics(predicted_image, fixed_tensor, flowfield, calculate_MSE, calculate_dice,
                                            calculate_jac, show_difference)

        flowfield_array.append(flowfield)
        for metric in metrics:
            results.append(metric)
        del predicted_image, _, flowfield
        end_time = datetime.now()
        print("elapsed time:", end_time - start_time, flush=True)

    for i, model in enumerate(model_array_lab):
        start_time = datetime.now()
        predicted_image, flowfield = LabIRN_predict(model, fixed_tensor, moving_tensor)
        print("lab flow shape:",flowfield.shape)
        print(torch.max(flowfield),torch.min(flowfield))

        print("prediction_time:", datetime.now() - start_time, flush=True)
        metrics, items = evaulation_metrics(predicted_image, fixed_tensor, flowfield, calculate_MSE, calculate_dice,
                                            calculate_jac, show_difference)

        flowfield_array.append(flowfield)
        for metric in metrics:
            results.append(metric)
        del predicted_image, flowfield
        end_time = datetime.now()
        print("elapsed time:", end_time - start_time, flush=True)
    if calculate_dice:
        start_time = datetime.now()
        dice_scores = []
        hausdorff_scores = []
        hausdorff_scores.append(deform_contour(flowfield_array[:4], scan_key, root_path_contour, dimensions[:2]))
        hausdorff_scores.append(deform_contour(flowfield_array[4:], scan_key, root_path_contour, dimensions[:2]))

        # dice_scores.append(deform_mask(flowfield_array[:4], scan_key, root_path_contour, dimensions[:2]))
        # dice_scores.append(deform_mask(flowfield_array[4:], scan_key, root_path_contour, dimensions[:2]))
        file.write(str(hausdorff_scores) + "\n")
        print("elapsed time Dice:", datetime.now() - start_time, flush=True)
        del hausdorff_scores
        del dice_scores
    file.write(str(results) + "\n")
    file.close()
    print("All Good", flush = True)
    del fixed_tensor
    del moving_tensor

    torch.cuda.empty_cache()
    counter += 1