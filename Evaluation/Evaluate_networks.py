from Voxelmorph_model import voxelmorph
from contours.contour import *
from Voxelmorph_model.voxelmorph.torch.layers import SpatialTransformer
# from skimage.util import compare_images
from Voxelmorph_model.load_voxelmorph_model import load_voxelmorph_model
from LapIRN_model.Code.Test_cLapIRN import *
from datetime import datetime


def deform_contour(flow_field_array, scan_key, root_path, z_shape):
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

    contour_data_moving = h5py.File(path_contour_moving + '/contour_mask.hdf5', "r")
    contour_data_fixed = h5py.File(path_contour_fixed + '/contour_mask.hdf5', "r")
    roi_names = ['Esophagus', 'RLung', 'LLung', 'Tumor', 'LN', 'Vertebra', 'Carina']

    # iterate over all the contours
    warped_mask = torch.zeros((len(flow_field_array) + 1, len(roi_names), z_shape[-1] - z_shape[0], 512, 512))
    dice_score_warped = np.zeros((len(flow_field_array) + 1, len(roi_names)))
    print(np.shape(warped_mask))
    transformer = SpatialTransformer((z_shape[-1] - z_shape[0], 512, 512), mode='nearest')

    initial_roi = True
    for roi_index, roi_name in enumerate(roi_names):
        # Find the correct index for the specific roi.
        try:
            mask_moving = np.float64(contour_data_moving[roi_name])[z_shape[0]:z_shape[1], ...]
            mask_fixed = np.float64(contour_data_fixed[roi_name])[z_shape[0]:z_shape[1], ...]
            mask_moving_tensor = torch.from_numpy(mask_moving[None, None, ...]).type(torch.FloatTensor)

            if initial_roi is True:
                flow_fields_upsampled = []
                for flow_field in flowfield_array:
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
            warped_mask[i + 1, roi_index] = transformer(mask_moving_tensor.cuda(), flow_field.cuda())

            # Calculate the dice score between the warped mask  and the fixed mask.
            dice_score_warped[i + 1, roi_index] = \
            voxelmorph.py.utils.dice(warped_mask[i + 1, roi_index].detach().numpy(), mask_fixed, 1)[
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
# root_path_data = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-256-h5/"
# root_path_contour = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-512/"
# trained_model_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/saved_models/"

with open(root_path_data + "scan_dictionary.json", 'r') as file:
    ct_path_dict = json.load(file)
with open(root_path_contour + "contour_dictionary.json", 'r') as file:
    contour_dict = json.load(file)

model_array_VM = []

# Import voxelmorph models
voxelmorph_models = ["training_2022_07_02_07_48_00", "training_2022_07_02_07_48_17"]
# voxelmorph_models = ["delftblue_NCC"]

for model_name in voxelmorph_models:
    model_array_VM.append(load_voxelmorph_model(trained_model_path, model_name))

model_array_lab = []

# Import voxelmorph models
lab_models = ["training_2022_08_16_11_04_43", "training_2022_08_17_08_15_24"]
for model_name in lab_models:
    model_array_lab.append(load_LapIRN_model(trained_model_path, model_name))

print("Models imported", flush=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters for evaluation dataset
patient_id_evaluation = ["108"]
scan_id_evaluation = None
batch_size = 1
dimensions = [0, 80, 0, 256, 0, 256]
shift = [0, 0, 0, 0]

# Make an evaluation dataset.
evaluation_scan_keys = scan_key_generator(ct_path_dict, patient_id_evaluation, scan_id_evaluation)
evaluation_set = generate_dataset(evaluation_scan_keys, root_path_data, ct_path_dict, dimensions, shift, batch_size,
                                  shuffle=False)

calculate_MSE = True
calculate_dice = False
calculate_jac = True
show_difference = False

counter = 1

z = 20
results_file = "/scratch/thomasvanderme/saved_models/results.txt"
# results_file = "./results.txt"

initial_results_file = True



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
        flowfield = flowfield[0].permute(1, 2, 3, 0).cpu().detach().numpy()
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
        # lab model
        predicted_image, flowfield = LabIRN_predict(model, fixed_tensor, moving_tensor)
        print("prediction_time:", datetime.now() - start_time, flush=True)
        metrics, items = evaulation_metrics(predicted_image, fixed_tensor, flowfield, calculate_MSE, calculate_dice,
                                            calculate_jac, show_difference)

        flowfield_array.append(flowfield)
        # file.write(str(metrics) + "\n")
        for metric in metrics:
            results.append(metric)
        del predicted_image, flowfield
        end_time = datetime.now()
        print("elapsed time:", end_time - start_time, flush=True)
    if calculate_dice:
        start_time = datetime.now()
        dice_scores = deform_contour(flowfield_array, scan_key, root_path_contour, dimensions[:2])
        file.write(str(dice_scores) + "\n")
        print("elapsed time Dice:", datetime.now() - start_time, flush=True)
        del dice_scores
    file.write(str(results) + "\n")
    file.close()
    print("All Good", flush = True)
    del fixed_tensor
    del moving_tensor

    torch.cuda.empty_cache()
    counter += 1
    if counter > 3:
        break
