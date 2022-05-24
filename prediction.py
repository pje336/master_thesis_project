import torch

import voxelmorph
from CT_path_dict.ct_path_dict import ct_path_dict, contour_dict
from contours.contour import *
from dataset_generator import generate_dataset, scan_key_generator
from voxelmorph.torch.layers import SpatialTransformer
from skimage.util import compare_images
import matplotlib.pyplot as plt
from slice_viewer import slice_viewer


def read_model_parameters_from_file(model_path: str, filename: str = "training_parameters.txt"):
    """
    Read the model parameters from the file training_parameters.txt and convert the values into variables.
    Args:
        model_path: [string] path to the folder with trained model.
        filename: [string]  filename of file with training parameters, by default is "training_parameters.txt"

    Returns: learning_rate, epochs, batch_size, loss_weights, patient_id, scan_id_training, scan_id_validation, \
           validation_batches, nb_features, data_shape

    """
    text = open(model_path + filename).read()  # read the file
    exec(text[text.find("\n") + 1:text.find("losses")], globals())  # execute the text to set the variable values.
    return learning_rate, epochs, batch_size, loss_weights, patient_id_training, scan_id_training, patient_id_validation, scan_id_validation, \
           validation_batches, nb_features, data_shape, int_downsize


def deform_contour(flow_field, scan_key, root_path, ct_path_dict, contour_dict, z_shape):
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
    contour_data_moving = dicom.read_file(path_contour_moving + '/1-1.dcm')
    contour_data_fixed = dicom.read_file(path_contour_fixed + '/1-1.dcm')

    roi_names = ['Esophagus', 'RLung', 'LLung', 'Tumor', 'LN', 'Vertebra', 'Carina']

    # iterate over all the contours
    warped_mask = torch.zeros((len(roi_names), z_shape[-1] - z_shape[0], 512, 512))
    dice_score_warped = np.zeros(len(roi_names))
    dice_score_orignal = np.zeros(len(roi_names))
    for roi_index in range(len(roi_names)):
        # Find the correct index for the specific roi.
        index_f_phase = get_roi_names(contour_data_fixed).index(roi_names[roi_index] + "_c" + f_phase[0][0] + '0')
        index_m_phase = get_roi_names(contour_data_moving).index(roi_names[roi_index] + "_c" + m_phase[0][0] + '0')

        # Get the contour volume and convert to float tensor..
        mask_moving = (get_mask(path_images_moving, path_contour_moving, index_m_phase)[z_shape[0]:z_shape[1]] > 0) * 1
        mask_moving_tensor = torch.tensor(np.float64(mask_moving[None, None, ...]), dtype=torch.float)
        print(mask_moving_tensor.shape)

        mask_fixed = ((get_mask(path_images_fixed, path_contour_fixed, index_f_phase)[z_shape[0]:z_shape[1]] > 0) * 1)

        # upsample the flowfield to the correct size at first iteration.
        if roi_index == 0:
            scale_factor = tuple(np.array(mask_moving_tensor.shape)[-3:] / np.array(flow_field.shape)[-3:])
            flow_field_upsampled = torch.nn.functional.interpolate(flow_field, scale_factor=scale_factor,
                                                                   mode='trilinear', align_corners=True)
            transformer = SpatialTransformer(np.shape(mask_moving), mode='nearest')

        print(flow_field_upsampled.shape)
        # Apply transformation to get warped_contour
        warped_mask[roi_index] = transformer(mask_moving_tensor, flow_field_upsampled)

        # Calculate the dice score between the warped mask  and the fixed mask.
        dice_score_warped[roi_index] = voxelmorph.py.utils.dice(warped_mask[roi_index].detach().numpy(), mask_fixed, 1)[
            0]
        dice_score_orignal[roi_index] = voxelmorph.py.utils.dice(mask_moving, mask_fixed, 1)[0]
    return warped_mask, dice_score_warped, dice_score_orignal


# Filepaths for the CT data and the trained model.
root_path_data = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-256/"
root_path_contour = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-512/"
trained_model_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/saved_models/training_2022_05_24_12_19_03/"
trained_model_dict_name = "voxelmorph_model_epoch_3.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Obtain training parameters.
learning_rate, epochs, batch_size, loss_weights, patient_id_training, scan_id_training, patient_id_validation, scan_id_validation, \
validation_batches, nb_features, data_shape, int_downsize = read_model_parameters_from_file(trained_model_path)

# Make the model and load the weights.
model = voxelmorph.networks.VxmDense(data_shape, nb_features, int_downsize=int_downsize, bidir=True)
model.load_state_dict(torch.load(trained_model_path + trained_model_dict_name, map_location=device))
model.to(device)
model.eval()
print("Model imported")

# Parameters for evaluation dataset
patient_id_evaluation = ["108"]
scan_id_evaluation = None
batch_size = 1
dimensions = [0, 80, 0, 256, 0, 256]
# dimensions = [0, 80, 59, 187, 59, 187]
# dimensions = [0, 80, 59, 187, 59, 187]
# dimensions = [0, 80, 80, 176, 80, 176]
shift = [0,0, 0, 0]

# Make an evaluation dataset.
# evaluation_scan_keys = scan_key_generator(ct_path_dict, patient_id_evaluation, scan_id_evaluation)
evaluation_scan_keys = [[('108'), ('06-21-1999-p4-15001'), [('60'), ('30')]]]
evaluation_set = generate_dataset(evaluation_scan_keys, root_path_data, ct_path_dict, dimensions, shift, batch_size)

calculate_dice = False
calculate_jac = False
show_difference = True
z = 20
# Run through all the samples in the evaluation_set.
for moving_tensor, fixed_tensor, scan_key in evaluation_set:
    print(scan_key)

    moving_tensor = moving_tensor.to(device)
    fixed_tensor = fixed_tensor.to(device)

    prediction = model(moving_tensor, fixed_tensor)



    if show_difference:
        prediction_array = prediction[0][0,0].detach().numpy()
        source_array= moving_tensor[0,0].detach().numpy()
        target_array = fixed_tensor[0,0].detach().numpy()
        slice_viewer([source_array],"source")
        slice_viewer([source_array,target_array],["source","target"])
        diff_ps = compare_images(prediction_array, source_array, method='diff')
        diff_pt = compare_images(prediction_array, target_array, method='diff')
        diff_ts = compare_images(target_array, source_array, method='diff')
        # slice_viewer([source_array,diff_ts, target_array],["source","diff", "target"])

        print(np.max(diff_ps),np.max(diff_pt),np.max(diff_ts))
        titles = ["Fixed","Prediction","Moving","diff predict - moving","diff prediction - fixed","diff fixed - moving"]
        slice_viewer([target_array,prediction_array,source_array,diff_ps,diff_pt,diff_ts], titles , (2,3))
        titles = ["diff predict - moving","diff prediction - fixed","diff fixed - moving"]
        slice_viewer([diff_ps, diff_pt, diff_ts], titles, (1, 3))

    if calculate_dice:
        # compute the deformation of the contours.
        warped_mask, dice_score_warped, dice_score_orignal = deform_contour(prediction[-1], scan_key, root_path_contour,
                                                                            ct_path_dict, contour_dict, dimensions[:2])
        print("dice scores for all warped contours:", dice_score_warped)
        print("dice scores for all orignal contours:", dice_score_orignal)

    if calculate_jac:
        # Calculate jacobian for every voxel in deformation map.
        jac = voxelmorph.py.utils.jacobian_determinant(prediction[-1][0].permute(1, 2, 3, 0).cpu().detach().numpy())
        print("percentage of voxel with negative jacobian:", np.size(jac[jac < 0]) / np.size(jac) * 100)

    # del prediction
    # del fixed_tensor
    # del moving_tensor
