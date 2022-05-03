import torch

import voxelmorph
from CT_path_dict.ct_path_dict import ct_path_dict, contour_dict
from contours.contour import *
from dataset_generator import scan_key_generator, generate_dataset
from voxelmorph.torch.layers import SpatialTransformer



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
    exec(text[text.find("\n") + 1:], globals())  # execute the text to set the variable values.
    return learning_rate, epochs, batch_size, loss_weights, patient_id, scan_id_training, scan_id_validation, \
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
    [[(patient_id), (scan_id), (f_phase, m_phase)]] = scan_key
    path_images_moving = root_path + ct_path_dict[patient_id[0]][scan_id[0]][m_phase[0]]
    path_contour_moving = root_path + contour_dict[patient_id[0]][scan_id[0]][m_phase[0]]
    path_images_fixed = root_path + ct_path_dict[patient_id[0]][scan_id[0]][f_phase[0]]
    path_contour_fixed = root_path + contour_dict[patient_id[0]][scan_id[0]][f_phase[0]]

    contour_data_moving = dicom.read_file(path_contour_moving + '/1-1.dcm')

    # iterate over all the contours/
    warped_contour = torch.zeros((len(get_roi_names(contour_data_moving)),80,512,512))
    dice_score = np.zeros(len(get_roi_names(contour_data_moving)))
    for index in range(len(get_roi_names(contour_data_moving))):
        # Get the contour volume and convert to float tensor..
        mask_moving = get_mask(path_images_moving, path_contour_moving, index)[z_shape[0]:z_shape[1]]
        mask_moving_tensor = torch.tensor(np.float64(mask_moving[None, None, ...]), dtype=torch.float)
        mask_fixed = get_mask(path_images_fixed, path_contour_fixed, index)[z_shape[0]:z_shape[1]]
        mask_fixed_tensor = torch.tensor(np.float64(mask_fixed[None, None, ...]), dtype=torch.float)

        # upsample the flowfield to the correct size.
        scale_factor = tuple(np.array(mask_moving_tensor.shape)[-3:] / np.array(flow_field.shape)[-3:])
        flow_field_upsampled = torch.nn.functional.interpolate(flow_field, scale_factor= scale_factor,
                                                               mode='trilinear', align_corners=True)

        # apply transformation to get warped_contour
        transformer = SpatialTransformer(np.shape(mask_moving), mode = 'nearest')
        warped_contour[index] = transformer(mask_moving_tensor, flow_field_upsampled)cd

        dice_score[index] = voxelmorph.py.utils.dice(warped_contour[index].detach().numpy() , mask_fixed_tensor.detach().numpy())[0]

    return warped_contour, dice_score


# Filepaths for the CT data and the trained model.
root_path_data = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-256/"
root_path_contour = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-512/"
trained_model_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/saved_models/training_2022_04_25_12_33_59/"
trained_model_dict_name = "voxelmorph_model_epoch_9.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Obtain training parameters.
learning_rate, epochs, batch_size, loss_weights, patient_id, scan_id_training, scan_id_validation, validation_batches, \
nb_features, data_shape, int_downsize = read_model_parameters_from_file(trained_model_path)

# Make the model and load the weights.
model = voxelmorph.networks.VxmDense(data_shape, nb_features, int_downsize=int_downsize, bidir=True)
model.load_state_dict(torch.load(trained_model_path + trained_model_dict_name, map_location=device))
model.to(device)
model.eval()
print("Model imported")

# Parameters for evaluation dataset
patient_id_evaluation = ["107"]
scan_id_evaluation = ["06-02-1999-p4-89680"]
batch_size = 1
dimensions = [0, 80, 0, 256, 0, 256]
shift = [0, 0, 0, 0]


# Make an evaluation dataset.
evaluation_scan_keys = scan_key_generator(ct_path_dict, patient_id_evaluation, scan_id_evaluation)
evaluation_set = generate_dataset(evaluation_scan_keys, root_path_data, ct_path_dict, dimensions, shift, batch_size)
# Run through all the samples in the evaluation_set.
for moving_tensor, fixed_tensor, scan_key in evaluation_set:
    print(scan_key)

    moving_tensor = moving_tensor.to(device)
    fixed_tensor = fixed_tensor.to(device)
    prediction = model(moving_tensor, fixed_tensor)

    # compute the deformation of the contours.
    warped_contour, dice_score = deform_contour(prediction[-1], scan_key, root_path_contour, ct_path_dict, contour_dict, dimensions[:2])
    print("dice scores for all contours:", dice_score)
    # Calculate jacobian for every voxel in deformation map.
    jac = voxelmorph.py.utils.jacobian_determinant(prediction[-1][0].permute(1, 2, 3, 0).cpu().detach().numpy())
    print("percentage of voxel with negative jacobian:", np.size(jac[jac < 0]) / np.size(jac) * 100)

    del prediction
    del fixed_tensor
    del moving_tensor
