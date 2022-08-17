import torch

import json
from contours.contour import *
from Network_Functions.dataset_generator import generate_dataset, scan_key_generator
from Voxelmorph_model.voxelmorph import SpatialTransformer
from datetime import datetime



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
        dice_score_warped[roi_index] = Voxelmorph_model.voxelmorph.py.utils.dice(warped_mask[roi_index].detach().numpy(), mask_fixed, 1)[
            0]
        dice_score_orignal[roi_index] = Voxelmorph_model.voxelmorph.py.utils.dice(mask_moving, mask_fixed, 1)[0]
    return warped_mask, dice_score_warped, dice_score_orignal

# .\venv\Scripts\activate
# cd C:\Users\pje33\GitHub\master_thesis_project\
# Filepaths for the CT data and the trained model.
root_path_data = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-256-h5/"
root_path_contour = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-512/"
trained_model_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/saved_models/"


with open(root_path_data + "scan_dictionary.json", 'r') as file:
    ct_path_dict = json.load(file)
with open(root_path_contour + "contour_dictionary.json", 'r') as file:
    contour_dict = json.load(file)

trained_model_folder = ["delftblue_MSE"]



epoch = 13
print("epoch used: ", epoch)

model_array = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


for model_folder in trained_model_folder:
    # Obtain training parameters.
    learning_rate, epochs, batch_size, loss_weights, patient_id_training, scan_id_training, patient_id_validation, scan_id_validation, \
    validation_batches, nb_features, data_shape, int_downsize = read_model_parameters_from_file(trained_model_path + model_folder  + "/")

    # Make the model and load the weights.
    model = Voxelmorph_model.voxelmorph.networks.VxmDense(data_shape, nb_features, int_downsize=int_downsize, bidir=True, tanh = True)

    model.load_state_dict(torch.load(trained_model_path + model_folder + "/voxelmorph_model_epoch_{}.pth".format(epoch), map_location=device))
    model.to(device)
    model.eval()
    model_array.append(model)

print("Models imported")

# Parameters for evaluation dataset
patient_id_evaluation = None
scan_id_evaluation = None
batch_size = 1
dimensions = [0, 80, 0, 256, 0, 256]
shift = [-10,10, -10, 10]

# Make an evaluation dataset.
evaluation_scan_keys = scan_key_generator(ct_path_dict, patient_id_evaluation, scan_id_evaluation)
evaluation_set = generate_dataset(evaluation_scan_keys, root_path_data, ct_path_dict, dimensions, shift, batch_size, shuffle=True)

calculate_dice = False
calculate_jac = False
show_difference = True
calculate_MSE = False

total_losses = np.zeros(len(model_array) + 1)

counter = 1

z = 20

if calculate_MSE:
    MSE = Voxelmorph_model.voxelmorph.torch.losses.MSE().loss

time = []
start = datetime.now()

# Run through all the samples in the evaluation_set.
for moving_tensor, fixed_tensor, scan_key in evaluation_set:
    stop = datetime.now()
    # print(scan_key)
    print(stop - start)
    time.append(stop - start)
    start = datetime.now()

print(time)