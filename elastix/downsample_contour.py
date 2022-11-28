import numpy as np

from contours.contour import *
import json
import sparse
import SimpleITK as sitk
from scipy.ndimage import zoom

root_path_contour = "C:/Users/pje33/Desktop/4d-lung/manifest-1665386976937/4D-Lung/"

with open(root_path_contour + "scan_dictionary.json", 'r') as file:
    ct_path_dict = json.load(file)
with open(root_path_contour + "contour_dictionary.json", 'r') as file:
    contour_dict = json.load(file)
root_path_data = "C:/Users/pje33/Desktop/4d-lung/manifest-1665386976937/4D-Lung/"


for patient_id in ["102"]:
    for scan_id in ct_path_dict[patient_id].keys():


        # for m_phase in ct_path_dict[patient_id][scan_id].keys():
        for m_phase in ["50"]:

            file_path = root_path_data + ct_path_dict[patient_id][scan_id][m_phase]

            index = file_path[::-1].find("/")

            path_images_moving = root_path_contour + ct_path_dict[patient_id][scan_id][m_phase]
            path_contour_moving = root_path_contour + contour_dict[patient_id][scan_id][m_phase]
            print(path_contour_moving)
            # obtain contour data.
            contour_data_moving = dicom.read_file(path_contour_moving + '/1-1.dcm')
            roi_names = get_roi_names(contour_data_moving)
            initial_roi = True
            for roi_index in range(len(roi_names)):
                print(roi_names[roi_index])

                # Find the correct index for the specific roi.
                # try:
                contour_moving = sparse.load_npz(
                    path_contour_moving + "/sparse_contour_{}.npz".format(roi_names[roi_index].split("_")[0])).todense()
                contour_moving_zoomed = zoom(contour_moving, (1, 1 / 2, 1 / 2), mode = "nearest")
                contour_moving_zoomed = np.flip(contour_moving_zoomed,axis = 0)[-80:]
                # contour_moving_zoomed = np.flip(contour_moving_zoomed,axis = 0)
                print(np.shape(contour_moving_zoomed))


                # make and save sparse Numpy tensor.
                mask_moving_sparse = sparse.COO(contour_moving_zoomed)
                sparse.save_npz(
                    path_contour_moving + '/sparse_contour_{}_256.npz'.format(roi_names[roi_index].split("_")[0]),
                    mask_moving_sparse)

                # Make and save nii file.
                output_file = file_path[:-index] + "/contour_{}.nii".format(roi_names[roi_index])
                img = sitk.GetImageFromArray(contour_moving_zoomed)
                sitk.WriteImage(img, output_file)

                # except:
                #     print("The following ROI was not found:", roi_names[roi_index], flush=True)
                #     continue



