"""
Script to generate sparse 3d matrix with boolean values of contours from DICOM files.
"""

from contours.contour_functions import *
import json
import sparse

root_path_contour = "C:/Users/pje33/Desktop/4d-lung/manifest-1665386976937/4D-Lung/"

with open(root_path_contour + "scan_dictionary.json", 'r') as file:
    ct_path_dict = json.load(file)
with open(root_path_contour + "contour_dictionary.json", 'r') as file:
    contour_dict = json.load(file)


for patient_id in ct_path_dict.keys():
    for scan_id in ct_path_dict[patient_id].keys():
        for m_phase in ct_path_dict[patient_id][scan_id].keys():
            print(scan_id,m_phase)
            path_images_moving = root_path_contour + ct_path_dict[patient_id][scan_id][m_phase]
            path_contour_moving = root_path_contour + contour_dict[patient_id][scan_id][m_phase]
            # obtain contour data.
            contour_data_moving = dicom.read_file(path_contour_moving + '/1-1.dcm')
            roi_names = get_roi_names(contour_data_moving)
            initial_roi = True
            for roi_index in range(len(roi_names)):
                # Find the correct index for the specific roi.
                try:
                    index_m_phase = get_roi_names(contour_data_moving).index(roi_names[roi_index])
                    mask_moving = (get_mask(path_images_moving, path_contour_moving, index_m_phase) > 0) * 1

                    mask_moving_sparse = sparse.COO(mask_moving)
                    sparse.save_npz(path_contour_moving+'/sparse_contour_{}.npz'.format(roi_names[roi_index].split("_")[0]), mask_moving_sparse)
                    #
                except:
                    print("The following ROI was not found:", roi_names[roi_index], flush=True)
                    continue






