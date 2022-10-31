"""
Get the contour points and save them in a txt file.

TODO: think about scaling and the origin. 
"""

import numpy as np
from contours.contour import *
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
                    points = get_points(path_images_moving, path_contour_moving, index_m_phase)
                    print(points)
                    points = np.divide(points,(2,2,1))
                    print(points)
                    print("saving")
                    # np.savetxt(path_contour_moving+'/../contours/points_contour_{}_{}.txt'.format(roi_names[roi_index].split("_")[0], m_phase), points, delimiter=",", fmt='%f')


                except:
                    print("The following ROI was not found:", roi_names[roi_index], flush=True)
                    continue






