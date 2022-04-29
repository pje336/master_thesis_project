"""
This file generates a python dictionary for the filepath of the 4DCT data.
The dictionary has the  format is [patient_number][scan_id][phase].
It prints the resulting dictionary in json format. This can be saved in an other file.
"""
import json
import os

dictionary_CT_data = {}
dictionary_contours = {}

patient_list = []
scan_list = []
root_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-512/"
# iterate through all sub folders
for path, subdirs, files in os.walk(root_path):
    # split the filepath
    splitted_path = path[len(root_path):].split('\\')

    # check if it is folder with CT data and if the number of files is larger than 1.
    if len(splitted_path) >= 3:
        # find some words to get the phase.
        pos_gated = splitted_path[-1].find("Gated") + len("Gated") + 1
        pos_dot = splitted_path[-1].find(".", pos_gated)

        phase = splitted_path[-1][pos_gated:pos_dot]
        patient_number = splitted_path[0][:3]
        scan_id = splitted_path[1]

        # check if the a dict for the patient_number or scan_id already exists.
        # if not, make a dict.
        if patient_number not in patient_list:
            dictionary_CT_data[patient_number] = {}
            dictionary_contours[patient_number] = {}
            patient_list.append(patient_number)
        if scan_id not in scan_list:
            dictionary_CT_data[patient_number][scan_id] = {}
            dictionary_contours[patient_number][scan_id] = {}
            scan_list.append(scan_id)

        # Insert filepath into the dict.
        # If it has more than one file it are the CT slices
        if len(next(os.walk(path))[-1]) > 1:
            dictionary_CT_data[patient_number][scan_id][phase] = path[len(root_path):].replace('\\', "/")
        else:  # Else it are the contours.
            dictionary_contours[patient_number][scan_id][phase] = path[len(root_path):].replace('\\', "/")

# print the dict in json format.
scan_file = open(root_path + "scan_dictionary.json", "w")
scan_file = json.dump(dictionary_CT_data, scan_file, indent= 4)

contour_file  = open(root_path + "contour_dictionary.json", "w")
contour_file = json.dump(dictionary_contours, contour_file,  indent= 4)

