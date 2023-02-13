"""
This file generates a python dictionary for the filepath of the 4DCT dicom dataset.
The dictionary has the  format is [patient_number][scan_id][phase].
It prints the resulting dictionary in json format. This can be saved in another python file.
"""
import json
import os

dictionary_CT_data = {}

patient_list = []
scan_list = []
root_path = "C://Users//pje33//Downloads//4D_CT_lyon//"
# iterate through all sub folders
for path, subdirs, files in os.walk(root_path):
    # split the filepath
    splitted_path = path[len(root_path):].split('\\')

    # check if it is folder with CT data and if the number of files is larger than 1.
    if len(splitted_path) > 2:
        print(splitted_path)

        # find some words to get the phase.
        pos_gated = splitted_path[-1].find("Gated") + len("Gated") + 1
        pos_dot = splitted_path[-1].find(".", pos_gated)

        phase = splitted_path[-1][pos_gated:pos_dot]
        phase = splitted_path[-1]
        patient_number = splitted_path[0][:3]
        scan_id = splitted_path[1]
        print(patient_number)

        # check if the a dict for the patient_number or scan_id already exists.
        # if not, make a dict.
        if patient_number not in patient_list:
            dictionary_CT_data[patient_number] = {}
            patient_list.append(patient_number)
        if scan_id not in scan_list:
            dictionary_CT_data[patient_number][scan_id] = {}
            scan_list.append(scan_id)
        dictionary_CT_data[patient_number][scan_id][phase] = path[len(root_path):].replace('\\', "/")


# print the dict in json format.
scan_file = open(root_path + "scan_dictionary.json", "w")
scan_file = json.dump(dictionary_CT_data, scan_file, indent= 4)
print(dictionary_CT_data)

