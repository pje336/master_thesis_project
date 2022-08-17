###Hello


Files: 



###*ct_path_dict_generator.py* 
This script generates a python dictionary with the filepath of the 4DCT data.
It scans all the folder in the root_path and checks if it has more than one dicom file.\
Then it adds it to the dict. The dictionary has the format is [patient_number][scan_id][phase].\
It prints the resulting dictionary in json format. This can be saved in a seperate file.

###*ct_path_dict.py* 
This file contains a python dictionary with the filepaths of the 4DCT data.
This filepath needs to be concatinated to the root_path which ends with "4D-Lung/"
This dictionary was generated using "ct_path_dict_generator.py" file.
The dictionary format is: [patient_number][scan_id][phase].


###*resize_dicom.py* 
This script can be used to resize dicom files from the original size to a set size.
This is done in the x and y direction. It also generates the according new folders.