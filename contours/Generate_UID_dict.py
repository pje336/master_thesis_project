import os
import json
import pydicom

path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/test/1.000000-P4P107S300I00003 Gated 0.0A-97958/"

full_path, dirs, files = next(os.walk(path))
files.pop(files.index("1-1.dcm"))

UID_dict = {}
for file in files:
    data = pydicom.dcmread(path + file)
    UID_dict[data.SOPInstanceUID] = file

json_file = open(path + "data.json", "w")
json_file = json.dump(UID_dict, json_file)
