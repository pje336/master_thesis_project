import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import compare_images

import json
from Evaluation.lap_model.slice_viewer_flow import slice_viewer




root_path_data = "C:/Users/pje33/Desktop/4d-lung/manifest-1665386976937/4D-Lung/"
with open(root_path_data + "scan_dictionary.json", 'r') as file:
    ct_path_dict = json.load(file)

patient_id = "103"
scan_id = "06-17-1998-NA-p4-43192"
m_phase = "0"
f_phase = "0"

output_folder = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/elastix/outputs/{}/".format(patient_id)
prediction_path = output_folder + '{}_to_{}/result.nii'.format(m_phase,f_phase)
predicted_file = sitk.ReadImage(prediction_path)
predicted_image = sitk.GetArrayFromImage(predicted_file)
print(np.amax(predicted_image))


file_path = root_path_data + ct_path_dict[patient_id][scan_id][m_phase]

index = file_path[::-1].find("/")
file_path_f = file_path[:-index] + "/{}_256.nii".format(f_phase)
file_path_m = file_path[:-index] + "/{}_256.nii".format(m_phase)

prediction_path = "C:\\Users\\pje33\\Desktop\\4d-lung\\manifest-1665386976937\\4D-Lung\\102_HM10395\\03-25-1998-NA-p4-57341\\1.000000-P4P102S300I00003 Gated 0.0A-720.1"
# predicted_file = sitk.ReadImage(prediction_path + "\\contour_LLung.nii")
# predicted_image = sitk.GetArrayFromImage(predicted_file)
# plt.imshow(predicted_image[40])
# plt.show()

m_files = sitk.ReadImage(file_path_m)
m_image = sitk.GetArrayFromImage(m_files)

f_files = sitk.ReadImage(file_path_f)
f_image = sitk.GetArrayFromImage(f_files)





titles = ["diff predict - moving", "diff prediction - fixed", "diff fixed - moving", "moving", "prediction",
          "Fixed"]

diff_ps = compare_images(predicted_image, m_image, method='diff')
diff_pt = compare_images(predicted_image, f_image, method='diff')
diff_ts = compare_images(f_image, m_image, method='diff')

slice_viewer([diff_ps, diff_pt, diff_ts, m_image,predicted_image , f_image], titles,
             shape=(2, 4), flow_field=np.zeros((1,3,80,256,256)))
titles = ["moving", "prediction", "Fixed"]
# slice_viewer([m_image[::-1], predicted_image[::-1], f_image[::-1]], titles, flow_field =np.zeros((1,3,80,256,256)))