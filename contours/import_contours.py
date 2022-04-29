from CT_path_dict.ct_path_dict import *

from contours.contour import *
from slice_viewer import slice_viewer

root_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/4D_lung_CT/4D-Lung-512/"
patient_id = "107"
scan_id = "06-02-1999-p4-89680"
phase = "50"


path_images = root_path +  ct_path_dict[patient_id][scan_id][phase]
path_contour = root_path + contour_dict[patient_id][scan_id][phase]

index = 2

contour_data = dicom.read_file(path_contour + '/1-1.dcm')
print(get_roi_names(contour_data))



images, contours = get_data(path_images,path_contour, index)
mask_1 = get_mask(path_images, path_contour, index=1)
mask_2 = get_mask(path_images, path_contour, index=2)
mask_3 = get_mask(path_images, path_contour, index=3)

slice_viewer([mask_1,mask_2,mask_3])
