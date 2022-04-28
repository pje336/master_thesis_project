from contours.contour import *
from slice_viewer import slice_viewer
import cv2 as cv


path_images ='C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/test/1.000000-P4P107S300I00003 Gated 0.0A-97958'
path_contour ='C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/test/1.000000-P4P107S300I00003 Gated 0.0A-388.1'

index = 2

contour_data = dicom.read_file(path_contour + '/1-1.dcm')
print(get_roi_names(contour_data))



images, contours = get_data(path_images,path_contour, index)
mask_1 = get_mask(path_images, path_contour, index=1)
mask_2 = get_mask(path_images, path_contour, index=2)
mask_3 = get_mask(path_images, path_contour, index=3)

slice_viewer([mask_1,mask_2,mask_3])
