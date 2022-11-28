import SimpleITK as sitk
import sys
import os
import numpy as np
import json



root_path_data = "C:/Users/pje33/Downloads/4D_CT_lyon_512/"
with open(root_path_data + "scan_dictionary.json", 'r') as file:
    ct_path_dict = json.load(file)

desired_resolution = [0.9766*2, 0.9766*2, 3]


y_offset = {"121": 85.9375,
            "122": 54.6875,
            "123": 54.4921875,
            "124": 55.46875,
            "125": 55.078125,
            "126": 55.078125}


# for patient_id in ct_path_dict.keys():
for patient_id in ct_path_dict.keys():
    for scan_id in ct_path_dict[patient_id].keys():
        indexes_of_points_outside = []
        for phase in ct_path_dict[patient_id][scan_id].keys():

            file_path = ct_path_dict[patient_id][scan_id][phase]
            index = file_path[::-1].find("/")
            input_file = root_path_data + file_path
            output_file = root_path_data + file_path[:-index] + "{}_256.nii".format(phase)
            print("Reading Dicom directory:", input_file)
            print("Writing image:", output_file)


            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(input_file)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()[:,:,-120:]
            spacing = image.GetSpacing()
            size = image.GetSize()
            # print(image)
            # print("Image size:", size[0], size[1], size[2])
            # print("Origin", image.GetOrigin())
            # print("Image spacing image:", image.GetSpacing())



            new_size =[int(spacing[0] / desired_resolution[0] * 512) + 1,
                       int(spacing[1] / desired_resolution[1] * 512) + 1,
                       int(size[2] * spacing[2] / desired_resolution[2])]
            # print("new size:", new_size)
            # print(size[2] * spacing[2] / desired_resolution[2])
            # resample the image to the new spacing
            image_256 = sitk.Resample(
                image1=image,
                size= new_size,
                transform=sitk.Transform(),
                interpolator=sitk.sitkLinear,
                outputOrigin=image.GetOrigin(),
                outputSpacing=[0.97*2,0.97*2,3],
                outputDirection=image.GetDirection(),
                defaultPixelValue=0,
                outputPixelType=image.GetPixelID(),
            )



            if new_size[0] < 256: # if it is to small pad it
                # print("padding")

                # Make an image of the correct size and set all values to zero
                image_256_empty = sitk.Resample(
                    image1=image,
                    size=[256,256,new_size[-1]],
                    transform=sitk.Transform(),
                    interpolator=sitk.sitkLinear,
                    outputOrigin=image.GetOrigin(),
                    outputSpacing=[0.97 * 2, 0.97 * 2, 3],
                    outputDirection=image.GetDirection(),
                    defaultPixelValue=0,
                    outputPixelType=image.GetPixelID(),
                )
                image_256_empty[:,:,:] = 0


                # fill the center of the matrix with the orinal image
                index = 256 - new_size[0]
                image_256_empty[index // 2 : -index // 2  , index // 2: -index // 2,:] = image_256

                # Update the origin
                image_256_empty.SetOrigin((image_256.GetOrigin()[0] - index // 2 * desired_resolution[0],
                                     image_256.GetOrigin()[1] - index // 2 * desired_resolution[1],
                                     image_256.GetOrigin()[-1]))
                image_256 = image_256_empty


            if new_size[0] > 256:  # else crop
                print("cropping")
                index = new_size[0] - 256
                image_256 = image_256[index//2:-index//2,index//2:-index//2,:]
                # The origin is automatically adjusted

            #
            # print("Image size:", image_256[:,:,-80:].GetSize())
            # print("new origin", image_256.GetOrigin())
            print(output_file)
            sitk.WriteImage(image_256, output_file)


            def convert_points_physical_to_voxel(points, origin, spacing):
                voxel_point = points - origin
                voxel_point /= spacing
                return voxel_point

        #     # Read points:
        #     try:
        #         points_file = root_path_data + "{}_HM10395/points/{}.pts.txt".format(patient_id,phase.zfill(2))
        #         points = np.loadtxt(points_file)
        #     except:
        #         continue
        #
        #     points[:,1] += y_offset[patient_id] # add offset in the y-direction
        #     voxel_point = convert_points_physical_to_voxel(points, image_256.GetOrigin(), image_256.GetSpacing())
        #     print(len(voxel_point))
        #
        #     # Now we need to remove points which are outsite of the resampled volume
        #
        #     for i, point in enumerate(voxel_point):
        #         if point[0] <= 0 or point[1] <= 0 or point[2] <= 0:
        #             print(point, points[i])
        #             indexes_of_points_outside.append(i)
        #
        #
        #     np.savetxt(root_path_data + "{}_HM10395/points/{}_shifted.pts.txt".format(patient_id, phase.zfill(2)),points, fmt = "%f")
        #     np.savetxt(root_path_data + "{}_HM10395/points/{}_voxel.pts.txt".format(patient_id, phase.zfill(2)),voxel_point, fmt = "%f")
        #
        # if len(indexes_of_points_outside) > 0:
        #     print(set(indexes_of_points_outside))
        #     for phase in ct_path_dict[patient_id][scan_id].keys():
        #         try:
        #             points_file = root_path_data + "{}_HM10395/points/{}_shifted.pts.txt".format(patient_id, phase.zfill(2))
        #             points_voxel_file = root_path_data + "{}_HM10395/points/{}_voxel.pts.txt".format(patient_id, phase.zfill(2))
        #             points = np.loadtxt(points_file)
        #             voxel_point = np.loadtxt(points_voxel_file)
        #
        #         except:
        #             continue
        #         # delete the points outsite of the volume.
        #         print(len(voxel_point))
        #         points = np.delete(points, list(set(indexes_of_points_outside)), axis  = 0 )
        #         voxel_point = np.delete(voxel_point, list(set(indexes_of_points_outside)), axis  = 0 )
        #         print(len(voxel_point))
        #         # print(voxel_point)
        #         np.savetxt(root_path_data + "{}_HM10395/points/{}_shifted.pts.txt".format(patient_id, phase.zfill(2)), points, fmt = "%f")
        #         np.savetxt(root_path_data + "{}_HM10395/points/{}_voxel.pts.txt".format(patient_id, phase.zfill(2)), voxel_point, fmt = "%f")
