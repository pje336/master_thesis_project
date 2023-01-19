"""
Script to convert the model weights from the pth file to an file with the entire model.
This is needed for the evaluation script. Make sure the model class is the correct one.
# Code to call this function:
# python -m Evaluation.lap_model.save_model.py --number_res_blocks 5 --number_of_res_filters 8 --root_path_model
"""

import glob
import os
import sys
from argparse import ArgumentParser
import torch
from LapIRN_model.Code.miccai2021_model_temp import Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1, \
    Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl_general

parser = ArgumentParser()
parser.add_argument("--root_path_model", type=str,
                    dest="root_path_model",
                    default='../',
                    help="data path for training images")

parser.add_argument("--number_res_blocks", type=int,
                    dest="number_res_blocks", default=5,
                    help="Number of number_res_blocks in level 1")

parser.add_argument("--number_of_res_filters", type=int,
                    dest="number_of_res_filters", default=4,
                    help="Number of number_of_res_filters in level 1")

opt = parser.parse_args()
root_path_dict = opt.root_path_model
number_of_res_blocks_lvl_1 = opt.number_res_blocks
number_of_res_filters_lvl_1 = opt.number_of_res_filters

start_channel = 8
# Define the shape of the images
imgshape = (80, 256, 256)
imgshape_8 = (imgshape[0] / 8, imgshape[1] / 8, imgshape[2] / 8)
imgshape_4 = (imgshape[0] / 4, imgshape[1] / 4, imgshape[2] / 4)
imgshape_2 = (imgshape[0] / 2, imgshape[1] / 2, imgshape[2] / 2)

# Set the parameters for the network.
range_flow = 0.4
max_smooth = 10.

# Make network from the classes
model_lvl1 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True,
                                                                     imgshape=imgshape_4, range_flow=range_flow,
                                                                     number_res_blocks=number_of_res_blocks_lvl_1,
                                                                     number_of_res_filters=number_of_res_filters_lvl_1)

model_lvl2 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl_general(2, 3, start_channel, is_train=True,
                                                                            imgshape=imgshape_2,
                                                                            range_flow=range_flow, number_res_blocks=5,
                                                                            number_of_res_filters=4,
                                                                            e0_filter_previous_model=number_of_res_filters_lvl_1 * 8,
                                                                            model_previous=model_lvl1)

model_lvl3 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl_general(2, 3, start_channel, is_train=False,
                                                                            imgshape=imgshape,
                                                                            range_flow=range_flow, number_res_blocks=5,
                                                                            number_of_res_filters=4,
                                                                            e0_filter_previous_model=4 * 8,
                                                                            model_previous=model_lvl2)

# loading the weights into the model and saving it.
print(root_path_dict)
model_dict_filename = sorted(glob.glob(root_path_dict + "\*_model_stagelvl3_*.pth"))[-1] #select the last file
print(model_dict_filename)
model_lvl3.load_state_dict(torch.load(model_dict_filename, map_location=torch.device('cpu')))
torch.save(model_lvl3, root_path_dict + "/{}_res_blocks_{}_filters_entire_model.pth".format(number_of_res_blocks_lvl_1,
                                                                                            number_of_res_filters_lvl_1 * 8))

print("model_saved", flush=True)