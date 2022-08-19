from argparse import ArgumentParser

import glob
from Network_Functions.dataset_generator import *
import json
from Evaluation.slice_viewer_flow import slice_viewer


from LapIRN_model.Code.Functions import generate_grid_unit, transform_unit_flow_to_flow
from LapIRN_model.Code.miccai2021_model import Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1, \
    Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2, Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3, \
    SpatialTransform_unit, SpatialTransformNearest_unit




def LabIRN_predict(model, fixed_img, moving_img):
    reg_input = 0.4
    # print("Current reg_input: ", str(reg_input))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = SpatialTransform_unit().cuda()
    transform_nearest = SpatialTransformNearest_unit()

    transform.eval()
    grid = generate_grid_unit(fixed_img.shape[2:])
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).float()

    with torch.no_grad():
        reg_code = torch.tensor([reg_input], dtype=fixed_img.dtype, device=fixed_img.device).unsqueeze(dim=0)

        F_X_Y = model(moving_img, fixed_img, reg_code)

        image = transform(moving_img, F_X_Y.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]

        F_X_Y_cpu = F_X_Y.data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
        flow = transform_unit_flow_to_flow(F_X_Y_cpu)

    return image, flow


def load_LapIRN_model(trained_model_folder, model_name, epoch=None, imgshape = (80, 256, 256)):
    range_flow = 0.4
    start_channel = 7
    imgshape_4 = (imgshape[0] / 4, imgshape[1] / 4, imgshape[2] / 4)
    imgshape_2 = (imgshape[0] / 2, imgshape[1] / 2, imgshape[2] / 2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_lvl1 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True,
                                                                         imgshape=imgshape_4,
                                                                         range_flow=range_flow)
    model_lvl2 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2(2, 3, start_channel, is_train=True,
                                                                         imgshape=imgshape_2,
                                                                         range_flow=range_flow, model_lvl1=model_lvl1)

    model = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3(2, 3, start_channel, is_train=False,
                                                                    imgshape=imgshape,
                                                                    range_flow=range_flow, model_lvl2=model_lvl2)


    if epoch is None:
        # Find the model_dict from the latest epoch.
        model_dict_filename = sorted(glob.glob(trained_model_folder + model_name +  "/LDR_*_stagelvl3_*.pth"))[-1]
    print(model_dict_filename)
    model.load_state_dict(torch.load(model_dict_filename,map_location=torch.device(device)))
    model.eval()

    return model


# if __name__ == '__main__':
#
#     imgshape = (80, 256, 256)
#     trained_model_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/saved_models/"
#     model_name = "training_2022_08_16_11_04_43"
#     model = load_LapIRN_model(trained_model_path,model_name)
#     range_flow = 0.4
#     LabIRN_predict(model, fixed_img, moving_img)
