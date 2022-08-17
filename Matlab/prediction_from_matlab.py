import numpy as np
import torch
from Voxelmorph_model.voxelmorph.torch.layers import SpatialTransformer
import Voxelmorph_model.voxelmorph as voxelmorph

def predict(fixed_image, moving_image, dose_dist, trained_model_path, model_name, epoch):
    """
    Args:
        fixed_image: [array][z,x,y] fixed CT image
        moving_image: [array][z,x,y] moving CT image
        dose_dist: [array][z,x,y] dose distribution to be registered.
        trained_model_path: [string] file path to the model folder
        model_name: [string] name of model folder to use.
        epoch: [int] model epoch to use

    Returns:
        registerd_dose_dist

    """
    data_shape = np.shape(fixed_image)
    int_downsize = 1
    nb_features = [
        [16, 32, 32, 32],
        [32, 32, 32, 32, 32, 16, 16]]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the model
    model = voxelmorph.networks.VxmDense(data_shape, nb_features, int_downsize=int_downsize, bidir=True, tanh=True)
    model.load_state_dict(
        torch.load(trained_model_path + model_name + "/voxelmorph_model_epoch_{}.pth".format(int(epoch)),
                   map_location=device))
    model.to(device)
    model.eval()
    transformer = SpatialTransformer(data_shape).to(device)


    fixed_tensor = torch.tensor(np.array(fixed_image), dtype=torch.float)
    moving_tensor = torch.tensor(np.array(moving_image), dtype=torch.float)
    dose_dist_tensor = torch.tensor(np.array(dose_dist), dtype=torch.float)

    print("start deforming")
    deformation_field = model(moving_tensor[None,None,...], fixed_tensor[None,None,...])[-1]
    print("done deforming")

    registerd_dose_dist = transformer(dose_dist_tensor[None,None,...], deformation_field)

    return registerd_dose_dist
