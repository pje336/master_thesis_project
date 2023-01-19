"""
Fuction to save slices of the CT scans with contour overlay and also the DVF.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch
import torch


def plot_contour_layers(moving_array, prediction_array, fixed_array,
                        combined_moving_contour, combined_warped_contour, combined_fixed_contour, flowfield, roi_names):
    """

    Args:
        moving_array: The moving np array of shape [1,1,80,512,512] for background
        prediction_array: The fixed np array of shape [1,1,80,512,512] for background
        fixed_array: The predicted np array of shape [1,1,80,512,512] for background
        combined_moving_contour: np array with the contour from the moving images of shape [1,1,80,512,512]
        combined_warped_contour: np array with the contour from the warped images of shape [1,1,80,512,512]
        combined_fixed_contour: np array with the contour from the fixed images of shape [1,1,80,512,512]
        flowfield: Upsampled flowfield
        roi_names: array with the usde rois.

    Returns:

    """

    # Voxel without value of 0 are not a contour and are set tot np.nan
    combined_moving_contour[combined_moving_contour == 0] = np.nan
    combined_warped_contour[combined_warped_contour == 0] = np.nan
    combined_fixed_contour[combined_fixed_contour == 0] = np.nan

    flowfield = torch.flip(flowfield, (2,))  # apprently the flowfield is upside down (?)

    names = ["Moving", "Prediction", "Fixed", "DVF x-y"]
    grid_size = 4
    layers = [0, 10, 20, 30, 40, 50, 60, 70]

    # Setup a grid for the DVF plot.
    x = np.linspace(0, 511, 512)
    y = np.linspace(0, 511, 512)
    xv, yv = np.meshgrid(x, y)

    # Set figure size.
    cm = 1 / 2.54
    fig, axs = plt.subplots(len(layers), 4, figsize=(8 * cm, 16 * cm))

    # Set the colors of the contours
    colors_array = ['purple', 'blue', 'red', 'orange', 'green']
    cmap = colors.ListedColormap(colors_array)
    bounds = [1, 2, 3, 4, 5, 6]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    legend_elements = []
    # Set the color to the ROI name
    for index, roi_name in enumerate(roi_names):
        legend_elements.append(Patch(facecolor=colors_array[index],
                                     label=roi_name))
    # Plot the layers. First plot the image then the contours
    for ax_id, layer in enumerate(layers):
        axs[ax_id, 0].imshow(moving_array[layer], cmap='Greys', aspect='auto')
        axs[ax_id, 0].imshow(combined_moving_contour[layer], cmap=cmap, norm=norm, interpolation='none', aspect='auto')

        axs[ax_id, 1].imshow(prediction_array[layer], cmap='Greys', aspect='auto')
        axs[ax_id, 1].imshow(combined_warped_contour[layer], cmap=cmap, norm=norm, interpolation='none', aspect='auto')

        axs[ax_id, 2].imshow(fixed_array[layer], cmap='Greys', aspect='auto')
        axs[ax_id, 2].imshow(combined_fixed_contour[layer], cmap=cmap, norm=norm, interpolation='none', aspect='auto')

        # Plot the xy DVF.
        axs[ax_id, 3].quiver(xv[::grid_size, ::grid_size], yv[::grid_size, ::grid_size],
                             flowfield[0, layer, ::grid_size, ::grid_size, 1],
                             flowfield[0, layer, ::grid_size, ::grid_size, 2], angles='xy',
                             scale_units='xy', scale=20)

        # Remove ax labels and add title for the top plot.
        for label_id, ax in enumerate(axs[ax_id]):
            ax.xaxis.set_visible(False)
            ax.get_yaxis().set_ticks([])
            # ax.set_aspect('equal')
            if ax_id == 0:
                ax.set_title(names[label_id])

        axs[ax_id, 0].set_ylabel(layer)

    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.2)
    # axs[0,0].legend(handles=legend_elements,  loc='upper center',
    #            mode='expand', ncol=len(roi_names), bbox_to_anchor=(1.5, 2.5))
    plt.tight_layout()
    plt.savefig("./{}.png".format("contour_slices"), dpi=2000)

    # plt.show()
    plt.close()
