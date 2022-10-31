"""
Code to view an overlay of the contours on the CT images.
"""

# Code taken from https://www.datacamp.com/tutorial/matplotlib-3d-volumetric-data
# https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots


import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def contour_viewer(volumes, title=None, shape=None, flow_field=None, roi_names=None):
    """
    Function to view slices of 3d volumes using matplotlib
    use j and k to scroll through slices
    Args:
        volumes: array with volumes
        title: Optional array with title for each plot


    """
    #
    colors_array = ['purple', 'blue', 'red', 'yellow', 'orange', 'lawngreen', 'lightseagreen', 'pink', 'saddlebrown',
                    'purple']
    cmap = colors.ListedColormap(colors_array)
    bounds = [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    legend_elements = []

    for index, roi_name in enumerate(roi_names):
        legend_elements.append(Patch(facecolor=colors_array[index + 1],
                                     label=roi_name))


    remove_keymap_conflicts({'j', 'k', 'J', 'K'})
    if shape is not None:
        fig, ax = plt.subplots(shape[0], shape[1], figsize=(shape[1] * 3.1, shape[0] * 3.1))
        ax = np.array(ax).flatten()
        while len(ax) > len(volumes) + 1:
            fig.delaxes(ax[shape[1] - 1])
            ax = np.delete(ax, shape[1] - 1)

    else:
        if flow_field is not None:
            fig, ax = plt.subplots(1, len(volumes) + 1, figsize=(12, 3))
        else:
            fig, ax = plt.subplots(1, len(volumes) // 2)

    for i in range(len(volumes) // 2):
        ax[i].ct = volumes[i]
        ax[i].contour = volumes[i + len(volumes) // 2]
        ax[i].contour[ax[i].contour == 0] = np.nan
        ax[i].index = volumes[i].shape[0] // 2
        ax[i].im = ax[i].imshow(ax[i].ct[ax[i].index], cmap="gray")
        ax[i].im = ax[i].imshow(ax[i].contour[ax[i].index], interpolation='none', vmin=0, cmap=cmap)

        if title is not None:
            ax[i].set_title(title[i])
    if flow_field is not None:
        grid_size = 8
        dimensions = np.shape(flow_field)
        ax[-1].index = dimensions[1] // 2
        x = np.linspace(0, dimensions[2] - 1, dimensions[2])
        y = np.linspace(0, dimensions[3] - 1, dimensions[3])
        xv, yv = np.meshgrid(x, y)
        ax[-1].clear()
        ax[-1].im = ax[-1].quiver(xv[::grid_size, ::grid_size], yv[::grid_size, ::grid_size],
                                  flow_field[0, ax[-1].index, ::grid_size, ::grid_size, 1],
                                  flow_field[0, ax[-1].index, ::grid_size, ::grid_size, 2])
        ax[-1].set_ylim(ax[-1].get_ylim()[::-1])
        ax[-1].volume = [flow_field, xv, yv, grid_size]
        ax[-1].set_title("x-y DVF")

    # add slice numbers
    ax[0].set_ylabel('slice {}'.format(ax[0].index))
    plt.legend(handles=legend_elements, bbox_to_anchor=(-0.5, -0.5), loc="lower right",
               mode='expand', ncol=2)

    fig.canvas.mpl_connect('key_press_event', process_key)
    # plt.tight_layout()
    plt.show()


def process_key(event):
    fig = event.canvas.figure
    axes = fig.axes
    if event.key == 'j' or event.key == 'J':
        previous_slice(axes)
    elif event.key == 'k' or event.key == 'K':
        next_slice(axes)
    fig.canvas.draw()


def previous_slice(axes):
    for ax in axes:
        ct = ax.ct
        contour = ax.contour

        ax.index = (ax.index - 1) % ct.shape[0]  # wrap around using %
        ax.images[0].set_array(ct[ax.index])
        ax.images[1].set_array(contour[ax.index])
    axes[0].set_ylabel('slice {}'.format(axes[0].index))


def next_slice(axes):
    for ax in axes:
        ct = ax.ct
        contour = ax.contour
        ax.index = (ax.index + 1) % ct.shape[0]
        ax.images[0].set_array(ct[ax.index])
        ax.images[1].set_array(contour[ax.index])
    axes[0].set_ylabel('slice {}'.format(axes[0].index))
