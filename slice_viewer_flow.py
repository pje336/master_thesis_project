# Code taken from https://www.datacamp.com/tutorial/matplotlib-3d-volumetric-data
# https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots


import matplotlib.pyplot as plt
import numpy as np
def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def slice_viewer(volumes, title = None, shape = None, flow_field = None ):
    """
    Function to view slices of 3d volumes using matplotlib
    use j and k to scroll through slices
    Args:
        volumes: array with volumes
        title: Optional array with title for each plot


    """
    remove_keymap_conflicts({'j', 'k', 'J', 'K'})
    if shape is not None:
        fig, ax = plt.subplots(shape[0], shape[1])
        ax = np.array(ax).flatten()
        while len(ax) > len(volumes):
            fig.delaxes(ax[-1])
            ax = ax[:-1]

    else:
        if flow_field is not None:
           fig, ax = plt.subplots(1, len(volumes)+1, figsize  = (12,3))
        else:
            fig, ax = plt.subplots(1, len(volumes))


    for i in range(len(volumes)):
        ax[i].volume = volumes[i]
        ax[i].index = volumes[i].shape[0] // 2
        ax[i].im = ax[i].imshow(volumes[i][ax[i].index], vmin = np.amin(volumes), vmax = np.amax(volumes))

        if title is not None:
            ax[i].set_title(title[i])
    if flow_field is not None:
        grid_size = 4
        dimensions = np.shape(flow_field)
        ax[-1].index = dimensions[0]//2

        x = np.linspace(0, dimensions[1] - 1, dimensions[1])
        y = np.linspace(0, dimensions[2] - 1, dimensions[2])
        xv, yv = np.meshgrid(x, y)
        ax[-1].im = ax[-1].quiver(xv[::grid_size,::grid_size], yv[::grid_size,::grid_size],flow_field[ax[-1].index,::grid_size,::grid_size,1], flow_field[ax[-1].index,::grid_size,::grid_size,2])
        ax[-1].set_ylim(ax[-1].get_ylim()[::-1])
        ax[-1].volume = [flow_field, xv, yv, grid_size]

    # add slice numbers
    ax[0].set_ylabel('slice {}'.format(ax[0].index))

    # add color bar
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(ax[-1].im, cax=cbar_ax)
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.tight_layout()
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
    for ax in axes[:-1]:
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[0].set_array(volume[ax.index])
    axes[0].set_ylabel('slice {}'.format(axes[0].index))

    axes[-1].clear()
    [flow_field, xv, yv, grid_size] = axes[-1].volume
    axes[-1].index = (axes[-1].index - 1) % np.shape(flow_field)[0] # wrap around using %
    axes[-1].im = axes[-1].quiver(xv[::grid_size,::grid_size], yv[::grid_size,::grid_size],flow_field[axes[-1].index,::grid_size,::grid_size,1], flow_field[axes[-1].index,::grid_size,::grid_size,2])
    axes[-1].set_ylim(axes[-1].get_ylim()[::-1])


def next_slice(axes):

    for ax in axes[:-1]:
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])
    axes[0].set_ylabel('slice {}'.format(axes[0].index))


    axes[-1].clear()
    [flow_field, xv, yv, grid_size] = axes[-1].volume
    axes[-1].index =  (axes[-1].index + 1) % np.shape(flow_field)[0] # wrap around using %
    axes[-1].im = axes[-1].quiver(xv[::grid_size,::grid_size], yv[::grid_size,::grid_size],flow_field[axes[-1].index,::grid_size,::grid_size,1], flow_field[axes[-1].index,::grid_size,::grid_size,2])
    axes[-1].set_ylim(axes[-1].get_ylim()[::-1])


