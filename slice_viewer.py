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


def slice_viewer(volumes, title = None, shape = None ):
    """
    Function to view slices of 3d volumes using matplotlib
    use j and k to scroll through slices
    Args:
        volumes: array with volumes
        title: Optional array with title for each plot


    """
    remove_keymap_conflicts({'j', 'k'})
    if shape is not None:
        fig, ax = plt.subplots(shape[0], shape[1])
        ax = np.array(ax).flatten()
        while len(ax) > len(volumes):
            fig.delaxes(ax[-1])
            ax = ax[:-1]

    else:
        fig, ax = plt.subplots(1, len(volumes))


    for i in range(len(volumes)):
        ax[i].volume = volumes[i]
        ax[i].index = volumes[i].shape[0] // 2
        ax[i].im = ax[i].imshow(volumes[i][ax[i].index], vmin = np.amin(volumes), vmax = np.amax(volumes))

        if title is not None:
            ax[i].set_title(title[i])

    # add slice numbers
    ax[0].set_ylabel('slice {}'.format(ax[0].index))

    # add color bar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(ax[-1].im, cax=cbar_ax)

    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()


def process_key(event):
    fig = event.canvas.figure
    axes = fig.axes[:-1]
    if event.key == 'j':
        previous_slice(axes)
    elif event.key == 'k':
        next_slice(axes)
    fig.canvas.draw()


def previous_slice(axes):
    for ax in axes:
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[0].set_array(volume[ax.index])
    axes[0].set_ylabel('slice {}'.format(axes[0].index))


def next_slice(axes):
    for ax in axes:
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])
    axes[0].set_ylabel('slice {}'.format(axes[0].index))


