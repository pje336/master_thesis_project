import matplotlib.pyplot as plt


class IndexTracker:
    """
    class to update the plot for every slice.
    based on: https://matplotlib.org/stable/gallery/event_handling/image_slices_viewer.html

    :param axes: array with plt.axes
    :param Xs: array of data-arrays in shape [z,x,y]
    """

    def __init__(self, axes, Xs):
        self.slices, rows, cols = Xs[0].shape
        self.ind = self.slices // 2

        # check how many axes there are.
        try:
            self.length = len(axes)
        except:
            self.length = axes.numCols
            axes = [axes]

        # make a self item for each axes.
        for i, ax in enumerate(axes):
            exec("self.ax{} = ax".format(i))
            exec("self.X{} = Xs[{}]".format(i, i))
            exec("self.im{} = self.ax{}.imshow(self.X{}[self.ind,:, :])".format(i, i, i))
        self.update()

    def on_scroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        for i in range(self.length):
            exec("self.im{} = self.ax{}.imshow(self.X{}[self.ind, :, :])".format(i, i, i))
            exec("self.ax{}.set_ylabel('slice %s' % self.ind)".format(i))
            exec("self.im{}.axes.figure.canvas.draw()".format(i))


def slice_viewer(CT_data):
    """
    function to view slices of CT data using plt. You can scroll through the slices.
    :param CT_data: array of data-arrays in shape [z,x,y]
    """
    fig, axes = plt.subplots(1, len(CT_data))
    tracker = IndexTracker(axes, CT_data)
    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    plt.show()
