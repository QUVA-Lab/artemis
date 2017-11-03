from artemis.plotting.data_conversion import put_data_in_grid
from matplotlib import pyplot as plt
import numpy as np


def select_image(images, selection_callback):
    """
    Givel an array of images, show them all and allow seection by clicking.
    :param images:
    :param selection_callback:
    :return:
    """

    data = put_data_in_grid(images, is_color_data=True)

    plt.figure(figsize=(12, 12))
    plt.imshow(data)
    plt.gca().tick_params(labelbottom = 'off')
    plt.gca().tick_params(labelleft = 'off')

    # dbplot(first_ims, 'First BBox Images')
    def callback(event):
        n_cols = int(np.ceil(np.sqrt(len(images))))
        n_rows = int(np.ceil(len(images)/n_cols))
        ax_size_x = plt.gca().get_xlim()[1]
        ax_size_y = plt.gca().get_ylim()[0]

        frac_x = event.xdata/ax_size_x
        frac_y = event.ydata/ax_size_y
        print (frac_x, frac_y)

        col_ix = int(n_cols*frac_x)
        row_ix = int((n_rows+1)*frac_y)

        ix = row_ix*n_cols + col_ix

        selection_callback(ix)

    plt.gcf().canvas.callbacks.connect('button_press_event', callback)
    plt.show()


def select_pixel(image, selection_callback):

    # plt.figure()
    plt.imshow(image)

    def callback(event):
        selection_callback((int(event.ydata), int(event.xdata)))

    plt.gcf().canvas.callbacks.connect('button_press_event', callback)
    plt.show()
