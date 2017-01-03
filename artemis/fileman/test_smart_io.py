from artemis.fileman.smart_io import smart_load, smart_save
import numpy as np


def test_smart_image_io(plot = False):

    image = smart_load('https://raw.githubusercontent.com/petered/data/master/images/artemis.jpeg', use_cache=True)
    smart_save(image[:, ::-1, :], 'output/simetra.png')
    rev_image = smart_load('output/simetra.png')
    assert np.array_equal(rev_image, rev_image)
    if plot:
        from artemis.plotting.db_plotting import dbplot
        dbplot(image, 'Artemis')
        dbplot(rev_image, 'Simetra', hang=True)


if __name__ == '__main__':
    test_smart_image_io(plot=False)