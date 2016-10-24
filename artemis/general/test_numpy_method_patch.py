__author__ = 'peter'
import artemis.general.numpy_method_patch as np


def test_numpy_method_patch():

    assert np.all(np.arange(-5, 5).abs() == range(5, 0, -1)+range(5))

    # Or more succinctly
    assert (np.arange(-5, 5).abs() == range(5, 0, -1)+range(5)).all()


if __name__ == '__main__':

    test_numpy_method_patch()
