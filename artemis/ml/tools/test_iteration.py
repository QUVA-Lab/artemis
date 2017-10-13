from artemis.ml.tools.iteration import minibatch_index_generator, checkpoint_minibatch_index_generator, \
    zip_minibatch_iterate_info, minibatch_process

__author__ = 'peter'
import numpy as np


def test_minibatch_index_generator():

    n_samples = 48
    n_epochs = 1.5
    minibatch_size = 5

    data = np.arange(n_samples)

    expected_total_samples = int(len(data)*n_epochs)

    for slice_when_possible in (True, False):

        i = 0
        for ix in minibatch_index_generator(n_samples = n_samples, n_epochs=n_epochs, final_treatment='truncate',
                slice_when_possible = slice_when_possible, minibatch_size=minibatch_size):
            assert np.array_equal(data[ix], np.arange(i, min(expected_total_samples, i+minibatch_size)) % n_samples)
            i += len(data[ix])
        assert i == expected_total_samples == 72

        i = 0
        for ix in minibatch_index_generator(n_samples = n_samples, n_epochs=n_epochs, final_treatment='stop',
                slice_when_possible = slice_when_possible, minibatch_size=minibatch_size):
            assert np.array_equal(data[ix], np.arange(i, min(expected_total_samples, i+minibatch_size)) % n_samples)
            i += len(data[ix])
        assert i == int(expected_total_samples/minibatch_size) * minibatch_size == 70


def test_minibatch_index_even():

    n_samples = 5
    n_epochs=2

    ixs = list(minibatch_index_generator(n_samples=n_samples, n_epochs=n_epochs, minibatch_size=1, slice_when_possible=False, final_treatment='truncate'))

    assert ixs==[[i%n_samples] for i in range(n_samples*n_epochs)]


def test_checkpoint_minibatch_generator():
    n_samples = 48
    data = np.arange(n_samples)
    for checkpoints in ([0, 20, 30, 63, 100], [20, 30, 63, 100]):
        for slice_when_possible in (True, False):
            iterator = checkpoint_minibatch_index_generator(n_samples=n_samples, checkpoints=checkpoints, slice_when_possible=slice_when_possible)
            assert np.array_equal(data[next(iterator)], np.arange(20))
            assert np.array_equal(data[next(iterator)], np.arange(20, 30))
            assert np.array_equal(data[next(iterator)], np.arange(30, 63) % 48)
            assert np.array_equal(data[next(iterator)], np.arange(63, 100) % 48)
            try:
                next(iterator)
            except StopIteration:
                pass
            except:
                raise Exception("Failed to stop iteration")


def test_minibatch_iterate_info():

    n_samples = 100
    minibatch_size = 10

    training_arr = np.random.randn(n_samples, 12)
    test_arr = np.random.randint(10, size=n_samples)

    for kwargs in [dict(test_epochs=[0, 0.5, 1, 1.5]), dict(test_epochs=('every', 0.5), n_epochs=1.5)]:
        # Two different ways of parametrixing minibatch loop:
        # test_epochs=[0, 0.5, 1, 1.5]:  "Test at these epochs.  Done after final test"
        # test_epochs=('every', 0.5), n_epochs=1.5  : "Test every 0.5 epochs.  Done after test at epcoh 1.5

        test_epochs = []
        iterator = 0

        for (training_minibatch, test_minibatch), info in zip_minibatch_iterate_info((training_arr, test_arr),
                    minibatch_size=minibatch_size, **kwargs):

            ixs = (np.arange(minibatch_size)+iterator*minibatch_size) % n_samples
            epoch = iterator * minibatch_size / float(n_samples)
            print((epoch, info.epoch))
            print(info.done)
            assert np.allclose(epoch, info.epoch)
            assert np.array_equal(ixs, np.arange(info.sample, info.sample+minibatch_size) % n_samples)
            assert np.array_equal(training_minibatch, training_arr[ixs])
            assert np.array_equal(test_minibatch, test_arr[ixs])

            if info.test_now:
                test_epochs.append(info.epoch)

            if not info.done:
                iterator += 1

        assert iterator == 15
        assert test_epochs == [0, 0.5, 1, 1.5]  # Note... last test time is not included... difficult to say what should be expected here.


def test_minibatch_process():

    x = np.random.randn(5, 3)
    mat = np.random.randn(3, 4)

    def func(x):
        return x.dot(mat)

    y1 = func(x)
    assert y1.shape==(5, 4)
    y2 = minibatch_process(func, minibatch_size=2, mb_args=(x, ))

    assert np.allclose(y1, y2)  # weird numpy rounding makes it not exactly equal


if __name__ == '__main__':
    test_minibatch_index_even()
    test_minibatch_process()
    test_minibatch_iterate_info()
    test_minibatch_index_generator()
    test_checkpoint_minibatch_generator()
