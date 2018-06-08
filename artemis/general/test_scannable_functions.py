import numpy as np
from artemis.general.scannable_functions import scannable


def test_moving_average():

    @scannable(state=dict(avg=0))
    def moving_average(x, avg, decay):
        return (1-decay)*avg + decay*x

    seq = np.random.randn(100) + np.sin(100/5)

    f = moving_average.scan()
    simply_smoothed_signal = [f(x=x, decay=1./(t+1)) for t, x in enumerate(seq)]
    truth = np.cumsum(seq)/np.arange(1, len(seq)+1)
    assert np.allclose(simply_smoothed_signal, truth)
    assert list(f.state.keys())==['avg']
    assert np.allclose(f.state['avg'], np.mean(seq))

    f = moving_average.scan()
    exponentially_smoothed_signal = [f(x=x, decay=0.1) for x in seq]
    truth = [avg for avg in [0] for x in seq for avg in [0.9*avg + 0.1*x]]
    assert np.allclose(exponentially_smoothed_signal, truth)


def test_rnn_type_comp():

    n_samples = 10
    n_in, n_hid, n_out = 3, 4, 5

    rng = np.random.RandomState(1234)

    w_xh = rng.randn(n_in, n_hid)
    w_hh = rng.randn(n_hid, n_hid)
    w_hy = rng.randn(n_hid, n_out)

    @scannable(state={'hid': np.zeros(n_hid)}, output=['out', 'hid'], returns='out')
    def rnn_like_func(x, hid):
        new_hid = np.tanh(x.dot(w_xh) + hid.dot(w_hh))
        out = new_hid.dot(w_hy)
        return out, new_hid

    seq = np.random.randn(n_samples, n_in)

    initial_state = rng.randn(n_hid)

    # The OLD way of doing things.
    outputs = []
    h = initial_state
    for x in seq:
        y, h = rnn_like_func(x, h)
        outputs.append(y)

    # The NEW way of doing things.
    rnn_step = rnn_like_func.scan(hid=initial_state)
    outputs2 = [rnn_step(x) for x in seq]
    assert np.allclose(outputs, outputs2)
    assert np.allclose(rnn_step.state['hid'], h)


if __name__ == '__main__':
    test_moving_average()
    test_rnn_type_comp()
