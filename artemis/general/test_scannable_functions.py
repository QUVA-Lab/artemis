import numpy as np
import pytest

from artemis.general.scannable_functions import scannable, immutable_scan, mutable_scan


def test_simple_moving_average():

    seq = np.random.randn(100) + np.sin(np.linspace(0, 10, 100))

    @scannable(state=['avg', 'n'], returns=['avg', 'n'], output='avg')
    def simple_moving_average(x, avg=0, n=0):
        return (n/(1.+n))*avg + (1./(1.+n))*x, n+1

    f = simple_moving_average.mutable_scan()
    averaged_signal = [f(x=x) for t, x in enumerate(seq)]
    truth = np.cumsum(seq)/np.arange(1, len(seq)+1)
    assert np.allclose(averaged_signal, truth)
    assert np.allclose(f.avg, np.mean(seq))


def test_moving_average():

    @scannable(state='avg')
    def moving_average(x, decay, avg=0):
        return (1-decay)*avg + decay*x

    seq = np.random.randn(100) + np.sin(np.linspace(0, 10, 100))

    f = moving_average.mutable_scan()
    simply_smoothed_signal = [f(x=x, decay=1./(t+1)) for t, x in enumerate(seq)]
    truth = np.cumsum(seq)/np.arange(1, len(seq)+1)
    assert np.allclose(simply_smoothed_signal, truth)
    assert list(f._fields)==['avg']
    assert np.allclose(f.avg, np.mean(seq))

    f = moving_average.mutable_scan()
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

    @scannable(state='hid', returns=['out', 'hid'], output='out')
    def rnn_like_func(x, hid= np.zeros(n_hid)):
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
    rnn_step = rnn_like_func.mutable_scan(hid=initial_state)
    outputs2 = [rnn_step(x) for x in seq]
    assert np.allclose(outputs, outputs2)
    assert np.allclose(rnn_step.hid, h)

    # Now try the immutable version:
    rnn_step = rnn_like_func.immutable_scan(hid=initial_state)
    outputs3 = [output for rnn_step in [rnn_step] for x in seq for rnn_step, output in [rnn_step(x)]]
    assert np.allclose(outputs, outputs3)


def test_bad_beheviour_caught():
    seq = np.random.randn(100) + np.sin(np.linspace(0, 10, 100))

    with pytest.raises(AssertionError):  # Typo in state name
        @scannable(state='avgfff')
        def moving_average_with_typo(x, decay, avg=0):
            return (1-decay)*avg + decay*x
        f = moving_average_with_typo.mutable_scan()

    with pytest.raises(AssertionError):  # Should really be done before instance-creation, but whatever.
        @scannable(state='avg', returns='avgf')
        def moving_average_with_typo(x, decay, avg=0):
            return (1-decay)*avg + decay*x
        f = moving_average_with_typo.mutable_scan()

    with pytest.raises(ValueError):  # Invalid return name
        @scannable(state=['avg'], returns=['avg'], output='avgf')
        def moving_average_with_typo(x, decay, avg=0):
            return (1-decay)*avg + decay*x
        f = moving_average_with_typo.mutable_scan()

    with pytest.raises(TypeError):  # Wrong output format
        @scannable(state=['avg'], returns=['avg', 'something'], output='avg')
        def moving_average_with_typo(x, decay, avg=0):
            return (1-decay)*avg + decay*x
        f = moving_average_with_typo.mutable_scan()
        simply_smoothed_signal = [f(x=x, decay=1./(t+1)) for t, x in enumerate(seq)]


def test_stateless_updater():

    # Direct API
    def moving_average(x, avg=0, t=0):
        t_next = t+1.
        return avg*t/t_next+x/t_next, t_next

    sup = immutable_scan(moving_average, state=['avg', 't'], returns = ['avg', 't'], output='avg')
    sup2, avg = sup(3)
    assert avg==3
    sup3, avg = sup2(4)
    assert avg == 3.5
    sup2a, avg = sup2(1)
    assert avg == 2


def test_stateless_updater_with_decorator():
    # Using Decordator
    @scannable(state=['avg', 't'], output='avg', returns=['avg', 't'])
    def moving_average(x, avg=0, t=0):
        t_next = t+1.
        return avg*t/t_next+x/t_next, t_next

    sup = moving_average.immutable_scan()
    sup2, avg = sup(3)
    assert avg==3
    sup3, avg = sup2(4)
    assert avg == 3.5
    sup2a, avg = sup2(1)
    assert avg == 2


def test_stateful_updater():

    # Direct API
    def moving_average(x, avg=0, t=0):
        t_next = t+1.
        return avg*t/t_next+x/t_next, t_next

    sup = mutable_scan(moving_average, state=['avg', 't'], returns = ['avg', 't'], output='avg')
    avg = sup(3)
    assert avg==3
    avg = sup(4)
    assert avg == 3.5


def test_stateful_updater_with_decorator():
    # Using Decordator
    @scannable(state=['avg', 't'], output='avg', returns=['avg', 't'])
    def moving_average(x, avg=0, t=0):
        t_next = t+1.
        return avg*t/t_next+x/t_next, t_next

    sup = mutable_scan(moving_average, state=['avg', 't'], returns = ['avg', 't'], output='avg')
    avg = sup(3)
    assert avg==3
    avg = sup(4)
    assert avg == 3.5


if __name__ == '__main__':
    test_simple_moving_average()
    test_moving_average()
    test_rnn_type_comp()
    test_bad_beheviour_caught()
    test_stateless_updater()
    test_stateless_updater_with_decorator()
    test_stateful_updater()
    test_stateful_updater_with_decorator()