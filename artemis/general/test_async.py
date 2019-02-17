import time
from functools import partial

from artemis.general.async import iter_asynchronously, iter_latest_asynchonously

LOAD_INTERVAL = 0.1

# SUM_INTERVAL = LOAD_INTERVAL + PROCESS_INTERVAL


def dataloader_example(upto):

    for i in range(upto):
        time.sleep(LOAD_INTERVAL)
        yield i


def test_async_dataloader():

    process_interval = 0.1
    start = time.time()
    for data in dataloader_example(upto=4):
        time.sleep(process_interval)
        elapsed = time.time()-start
        print('Sync Processed Data {} at t={:.3g}: '.format(data, elapsed))
    assert (LOAD_INTERVAL + process_interval)*4 < elapsed < (LOAD_INTERVAL + process_interval)*5
    print('Sync: {:.4g}s elapsed'.format(elapsed))

    start = time.time()
    for data in iter_asynchronously(partial(dataloader_example, upto=4)):
        time.sleep(process_interval)
        elapsed = time.time()-start
        print('Sync Processed Data {} at t={:.3g}: '.format(data, elapsed))
    print('Async: {:.4g}s elapsed'.format(elapsed))
    assert LOAD_INTERVAL + max(LOAD_INTERVAL, process_interval)*4 < elapsed < LOAD_INTERVAL + max(LOAD_INTERVAL, process_interval)*5


def test_async_value_setter():

    process_interval = 0.25

    start = time.time()
    data_points = []
    for data in iter_latest_asynchonously(gen_func = partial(dataloader_example, upto=10)):
        time.sleep(process_interval)
        data_points.append(data)
        elapsed = time.time()-start

    assert data_points[0] is None
    assert all(dn-dp > 1 for dn, dp in zip(data_points[2:], data_points[1:-1]))
    print(data_points)




if __name__ == '__main__':
    test_async_dataloader()
    test_async_value_setter()
