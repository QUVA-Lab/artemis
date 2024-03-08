from artemis.general.progress_indicator import ProgressIndicator
import time

def test_progress_inidicator():

    n_iter = 100

    pi = ProgressIndicator(n_iter, update_every='1s')

    start=time.time()
    for i in range(n_iter):
        time.sleep(0.001)
        if i % 10==0:
            with pi.pause_measurement():
                time.sleep(0.02)

    assert pi.get_elapsed() < (time.time() - start)/2.


if __name__ == '__main__':
    test_progress_inidicator()
