import time


class Speedometer(object):

    def __init__(self, mode='last', initial_progress = 0):
        assert mode in ('last', 'average')
        self._mode = mode
        self._start_time = self._last_time = time.time()
        self._last_progress = initial_progress
        self._pause_time = 0

    def __call__(self, progress=None):
        if progress is None:
            progress = self._last_progress + 1
        this_time = time.time()
        speed = (progress - self._last_progress) / (this_time - self._last_time)
        if self._mode == 'last':
            self._last_progress = progress
            self._last_time = this_time
        return speed

