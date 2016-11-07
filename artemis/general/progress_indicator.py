import time


class ProgressIndicator(object):

    def __init__(self, expected_iterations, update_every = (2, 'seconds'), post_info_callback = None):

        self._expected_iterations = expected_iterations
        update_interval, update_unit = update_every
        assert update_unit in ('seconds', 'percent', 'iterations')

        if update_unit == 'percent':
            update_unit = 'iterations'
            update_interval = update_interval(expected_iterations)
        self._update_unit = update_unit
        self._update_interval = update_interval
        self._start_time = time.time()
        self._should_update = {
            'iterations': self._should_update_iter,
            'seconds': self._should_update_time,
            }[self._update_unit]

        self._i = 1
        self._last_update = -float('inf')
        self._post_info_callback = post_info_callback

    def __call__(self, iteration = None):
        self.print_update(iteration)

    def print_update(self, iteration=None):
        self._i = self._i+1 if iteration is None else iteration+1
        self._current_time = time.time()
        if self._should_update() or self._i == self._expected_iterations:
            frac = float(self._i)/self._expected_iterations
            elapsed = time.time() - self._start_time
            remaining = elapsed * (1/frac-1) if frac > 0 else float('NaN')
            self._last_update = self._i if self._update_unit == 'iterations' else self._current_time
            print 'Progress: %s%%.  %.1fs Elapsed, %.1fs Remaining.%s' \
                % (int(100*frac), elapsed, remaining, (', %s' % (self._post_info_callback(), )) if self._post_info_callback is not None else '')


    def _should_update_time(self):
        return self._current_time-self._last_update > self._update_interval

    def _should_update_iter(self):
        return self._i - self._last_update > self._update_interval
