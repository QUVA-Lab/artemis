import time


class ProgressIndicator(object):

    def __init__(self, expected_iterations, name=None, update_every = (2, 'seconds'), post_info_callback = None, just_use_last=False):

        self._expected_iterations = expected_iterations
        update_interval, update_unit = update_every
        assert update_unit in ('seconds', 'percent', 'iterations')

        if update_unit == 'percent':
            update_unit = 'iterations'
            update_interval = update_interval(expected_iterations)
        self.name = name
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
        self.just_use_last = just_use_last
        self._last_time = self._start_time
        self._last_progress = 0

    def __call__(self, iteration = None):
        self.print_update(iteration)

    def print_update(self, progress=None):
        self._current_time = time.time()
        self._i = self._i+1 if progress is None else progress
        if self._should_update() or self._i == self._expected_iterations:
            frac = float(self._i)/self._expected_iterations
            elapsed = self._current_time - self._start_time
            if self.just_use_last:
                remaining = (self._current_time - self._last_time)/(frac - self._last_progress) * (1-frac) if frac > 0 else float('NaN')
            else:
                remaining = elapsed * (1 / frac - 1) if frac > 0 else float('NaN')
            elapsed = self._current_time - self._start_time
            self._last_update = self._i if self._update_unit == 'iterations' else self._current_time
            print 'Progress%s: %s%%.  %.1fs Elapsed, %.1fs Remaining.%s' \
                % ('' if self.name is None else ' of '+self.name, int(100*frac), elapsed, remaining, (', %s' % (self._post_info_callback(), )) if self._post_info_callback is not None else '')
        if self.just_use_last:
            self._last_time = self._current_time
            self._last_progress = frac

    def _should_update_time(self):
        return self._current_time-self._last_update > self._update_interval

    def _should_update_iter(self):
        return self._i - self._last_update > self._update_interval
