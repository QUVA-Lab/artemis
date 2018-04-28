import time


class ProgressIndicator(object):

    def __init__(self, expected_iterations=None, name=None, update_every = (2, 'seconds'), post_info_callback = None, show_total=False, just_use_last=False):

        self._expected_iterations = expected_iterations
        if isinstance(update_every, str):
            assert update_every.endswith('s'), 'Only support second-counds (eg "5s" for now)'
            update_every = (int(update_every[:-1]), 'seconds')
        elif isinstance(update_every, (float, int)):
            update_every = (update_every, 'iterations')

        update_interval, update_unit = update_every
        assert update_unit in ('seconds', 'percent', 'iterations')

        if update_unit == 'percent':
            update_unit = 'iterations'
            update_interval = update_interval/100.*expected_iterations
        self.name = name
        self._update_unit = update_unit
        self._update_interval = update_interval
        self._start_time = time.time()
        self._should_update = {
            'iterations': self._should_update_iter,
            'seconds': self._should_update_time,
            }[self._update_unit]

        self._i = 0
        self._last_update = -float('inf')
        self._post_info_callback = post_info_callback
        self.just_use_last = just_use_last
        self._last_time = self._start_time
        self._last_progress = 0
        self.show_total = show_total

    def __call__(self, iteration = None):
        self.print_update(iteration)

    def print_update(self, progress=None):
        self._current_time = time.time()
        elapsed = self._current_time - self._start_time
        if self._expected_iterations is None:
            print ('Progress{}: {:.1f}s Elapsed.  {} calls averaging {:.2g} calls/s'.format(
                '' if self.name is None else ' of '+self.name,
                elapsed,
                self._i+1,
                (self._i+1)/elapsed
                ))
        else:
            if progress is None:
                progress = self._i
            frac = float(progress)/(self._expected_iterations-1) if self._expected_iterations>1 else 1.
            if self._should_update() or progress == self._expected_iterations-1:
                elapsed = self._current_time - self._start_time
                if self.just_use_last is True:
                    remaining = (self._current_time - self._last_time)/(frac - self._last_progress) * (1-frac) if frac > 0 else float('NaN')
                else:
                    remaining = elapsed * (1 / frac - 1) if frac > 0 else float('NaN')
                elapsed = self._current_time - self._start_time
                self._last_update = progress if self._update_unit == 'iterations' else self._current_time
                print('Progress{}: {}%.  {:.1f}s Elapsed, {:.1f}s Remaining{}.  {}  {} calls averaging {:.2g} calls/s'.format(
                    '' if self.name is None else ' of '+self.name,
                    int(100*frac),
                    elapsed,
                    remaining,
                    ', {:.1f}s Total'.format(elapsed+remaining) if self.show_total else '',
                    self._post_info_callback() if self._post_info_callback is not None else '',
                    self._i+1,
                    (self._i+1)/elapsed
                    ))
        self._i += 1

        if self.just_use_last is True:
            self._last_time = self._current_time
            self._last_progress = frac

    def get_elapsed(self):
        return time.time() - self._start_time

    def get_iterations(self):
        return self._i

    def _should_update_time(self):
        return self._current_time-self._last_update > self._update_interval

    def _should_update_iter(self):
        return self._i - self._last_update > self._update_interval
