from time import time
from collections import OrderedDict
__author__ = 'peter'


class EZProfiler(object):

    def __init__(self, print_result = True, profiler_name = 'Profile', record_stop = True):
        self.print_result = print_result
        self.profiler_name = profiler_name
        self.record_stop = record_stop
        self._lap_times = OrderedDict()
        self._lap_times['Start'] = time()

    def lap(self, lap_name = None):
        """
        :param lap_name: How to identify this name
        """
        assert lap_name not in ('Start', 'Stop'), "Names 'Start' and 'Stop' are reserved."
        assert lap_name not in self._lap_times, "You already have a chekpoint called '%s'" % (lap_name, )
        self._lap_times[lap_name] = time()

    def __enter__(self):
        start_time = time()
        self.start_time = start_time
        return self

    def __exit__(self, *args):
        if self.record_stop:
            self._lap_times['Stop'] = time()
        if self.print_result:
            self.print_elapsed()

    def print_elapsed(self):
        print self.get_report()

    def get_report(self):
        keys = self._lap_times.keys()
        deltas = OrderedDict((key, self._lap_times[key] - self._lap_times[last_key]) for last_key, key in zip(keys[:-1], keys[1:]))
        return self.profiler_name + '\n  '.join(['']+['%s: Elapsed time is %.4gs' % (key, val) for key, val in deltas.iteritems()] +
            (['Total: %.4gs' % (self._lap_times.values()[-1] - self._lap_times.values()[0])] if len(deltas)>1 else []))
