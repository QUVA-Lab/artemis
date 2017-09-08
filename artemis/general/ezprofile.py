from logging import Logger
from time import time
from collections import OrderedDict

__author__ = 'peter'


class EZProfiler(object):

    def __init__(self, profiler_name = 'Profile', print_result = True, print_on_start=False, record_stop = True):
        """
        :param profiler_name: The name of this profiler (will be included in report)
        :param print_result: Print
        :param record_stop:
        :return:
        """
        self.print_result = print_result
        self.profiler_name = profiler_name
        self.record_stop = record_stop
        self.print_on_start = print_on_start
        self._lap_times = OrderedDict()
        self._lap_times['Start'] = time()

    def lap(self, lap_name = None):
        """
        :param lap_name: How to identify this name
        """
        assert lap_name not in ('Start', 'Stop'), "Names 'Start' and 'Stop' are reserved."
        assert lap_name not in self._lap_times, "You already have a chekpoint called '%s'" % (lap_name, )
        t = time()
        self._lap_times[lap_name] = t
        return t

    def get_current_time(self):
        return time() - self._lap_times['Start']

    def __enter__(self):
        start_time = time()
        self.start_time = start_time
        if self.print_on_start:
            print('{} Started'.format(self.profiler_name))
        return self

    def __exit__(self, *args):
        if self.record_stop:
            self._lap_times['Stop'] = time()
        if self.print_result is True:
            self.print_elapsed()
        elif isinstance(self.print_result, Logger):
            self.print_result.info(self.get_report())

    def print_elapsed(self):
        print(self.get_report())

    def get_report(self):
        keys = self._lap_times.keys()
        if len(keys)==2:
            return '%s: Elapsed time is %.4gs' % (self.profiler_name, self._lap_times['Stop']-self._lap_times['Start'])
        else:
            deltas = OrderedDict((key, self._lap_times[key] - self._lap_times[last_key]) for last_key, key in zip(keys[:-1], keys[1:]))
            return self.profiler_name + '\n  '.join(['']+['%s: Elapsed time is %.4gs' % (key, val) for key, val in deltas.items()] +
                (['Total: %.4gs' % (self._lap_times.values()[-1] - self._lap_times.values()[0])] if len(deltas)>1 else []))
