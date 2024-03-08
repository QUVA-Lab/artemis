from logging import Logger
from time import time
from collections import OrderedDict
from contextlib import contextmanager
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

    def get_total_time(self):
        assert 'Stop' in self._lap_times, "The profiler has not exited yet, so you cannot get total time."
        return self._lap_times['Stop'] - self._lap_times['Start']

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


_profile_contexts = OrderedDict()


@contextmanager
def profile_context(name, print_result = False):

    with EZProfiler(name, print_result=print_result) as prof:
        yield prof
    if name in _profile_contexts:
        n_calls, elapsed = _profile_contexts[name]
    else:
        n_calls, elapsed = 0, 0.
    n_calls, elapsed = n_calls+1, elapsed + prof.get_total_time()
    _profile_contexts[name] = (n_calls, elapsed)


def get_profile_contexts(names=None, fill_empty_with_zero = False):
    """
    :param names: Names of profiling contexts to get (from previous calls to profile_context).  If None, use all.
    :param fill_empty_with_zero: If names are not found, just fill with zeros.
    :return: An OrderedDict <name: (n_calls, elapsed)>
    """
    if names is None:
        return _profile_contexts
    else:
        if fill_empty_with_zero:
            return OrderedDict((k, _profile_contexts[k] if k in _profile_contexts else (0, 0.)) for k in names)
        else:
            return OrderedDict((k, _profile_contexts[k]) for k in names)


def get_profile_contexts_string(names=None, fill_empty_with_zero = False):

    profile = get_profile_contexts(names=names, fill_empty_with_zero=fill_empty_with_zero)
    string = ', '.join('{}: {:.3g}s/iter'.format(name, elapsed/n_calls) for name, (n_calls, elapsed) in profile.items())
    return string
