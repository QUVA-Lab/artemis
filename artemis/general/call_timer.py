
import time
_call_counts = {}


class CallStatus(object):

    def __init__(self, name, print_time = None):
        self.name = name
        self.start_time = time.time()
        self.call_count = 0
        self.print_time = print_time
        self.last_print_time = -float('inf')

    def update(self):
        current_time = time.time()
        self.call_count += 1

        if self.print_time is None or (current_time - self.last_print_time) > self.print_time:
            print 'Timer {}: {} iterations in {:.3}s ({:.3} iterations/second)'.format(self.name, self.call_count, current_time-self.start_time, self.call_count/(current_time-self.start_time))
            self.last_print_time = current_time


def report_call_timing(name, print_time=2):

    if name not in _call_counts:
        _call_counts[name] = CallStatus(name, print_time=print_time)

    _call_counts[name].update()
