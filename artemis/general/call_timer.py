
import time
_call_counts = {}


class CallStatus(object):

    def __init__(self, name, print_time = None):
        self.name = name
        self.start_time = time.time()
        self.call_count = 0
        self.print_time = print_time
        self.last_print_time = self.start_time

    def update(self):
        current_time = time.time()
        self.call_count += 1

        if self.print_time is None or (current_time - self.last_print_time) > self.print_time:
            print('Timer {}: {} iterations in {:.3}s ({:.3} iterations/second)'.format(self.name, self.call_count, current_time-self.start_time, self.call_count/(current_time-self.start_time)))
            self.last_print_time = current_time


def report_call_timing(name, print_time=2):
    """
    Print a periodic report of the number of times this function is called per second.

    This can be useful when trying to assess frequently a loop is running in your code.

    :param name: A "name" for this timer - a unique name which identifies what you're trying to time.
    :param print_time: How often (in seconds) you'd like to print an update.
    :return:
    """

    if name not in _call_counts:
        _call_counts[name] = CallStatus(name, print_time=print_time)

    _call_counts[name].update()
