from tempfile import mkstemp
import cProfile
import pstats
from artemis.general.display import surround_with_header
import os


def what_are_we_waiting_for(command, sort_by ='time', max_len = 20, print_here = True):
    """
    An easy way to show what is taking all the time when you run something.
    Taken from docs: https://docs.python.org/2/library/profile.html#module-cProfile

    :param command: A string python command
    :param sort_by: How to sort results. {'time', 'cumtime', 'calls', ...}.
        See https://docs.python.org/2/library/profile.html#pstats.Stats.sort_stats
    :param max_len: Maximum number of things to show in profile.
    :param print_here: Print the results here (instead of returning them).
    :return: A pstats.Stats object containing the profiling results.
    """
    _, filepath = mkstemp()
    try:
        cProfile.run(command, filepath)

    finally:
        p = pstats.Stats(filepath)
        os.remove(filepath)
        p.strip_dirs()
        p.sort_stats(sort_by)
        if print_here:
            print(surround_with_header('Profile for "{}"'.format(command), width=100, char='='))
            p.print_stats(max_len)
            print('='*100)
        return p
