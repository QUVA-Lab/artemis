

"""
Thanks to Brice for this piece of code.  Taken from https://stackoverflow.com/a/9969000/851699

"""
from collections import Iterable
import sys
if sys.version_info < (3, 0):
    from Queue import Queue
else:
    from queue import Queue
from threading import Thread


class Iteratorize(Iterable):
    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """

    def __init__(self, func):
        """
        :param Callable[Callable, Any] func: A function that takes a callback as an argument then runs.
        """
        self.mfunc = func
        self.q = Queue(maxsize=1)
        self.sentinel = object()

        def _callback(val):
            self.q.put(val)

        def gentask():
            ret = self.mfunc(_callback)
            self.q.put(self.sentinel)

            # start_new_thread(gentask, ())
        Thread(target=gentask).start()

    def __iter__(self):
        return self

    def __next__(self):

        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj
