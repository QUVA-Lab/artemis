

class Looper(object):

    QUIT = object()

    def __init__(self):
        self.iter = 0
        self.start_time = None
        self._callbacks = []

    def add_callback(self, callback, condition = None):
        self._callbacks.append((callback, condition))

    def add_one_iteration_callback(self, callback, iter):
        self.add_callback(callback, condition=lambda: self.iter==iter)

    def run(self, n_steps = None):
        if n_steps is not None:
            self.add_onetime_iteration_callback(lambda: False, )
        should_continue = True
        while True:
            for cb, condition in self._callbacks:
                if condition is None or condition():
                    should_continue = should_continue and cb() is not Looper.QUIT
                if not should_continue:
                    break
