

class TraceCollection(object):

    def __init__(self):
        self._traces = {}

    def record(self, value, name):
        if name not in self._traces:
            self._traces[name] = []
        self._traces[name].append(value)

    def get_trace(self, name):
        return self._traces[name]
