from collections import OrderedDict
import pickle
import os


class PersistentOrderedDict(OrderedDict):
    """
    A Persisten ordered dict.  Usage:

    with PersistentOrderedDict('my_file.pkl') as pod:
        pod['a'] = [1, 2, 3]
        pod['b'] = [4, 5, 6]

    with PersistentOrderedDict('my_file.pkl') as pod:
        assert pod.items() == [('a', [1, 2, 3]), ('b', [4, 5, 6])]

    This is similar to python's built in "shelve" module, but
    - It is ordered,
    - It is used in a "with" statement.
    """

    def __init__(self, file_path, pickle_protocol=2):
        self.file_path = file_path
        if os.path.exists(self.file_path):
            with open(self.file_path) as f:
                items = pickle.load(f)
        else:
            items = []
        self.pickle_protocol = pickle_protocol
        OrderedDict.__init__(self, items)

    def __enter__(self):
        return self

    def close(self):
        with open(self.file_path, 'w') as f:
            pickle.dump(self.items(), f, protocol=self.pickle_protocol)

    def __exit__(self, thing1, thing2, thing3):
        self.close()

    def get_text(self):
        return '\n'.join('{}: {}'.format(str(key), str(value)) for key, value in self.iteritems())

    def get_data(self):
        return OrderedDict(self)