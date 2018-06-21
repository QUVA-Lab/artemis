import os
import pickle
import time
from collections import OrderedDict


class PersistentOrderedDict(object):
    """
    A Persistent ordered dict.  Example:

    pod1 = PersistentOrderedDict('my_file.pkl')
    pod1['a'] = 1234

    pod2 = PersistentOrderedDict('my_file.pkl')
    assert pod2['a'] == 1234

    This is similar to python's built in "shelve" module, but
    - It is ordered,
    - There is no need to close (writing is done every time a key is set).

    This is not thread-safe, but it should be fine for situations when you only have one process writing to
    the object.
    """
    VERSION_IDENTIFIER = 'PersistentOrderedDictV2'

    def __init__(self, file_path, items=(), pickle_protocol=pickle.HIGHEST_PROTOCOL):
        """

        :param file_path: Path to file
        :param items: Optionally, the items to initially write
        :param pickle_protocol: Pickle protocol to use.
        """
        self.file_path = file_path
        self.pickle_protocol = pickle_protocol
        self._enable_write = True
        self._inner_dict = OrderedDict(items)
        self._last_check_code = None
        self._dict = OrderedDict()
        self._update_from_file()
        with self:
            for k, v in items:
                self[k] = v

    def _update_from_file(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'rb') as f:
                version = pickle.load(f)
                if version == self.VERSION_IDENTIFIER:
                    code = pickle.load(f)
                    if code==self._last_check_code:
                        return
                    else:
                        self._dict = pickle.load(f)
                else:
                    self._dict = dict(version)  # Backwards Compatibility

    def _write_file(self):
        make_file_dir(self.file_path)
        with open(self.file_path, 'wb') as f:
            self._last_check_code = hash((id(self), time.time()))
            pickle.dump(self.VERSION_IDENTIFIER)
            pickle.dump(self._last_check_code, f, protocol=self.pickle_protocol)
            pickle.dump(self._dict, f, protocol=self.pickle_protocol)

    def has_changed(self):
        """
        Just check if there has been an external change
        :return:
        """
        with open(self.file_path, 'rb') as f:
            code = pickle.load(f)
            return code!=self._last_check_code

    def __contains__(self, key):
        self._update_from_file()
        return key in self._dict

    def __setitem__(self, key, value):

        if self._enable_write:
            self._update_from_file()
            self._dict[key] = value
            self._write_file()
        else:
            self._dict[key] = value
            self._change_made = True

    def __getitem__(self, key):
        self._update_from_file()
        return self._dict[key]

    def items(self):
        self._update_from_file()
        return self._dict.items()

    def __enter__(self):
        self._change_made = False
        self._enable_write = False
        return self

    def __exit__(self, thing1, thing2, thing3):
        if self._change_made:
            self._write_file()
        self._enable_write = True

    def get_data(self):
        return self._dict.copy()


def make_file_dir(full_file_path):
    """
    Make the directory containing the file in the given path, if it doesn't already exist
    :param full_file_path:
    :returns: The directory.
    """
    full_local_dir, _ = os.path.split(full_file_path)
    try:
        os.makedirs(full_local_dir)
    except OSError:
        pass
    return full_file_path
