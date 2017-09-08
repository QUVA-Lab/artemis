import re

from six import string_types

__author__ = 'peter'


class ReDict(dict):
    """
    A dictionary where keys are regular expressions, which try to match the index you reference with.

    If the number of keys matching your index does not equal 1 (0 or >=2), you get a KeyError.

    See test_regexp_dict for examples, and documentation for python's "re" module for info on regular expressions.
    """

    def __init__(self, dict_initializer):
        """
        dict_initializer is any argument that you'd give to a dictionary with the exception that keys must be strings
            or None.  None in this case means default - in case nothing else matches.
        """
        dict.__init__(self, dict_initializer)
        assert all(isinstance(k, string_types) or k is None for k in self), 'All keys to a Redict must be strings or None'

    def __getitem__(self, index):
        match_found = False
        if index is None:
            return dict.__getitem__(self, None)
        for k, v in self.items():
            if k is not None and re.match(k, index) is not None:
                if match_found:
                    raise MultipleMatchesError('Multiple Matches to expression %s: %s.  If this is what you want use get_matches'
                        % (index, self.get_matches(index)))
                match_found = True
                value = v
        if not match_found:
            if None in self:
                value = self[None]
            else:
                raise KeyError('No matches to expression %s' % (index, ))
        return value

    def __contains__(self, index):

        return (None in self.keys()) if index is None else any(re.match(k, index) for k in self if k is not None)

    def get_matches(self, index, as_redict = True):
        """
        Get matching keys, NOT including the default None key.
        """
        matching_dict = {k: v for k, v in self.items() if k is not None and re.match(k, index) is not None}
        return ReDict(matching_dict) if as_redict else matching_dict

    def get(self, item, default):
        try:
            return self[item]
        except KeyError:
            return default


class MultipleMatchesError(Exception):
    pass


class ReCurseDict(ReDict):
    """
    A recursively nested ReDict.  You can build this out of nested dictionaries, so that when you reference
    an item that is another dictionary, it will reference that inner dictionary to find a match.
    """

    def __init__(self, dict_initialzer):
        """
        :param dict_initialzer: Don't try to be funny here an give dicts containing themselves.
        """
        ReDict.__init__(self, dict_initialzer)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = ReCurseDict(v)

    def __getitem__(self, item):
        result = ReDict.__getitem__(self, item)
        return result[item] if isinstance(result, ReCurseDict) else result
