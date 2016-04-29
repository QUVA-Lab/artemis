__author__ = 'peter'


class KwargDealer(object):
    """
    Python's kwargs mechanism provides a nice way to optionally override default values.  But there are occasionally
    legitimate cases where the {existance of/appropriate defaults for} some kwargs depend on the values of others.  For
    these cases, we use KwargDealer to distribute the appropriate defaults.
    """
    def __init__(self, kwargs):
        self._kwargs = kwargs

    def deal(self, default_dict):
        parameter_dict = default_dict.copy()
        for k in default_dict:
            if k in self._kwargs:
                parameter_dict[k] = self._kwargs[k]
                del self._kwargs[k]
        return parameter_dict

    def assert_empty(self):
        assert len(self._kwargs)==0, 'Unused kwargs remain: %s' % (self._kwargs, )

