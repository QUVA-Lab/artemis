import numpy as np

__author__ = 'peter'


def flatten_struct(struct, primatives = (int, float, np.ndarray, basestring, bool), custom_handlers = {},
        break_into_objects = True, detect_duplicates = True, memo = None):
    """
    Given some nested struct, return a list<*(str, primative)>, where primative
    is some some kind of object that you don't break down any further, and str is a
    string representation of how you would access that propery from the root object.

    Don't try any fancy circular references here, it's not going to go well for you.

    :param struct: Something, anything.
    :param primatives: A list of classes that will not be broken into.
    :param custum_handlers: A dict<type:func> where func has the form data = func(obj).  These
        will be called if the type of the struct is in the dict of custom handlers.
    :param break_into_objects: True if you want to break into objects to see what's inside.
    :return: list<*(str, primative)>
    """
    if memo is None:
        memo = {}

    if id(struct) in memo:
        return [(None, memo[id(struct)])]
    elif detect_duplicates:
        memo[id(struct)] = 'Already Seen object at %s' % hex(id(struct))

    if isinstance(struct, primatives):
        return [(None, struct)]
    elif isinstance(struct, tuple(custom_handlers.keys())):
        handler = custom_handlers[custom_handlers.keys()[[isinstance(struct, t) for t in custom_handlers].index(True)]]
        return [(None, handler(struct))]
    elif isinstance(struct, dict):
        return sum([
            [("[%s]%s" % (("'%s'" % key if isinstance(key, str) else key), subkey if subkey is not None else ''), v)
                for subkey, v in flatten_struct(value, custom_handlers=custom_handlers, primatives=primatives, break_into_objects=break_into_objects, memo=memo, detect_duplicates=detect_duplicates)]
                for key, value in struct.iteritems()
            ], [])
    elif isinstance(struct, (list, tuple)):
        # for i, value in enumerate(struct):
        return sum([
            [("[%s]%s" % (i, subkey if subkey is not None else ''), v)
                for subkey, v in flatten_struct(value, custom_handlers=custom_handlers, primatives=primatives, break_into_objects=break_into_objects, memo=memo, detect_duplicates=detect_duplicates)]
                for i, value in enumerate(struct)
            ], [])
    elif struct is None or not hasattr(struct, '__dict__'):
        return []
    elif break_into_objects:  # It's some kind of object, lets break it down.
        return sum([
            [(".%s%s" % (key, subkey if subkey is not None else ''), v)
                for subkey, v in flatten_struct(value, custom_handlers=custom_handlers, primatives=primatives, break_into_objects=break_into_objects, memo=memo, detect_duplicates=detect_duplicates)]
                for key, value in struct.__dict__.iteritems()
            ], [])
    else:
        return [(None, memo[id(struct)])]


class ExpandingDict(dict):

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            self[key] = ExpandingDict()
            return dict.__getitem__(self, key)


def expand_struct(struct):

    expanded_struct = ExpandingDict()

    for k in struct.keys():
        exec('expanded_struct%s = struct["%s"]' % (k, k))

    return expanded_struct