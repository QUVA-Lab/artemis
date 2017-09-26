import hashlib
import pickle
from collections import OrderedDict
import itertools
import numpy as np
from six import string_types, next

_ALREADY_SEEN_CODE = 'dbf056790fabd3c7b79c1ddab7b7ee49'.encode('utf-8')
_END_CODE = 'e0abd6b36d6e295b6c8859cdffc773df'.encode('utf-8')


def fixed_hash_eq(obj1, obj2):
    """
    Return True if the objects have equal fixed-hashes.  You can use this instead of the "==" operator when comparing
    two objects that don't contain all python primatives.  All objects contained must be either primatives, numpy arrays,
    or extend the FixedHashObject interface.

    :param obj1: An arbitrary object
    :param obj2: Another arbitrary project,
    :return: A boolean, indicating whether their hashes are equal.
    """
    return compute_fixed_hash(obj1)==compute_fixed_hash(obj2)


def compute_fixed_hash(obj, try_objects=False, _hasher = None, _memo = None, _count=None):
    """
    Given an object, return a hash that will always be the same (not just for the lifetime of the
    object, but for all future runs of the program too).
    :param obj: Some nested container of primitives
    :param try_objects: Try to break into objects
    :param _hasher: (for internal use - note that this is stateful, so calling this function with this argument changes
        the hasher object)
    :param _memo: (for internal use - to remember hashed objects and avoid infinite recursion)
    :param _count: (for internal use - to uniquely identify objects with circular references)
    :return: A 32-character hexidecimal (0-f) hash code for your object.
    """
    if _memo is None:
        _memo = {}
    elif not isinstance(obj, (np.ndarray, int, float, bool)+string_types) and id(obj) in _memo:
        _hasher.update(_ALREADY_SEEN_CODE)
        _hasher.update(str(_memo[id(obj)]).encode('utf-8'))
        _hasher.update(_END_CODE)
        return _memo[id(obj)]

    if _count is None:
        _count = itertools.count()
    _memo[id(obj)] =next(_count)

    if _hasher is None:
        _hasher = hashlib.md5()

    kwargs = dict(_hasher=_hasher, try_objects=try_objects, _memo=_memo, _count=_count)

    _hasher.update(obj.__class__.__name__.encode('utf-8'))
    if isinstance(obj, np.ndarray):
        _hasher.update(pickle.dumps(obj.dtype, protocol=2))
        _hasher.update(pickle.dumps(obj.shape, protocol=2))
        _hasher.update(obj.tostring())
    elif isinstance(obj, (int, float, bool)+string_types) or (obj is None) or (obj in (int, str, float, bool)):
        _hasher.update(pickle.dumps(obj, protocol=2))
    elif isinstance(obj, (list, tuple)):
        for el in obj:
            compute_fixed_hash(el, **kwargs)
    elif isinstance(obj, set):
        for el in sorted(obj):
            compute_fixed_hash(el, **kwargs)
    elif isinstance(obj, dict):
        keys = obj.keys() if isinstance(obj, OrderedDict) else sorted(obj.keys())
        for k in keys:
            compute_fixed_hash(k, **kwargs)
            compute_fixed_hash(obj[k], **kwargs)
    elif isinstance(obj, FixedHashObject):  # See below... allows you to make custom hashables
        compute_fixed_hash(obj.get_hash_description(), **kwargs)
    elif hasattr(obj, 'memo_hashable'):  # Deprecated, just here for back-compatibility
        compute_fixed_hash(obj.memo_hashable(), **kwargs)
    elif try_objects:
        keys = sorted(obj.__dict__.keys())
        for k in keys:
            compute_fixed_hash(k, **kwargs)
            compute_fixed_hash(obj.__dict__[k], **kwargs)
    else:
        # TODO: Consider whether to pickle by default.  Note that pickle strings are not necessairly the same for identical objects.
        raise NotImplementedError("Don't have a method for hashing this %s" % (obj, ))

    _hasher.update(_END_CODE)  # Necessary to distinguish ([a, b], c) from ([a, b, c])
    result = _hasher.hexdigest()
    _memo[id(obj)] = result
    return result


class FixedHashObject(object):
    """
    Implement this interface to create an object that you can create fixed hashes form.  You can then call
    compute_fixed_hash on this object.
    """

    def get_hash_description(self):
        """
        Override this method to return some primitive (python ond numpy only) data structure that represents your object.
        This will be used to create the fixed hash.  Be careful.  It's up to you to ensure that the data stryucture you
        return fully represents the state of your object.
        :return: Any data structure consisting only of Python and Numpy primatives.
        """
        raise NotImplementedError('You need to override this method to compute a fixed hash this object: {}'.format(self))
