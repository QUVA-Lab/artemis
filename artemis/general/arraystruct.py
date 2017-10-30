from collections import OrderedDict
from contextlib import contextmanager
import itertools
import numpy as np
from abc import ABCMeta, abstractmethod
from artemis.general.should_be_builtins import izip_equal


class UniversalCollection(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __contains__(self, item):
        raise NotImplementedError()

    @abstractmethod
    def keys(self):
        raise NotImplementedError()

    @abstractmethod
    def values(self):
        raise NotImplementedError()

    @abstractmethod
    def has_key(self, key):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, ix):
        raise NotImplementedError()

    @abstractmethod
    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __eq__(self, other):
        return self.to_struct() == (other.to_struct() if isinstance(other, UniversalCollection) else other)

    @abstractmethod
    def to_struct(self):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    def items(self):
        return zip(self.keys(), self.values())

    @staticmethod
    def from_first_key(key):
        if key is next or isinstance(key, int):
            obj = DynamicSequence()
        else:
            obj = UniversalOrderedStruct()
        return obj

    @classmethod
    def from_struct(cls, struct):
        if isinstance(struct, (list, tuple)):
            return DynamicSequence(struct)
        elif isinstance(struct, OrderedDict):
            return UniversalOrderedStruct(struct)
        elif struct is None:
            return EmptyCollection()
        else:
            raise Exception("Don't know how to load a Universal Collection from {}".format(struct))

    def map(self, f):
        new_obj = self.__class__()
        for k, v in self.items():
            new_obj[k] = f(v)
        return new_obj


class CollectionIsEmptyException(Exception):
    pass


class EmptyCollection(UniversalCollection):

    def has_key(self, key):
        return False

    def __getitem__(self, ix):
        raise CollectionIsEmptyException()

    def __setitem__(self, ix, val):
        raise CollectionIsEmptyException()

    def __contains__(self, item):
        return False

    def __len__(self):
        return 0

    def keys(self):
        return []

    def values(self):
        return []

    def to_struct(self):
        return None

    def items(self):
        return []

    @staticmethod
    def from_struct(struct):
        if struct is not None:
            raise CollectionIsEmptyException()

    def __iter__(self):
        return


class DynamicSequence(list, UniversalCollection):

    def values(self):
        return self

    def keys(self):
        return range(len(self))

    def __getitem__(self, ix):
        if isinstance(ix, slice):
            return DynamicSequence(list.__getitem__(self, ix))
        elif isinstance(ix, (list, tuple)):
            return DynamicSequence((list.__getitem__(self, i) for i in ix))
        else:
            return list.__getitem__(self, ix)

    def __setitem__(self, ix, val):
        if ix is next:
            self.append(val)
        elif ix>=len(self):
            assert ix==len(self), 'If you assign to a DynamicSequence, the index must be no more than one greater than the length of the sequence.  Got index {} for length {}'.format(ix, len(self))
            self.append(val)
        else:
            list.__setitem__(self, ix, val)

    def has_key(self, key):
        try:
            self[key]
            return True
        except IndexError:
            return False

    def items(self):
        return zip(range(len(self)), self)

    def to_struct(self):
        return list(self)

    @classmethod
    def from_struct(cls, struct):
        assert isinstance(struct, list)
        return DynamicSequence(struct)


class UniversalOrderedStruct(OrderedDict, UniversalCollection):

    OLD_CONTAINS_FLAG = False

    @classmethod
    @contextmanager
    def use_old_contains(cls):
        cls.OLD_CONTAINS_FLAG = True
        yield
        cls.OLD_CONTAINS_FLAG = False

    def __setitem__(self, ix, val):
        with UniversalOrderedStruct.use_old_contains():
            OrderedDict.__setitem__(self, ix, val)

    def __repr__(self):
        with UniversalOrderedStruct.use_old_contains():
            return OrderedDict.__repr__(self)

    def __getitem__(self, selector):
        if isinstance(selector, (list, tuple, slice, np.ndarray)):
            if isinstance(selector, slice):
                assert selector.step is None, "Can't handle stepped slices yet (from selector {})".format(selector)
                keys = [k for k in self.keys() if selector.start<=k and selector.stop is None or selector.stop<k]
            else:
                keys = selector
            return UniversalOrderedStruct((k, self[k]) for k in keys)
        else:
            return OrderedDict.__getitem__(self, selector)

    def __contains__(self, item):
        if UniversalOrderedStruct.OLD_CONTAINS_FLAG:
            return OrderedDict.__contains__(self, item)  # Look for a key, not a value... Need to do this to allow some OrderedDict methods to function.
        else:
            return item in self.values()

    def to_struct(self):
        return OrderedDict((k, v) for k, v in self.items())

    @classmethod
    def from_struct(cls, struct):
        return UniversalOrderedStruct(struct)


class ArrayStruct(UniversalCollection):
    """
    A dynamic data structure that makes it easy to store and query entries.

    It behaves similarly to a numpy array, but does not require that data be regularly sized.

            a = ArrayStruct()
            a['a', 'aa1'] = 1
            a['a', 'aa2'] = 2
            a['b', 0, 'bb1'] = 4
            a['b', 0, 'bb2'] = 5
            a['b', 1, 'bb1'] = 6
            a['b', 1, 'bb2'] = 7

            assert a['b', :, 'bb1'].values() == [4, 6]
            assert a['b', :, :].values() == [4, 6]

    """

    def __init__(self, initial_struct = None, recurse = False):

        if recurse:
            self._struct = UniversalCollection.from_struct(initial_struct).map(
                (lambda x: (ArrayStruct(x, recurse=recurse if recurse is True else recurse-1)
                    if isinstance(x, (list, tuple, OrderedDict, UniversalCollection)) else x)))
        elif isinstance(initial_struct, UniversalCollection):
            self._struct = initial_struct
        else:
            self._struct = UniversalCollection.from_struct(initial_struct)

    def __setitem__(self, selecting_key, value):

        if isinstance(selecting_key, tuple):
            if len(selecting_key)==0:
                self._struct = value._struct if isinstance(value, ArrayStruct) else value
                return
            elif isinstance(selecting_key[-1], slice) or selecting_key[-1] is Ellipsis:
                open_keys = [s==slice(None) or s is Ellipsis for s in selecting_key]
                first_open_key_index = open_keys.index(True, )
                assert all(open_keys[first_open_key_index:]), "You can only assign slices and elipses at the end!.  Got {}".fiormat(selecting_key)
                return self.__setitem__(selecting_key[:first_open_key_index], ArrayStruct(value, recurse=len(selecting_key)-first_open_key_index))
            elif selecting_key[-1] is Ellipsis:
                return self.__setitem__(selecting_key[:first_key], ArrayStruct(value, recurse=len(selecting_key)-first_key))
            else:
                first_key = selecting_key[0]
                subkeys = selecting_key[1:]
        else:
            first_key = selecting_key
            subkeys = ()

        assert not isinstance(first_key, slice), "Currently, sliced assignment is only supported at the end of the indices."

        try:
            if len(subkeys)==0:  # You assign directly to this object
                self._struct[first_key] = value
            else:  # This is just getting a container for an assignment
                assert not isinstance(first_key, slice), 'Sliced assignment currently not supported'
                if not self._struct.has_key(first_key):
                    self._struct[first_key] = ArrayStruct()
                self._struct[first_key][subkeys] = value
        except CollectionIsEmptyException:
            self._struct = UniversalCollection.from_first_key(first_key)
            self.__setitem__(selecting_key, value)

    def __getitem__(self, selectors):

        if selectors == ():
            return self
        first_selector, selectors = (selectors[0], selectors) if isinstance(selectors, tuple) else (selectors, (selectors, ))
        try:
            new_substruct = self._struct[first_selector]
            if len(selectors)==1:  # Case 1: Simple, just return substruct
                return new_substruct
            else:
                # assert isinstance(new_substruct, ArrayStruct), 'You are selecting with {}, which indicates a depth of {}, but the structure has no more depth.'.format(list(':' if s==slice(None) else s for s in selectors), len(selectors))
                if isinstance(first_selector, (list, tuple, np.ndarray, slice)):  # Sliced selection, with more sub-indices
                    return ArrayStruct(new_substruct.map(lambda x: x.__getitem__(selectors[1:])))
                else:  # Simple selection, with more sub-indices
                    return new_substruct[selectors[1:]]
        except CollectionIsEmptyException:
            self[first_selector] = UniversalCollection.from_first_key(first_selector)
            return self.__getitem__(selectors)

    def deepvalues(self):
        return [(v.deepvalues() if isinstance(v, ArrayStruct) else v.values() if isinstance(v, UniversalCollection) else v) for v in self._struct]

    def map(self, f):
        return ArrayStruct(self._struct.map(f))

    def to_struct(self):
        return self._struct.map(lambda x: x.to_struct() if isinstance(x, UniversalCollection) else x)

    @classmethod
    def from_struct(cls, struct):
        return cls(initial_struct=struct, recurse=True)

    def keys(self):
        return self._struct.keys()

    def __contains__(self, item):
        return self._struct.__contains__(item)

    def __len__(self):
        return self._struct.__len__()

    def __iter__(self):
        return self._struct.__iter__()

    def has_key(self, key):
        return self._struct.has_key()

    def values(self):
        return self._struct.values()

    def _first_element_keys(self):
        first_el = next(iter(self._struct))
        if isinstance(first_el, ArrayStruct):
            return (self.keys(), )+first_el._first_element_keys()
        else:
            return (self.keys(), )

    def to_array_and_keys(self):
        keys = self._first_element_keys()
        first = True
        key_indices = tuple(range(len(k)) for k in keys)
        for numeric_ix, key_indices in izip_equal(itertools.product(*key_indices), itertools.product(*keys)):
            if first:
                this_data = np.array(self[key_indices])
                arr = np.zeros(tuple(len(k) for k in keys)+this_data.shape, dtype=this_data.dtype)
                arr[numeric_ix] = this_data
                first = False
            else:
                arr[numeric_ix] = self[key_indices]
        return keys, arr

    def to_array(self):
        _, arr = self.to_array_and_keys()
        return arr

    def arrayify_axis(self, axis, ixs = (), inplace=False):
        b = ArrayStruct.from_struct(self.to_struct()) if not inplace else self
        subkeys = list(self[ixs]._first_element_keys())
        subkeys[axis-len(ixs)] = (slice(None), )
        assigning_keys = list(subkeys)
        del assigning_keys[axis-len(ixs)]
        substruct = ArrayStruct()
        for src_key, dest_key in izip_equal(itertools.product(*subkeys), itertools.product(*assigning_keys)):
            substruct[dest_key] = np.array(self[ixs+src_key])
        b[ixs] = substruct
        return b
