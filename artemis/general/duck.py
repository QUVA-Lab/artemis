from collections import OrderedDict
import itertools
import numpy as np
from abc import ABCMeta, abstractmethod
from artemis.general.should_be_builtins import izip_equal
import sys

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
        elif struct is None or isinstance(struct, EmptyCollection):
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
        return iter([])


class InvalidKeyError(KeyError):
    pass


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
            try:
                return list.__getitem__(self, ix)
            except TypeError:
                raise InvalidKeyError('You tried getting index "{}" from a {}, but {} object can only be indexed by ints, slices, numeric lists'.format(ix, type(self).__name__, type(self).__name__))

    def __setitem__(self, ix, val):
        if ix is next:
            self.append(val)
        elif ix>=len(self):
            assert ix==len(self), 'If you assign to a DynamicSequence, the index must be no greater than the length of the sequence.  Got index {} for length {}'.format(ix, len(self))
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


class UniversalOrderedStruct(UniversalCollection):

    def __init__(self, *initializer):
        self._heart = OrderedDict(*initializer)

    def __contains__(self, item):
        return item in self._heart.values()

    def __setitem__(self, key, value):
        self._heart.__setitem__(key, value)

    def __getitem__(self, selector):
        if isinstance(selector, (list, slice, np.ndarray)):
            if isinstance(selector, slice):
                assert selector.step is None, "Can't handle stepped slices yet (from selector {})".format(selector)
                keys = [k for k in self._heart.keys() if selector.start<=k and selector.stop is None or selector.stop<k]
            else:
                keys = selector
            return UniversalOrderedStruct((k, self[k]) for k in keys)
        else:
            return self._heart.__getitem__(selector)

    def __repr__(self):
        rep = self._heart.__repr__()
        return self.__class__.__name__ + rep[len(OrderedDict.__class__.__name__):]

    def __iter__(self):
        return iter(self._heart.values())

    def __len__(self):
        return self._heart.__len__()

    def has_key(self, key):
        return self._heart.has_key(key)

    def keys(self):
        return self._heart.keys()

    def values(self):
        return self._heart.values()

    def to_struct(self):
        return self._heart.copy()

    @classmethod
    def from_struct(cls, struct):
        return cls(struct)


class Duck(UniversalCollection):
    """
    A dynamic data structure that makes it easy to store and query entries.  It's a cross between a dict, an array, and
    a list.

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
                (lambda x: (Duck(x, recurse=recurse if recurse is True else recurse - 1)
                    if isinstance(x, (list, tuple, OrderedDict, UniversalCollection, type(None))) else x)))
        elif isinstance(initial_struct, Duck):
            raise Exception('fdsf')
            # self._struct = DictArrayList.from_struct()
        elif isinstance(initial_struct, UniversalCollection):
            self._struct = initial_struct
        else:
            self._struct = UniversalCollection.from_struct(initial_struct)

    def __setitem__(self, selecting_key, value):

        if isinstance(selecting_key, tuple):
            if len(selecting_key)==0:
                self._struct = value._struct if isinstance(value, Duck) else value
                return
            elif isinstance(selecting_key[-1], slice) or selecting_key[-1] is Ellipsis:
                open_keys = [s==slice(None) or s is Ellipsis for s in selecting_key]
                first_open_key_index = open_keys.index(True, )
                assert all(open_keys[first_open_key_index:]), "You can only assign slices and elipses at the end!.  Got {}".fiormat(selecting_key)
                return self.__setitem__(selecting_key[:first_open_key_index], Duck(value, recurse=True if selecting_key[-1] is Ellipsis else len(selecting_key) - first_open_key_index))
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
                    self._struct[first_key] = Duck()
                self._struct[first_key][subkeys] = value
        except CollectionIsEmptyException:
            self._struct = UniversalCollection.from_first_key(first_key)
            self.__setitem__(selecting_key, value)

    def __getitem__(self, selectors):
        """
        :param selectors: A tuple of indices (as you would use to access an element in an array)
        :return: Either
            a) A New Duck (if the selectors specifiy a range of elements in the current Duck
            b) An element at a leaf node (if the selectors specify a partucular item)
        """
        if selectors == ():
            return self
        first_selector, selectors = (selectors[0], selectors) if isinstance(selectors, tuple) else (selectors, (selectors, ))
        try:
            new_substruct = self._struct[first_selector]
            if isinstance(new_substruct, UniversalCollection) and not isinstance(new_substruct, Duck):  # This will happen if the selector is a slice or something...
                new_substruct = Duck(new_substruct, recurse=False)
            if len(selectors)==1:  # Case 1: Simple... this is the last selector, so we can just return it.
                return new_substruct
            else:  # Case 2:
                # assert isinstance(new_substruct, ArrayStruct), 'You are selecting with {}, which indicates a depth of {}, but the structure has no more depth.'.format(list(':' if s==slice(None) else s for s in selectors), len(selectors))
                if isinstance(first_selector, (list, np.ndarray, slice)):  # Sliced selection, with more sub-indices
                    return new_substruct.map(lambda x: x.__getitem__(selectors[1:]))
                else:  # Simple selection, with more sub-indices
                    return new_substruct[selectors[1:]]
        except CollectionIsEmptyException:
            if first_selector==slice(None):
                return EmptyCollection()
            else:
                self[(first_selector, )] = UniversalCollection.from_first_key(first_selector)
                return self.__getitem__(selectors)

    def deepvalues(self):
        return [(v.deepvalues() if isinstance(v, Duck) else v.values() if isinstance(v, UniversalCollection) else v) for v in self._struct]

    def map(self, f):
        return Duck(self._struct.map(f))

    def _first_element_keys(self):

        these_keys = list(self.keys())

        try:
            first_el = next(iter(self._struct))  # Try to get the keys of the first element
        except StopIteration:
            return (these_keys, )  # If there is no first element, there're no keys to get
            # raise Exception("Can't get keys from struct {}".format(self._struct))
        if isinstance(first_el, Duck):
            return (these_keys, )+first_el._first_element_keys()
        else:
            return (these_keys, )

    def to_array_and_keys(self):
        keys = self._first_element_keys()
        first = True
        key_indices = tuple(range(len(k)) for k in keys)
        for numeric_ix, key_ix in izip_equal(itertools.product(*key_indices), itertools.product(*keys)):
            if first:
                this_data = np.array(self[key_ix])
                arr = np.zeros(tuple(len(k) for k in keys)+this_data.shape, dtype=this_data.dtype)
                arr[numeric_ix] = this_data
                first = False
            else:
                arr[numeric_ix] = self[key_ix]
        return keys, arr

    def to_array(self):
        _, arr = self.to_array_and_keys()
        return arr

    def arrayify_axis(self, axis, subkeys = (), inplace=False):
        b = Duck.from_struct(self.to_struct()) if not inplace else self
        if not isinstance(subkeys, tuple):
            subkeys = (subkeys,)
        indexing_keys = list(self[subkeys]._first_element_keys())
        indexing_keys[axis - len(subkeys)] = (slice(None),)
        indexing_keys = tuple(indexing_keys)
        assigning_keys = list(indexing_keys)
        del assigning_keys[axis - len(subkeys)]
        if len(assigning_keys)==0:
            assert len(indexing_keys)==1
            substruct = np.array(self[subkeys+indexing_keys[0]])
        else:
            substruct = Duck()
            for src_key, dest_key in izip_equal(itertools.product(*indexing_keys), itertools.product(*assigning_keys)):
                substruct[dest_key] = np.array(self[subkeys + src_key])
        b[subkeys] = substruct
        return b

    def to_struct(self):
        return self._struct.map(lambda x: x.to_struct() if isinstance(x, UniversalCollection) else x).to_struct()

    @classmethod
    def from_struct(cls, struct):
        return cls(initial_struct=struct, recurse=True)

    def __contains__(self, item):
        return self._struct.__contains__(item)

    def __len__(self):
        return self._struct.__len__()

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        # assert isinstance(other, self.__class__)
        if not len(self)==len(other):
            return False
        if not isinstance(other, (UniversalCollection, list, OrderedDict, np.ndarray)):
            return False
        for (our_key, our_value) in self.items():
            if self[our_key] != other[our_key]:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __iter__(self):
        return self._struct.__iter__()

    def open(self, *ixs):
        self[ixs] = Duck()
        ixs = tuple(-1 if ix is next else ix for ix in ixs)
        return self[ixs]

    def has_key(self, *key_chain):
        return self._struct.has_key()

    def keys(self, depth=None):
        if depth is None:
            for k in self._struct.keys():
                yield k
        elif depth=='full' or depth>0:
            if isinstance(depth, str):
                assert depth=='full'
                depth = sys.maxsize
            for k, v in self._struct.items():
                if isinstance(v, Duck) and depth>1:
                    for subkey in v.keys(depth=depth-1):
                        yield (k, )+subkey
                else:
                    yield (k, )
        else:
            raise Exception('"depth" can be "full", None, or an integer >= 1.  Got {}'.format(depth))

    def values(self, depth=None):
        for k in self.keys(depth=depth):
            yield self[k]

    def items(self, depth=None):
        for k in self.keys(depth=depth):
            yield k, self[k]
