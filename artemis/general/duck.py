from collections import OrderedDict
import itertools
import numpy as np
from abc import ABCMeta, abstractmethod
from artemis.general.display import arraystr, indent_string
from artemis.general.should_be_builtins import izip_equal
import sys


class CollectionIsEmptyException(Exception):
    pass


class InvalidKeyError(KeyError):
    pass


class InvalidInitializerError(Exception):
    pass


class InitializerTooShallowError(Exception):
    pass


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
        elif isinstance(struct, dict):
            return UniversalOrderedStruct(struct)
        elif struct is None or isinstance(struct, EmptyCollection):
            return EmptyCollection()
        else:
            raise InvalidInitializerError("Don't know how to load a Universal Collection from initializer: {}".format(struct))

    def map(self, f):
        new_obj = self.__class__()
        for k, v in self.items():
            new_obj[k] = f(v)
        return new_obj


class EmptyCollection(UniversalCollection):

    def has_key(self, key):
        return False

    def __getitem__(self, ix):
        if isinstance(ix, slice):
            return []
        else:
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


class DynamicSequence(list, UniversalCollection):
    """
    This mostly behaves like a list, except
    - It's also extended to have the key, value methods of an OrderedDict.
    - It also supports numpy-like indexing with a list of integers.
    - It allows you to assign with the built-in "next" as an index.  Eg "a[next] = 3" which means
      "append 3 to the list"
    """

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
        elif isinstance(ix, int):
            if ix==len(self):
                self.append(val)
            else:
                try:
                    list.__setitem__(self, ix, val)
                except IndexError:
                    raise InvalidKeyError('If you assign to a DynamicSequence, the index must be no greater than the length of the sequence.  Got index {} for length {}'.format(ix, len(self)))
        else:
            raise InvalidKeyError('Cannot index a DynamicSequence with non-integer index: {}'.format(ix))

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
    """
    Mostly like an OrderedDict, but it behaves like a list, in that (for x in struct) and (x in struct) looks over
    values, not keys.
    """

    def __init__(self, *initializer):
        if len(initializer)==1 and type(initializer[0]) is dict:
            d = initializer[0]
            initializer = [OrderedDict((k, d[k]) for k in sorted(d.keys()))]

        self._heart = OrderedDict(*initializer)

    def __contains__(self, item):
        return item in self._heart.values()

    def __setitem__(self, key, value):
        if key is next or isinstance(key, int):
            raise InvalidKeyError('This sequence is an {}, and cannot be given {} key: {}'.format(self.__class__.__name__, key.__class__.__name__, key))
        self._heart.__setitem__(key, value)

    def __getitem__(self, selector):
        if isinstance(selector, (list, slice, np.ndarray)):
            if isinstance(selector, slice):
                all_keys = list(self.keys())
                start_index = all_keys.index(selector.start) if selector.start is not None else None
                stop_index = all_keys.index(selector.stop) if selector.stop is not None else None
                keys = all_keys[start_index:stop_index:selector.step]
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
        return key in self._heart

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
    a list.  It behaves similarly to a numpy array, but does not require that data be regularly sized.

        a = Duck()
        a['a', 'aa1'] = 1
        a['a', 'aa2'] = 2
        a['b', 0, 'subfield1'] = 4
        a['b', 0, 'subfield2'] = 5
        a['b', 1, 'subfield1'] = 6
        a['b', 1, 'subfield2'] = 7

        assert list(a['b', 1, :]) == [6, 7]
        assert a['b', :, 'subfield1'] == [4, 6]

    """
    def __init__(self, initial_struct = None, recurse = False):
        if recurse is True:  # How many layers deep to recurse.  0=just the initial struct, 1=break into initial_struct_keys, ...
            self._struct = UniversalCollection.from_struct(initial_struct).map(
                (lambda x: (Duck(x, recurse=recurse if recurse is True else recurse - 1)
                    if isinstance(x, (dict, list, tuple, UniversalCollection, type(None))) else x)))
        elif isinstance(recurse, int) and recurse>0:  # How many layers deep to recurse.  0=just the initial struct, 1=break into initial_struct_keys, ...
            try:
                self._struct = UniversalCollection.from_struct(initial_struct).map((lambda x: (Duck(x, recurse=recurse - 1))))
            except InitializerTooShallowError:
                raise InitializerTooShallowError('Tried to initialize a Duck by breaking {} levels into an object, but the not all branches of go {} levels deep.  The object: {}'.format(recurse+1, recurse+1, initial_struct))
        elif isinstance(initial_struct, Duck):
            raise Exception('fdsf')
            # self._struct = DictArrayList.from_struct()
        elif isinstance(initial_struct, UniversalCollection):
            self._struct = initial_struct
        else:
            try:
                self._struct = UniversalCollection.from_struct(initial_struct)
            except InvalidInitializerError:
                raise InitializerTooShallowError('Initializer {} has no depth, and so cannot be used to initialize a Duct.'.format(initial_struct))

    def __setitem__(self, indices, value):
        """
        :param indices: A tuple of indices (as you would use to access an element in an array)
        :param value: An arbitrary object to place at those indices.  If it is another Duck, it will be considered a child node of this Duck.
        """
        key_chain = indices if isinstance(indices, tuple) else (indices,)
        if len(key_chain)==0:  # e.g.  a[()]=4   Typically this happens when assigning to a slice.  ie a[:]=[1,2,3] becomes a[()]=Duck([1,2,3]).  Similar syntax in numpy
            self._struct = value._struct if isinstance(value, Duck) else value  # Possibly weird edge cases could arise here...
        elif any(isinstance(k, slice) or k is Ellipsis for k in key_chain):  # When training elements are slices, ellipses.
            open_keys = [s==slice(None) or s is Ellipsis for s in key_chain]
            first_open_key_index = open_keys.index(True, )
            if not all(open_keys[first_open_key_index:]):
                raise InvalidKeyError("Currently, sliced assignment is only supported for the final indices.  Got {}".format(key_chain))
            self.__setitem__(key_chain[:first_open_key_index], Duck(value, recurse=True if key_chain[-1] is Ellipsis else len(key_chain) - first_open_key_index - 1))
        else:  # The common case: an unsliced assignment
            first_key, subkeys = key_chain[0], key_chain[1:]
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
                self.__setitem__(indices, value)

    def __getitem__(self, indices):
        """
        :param indices: A tuple of indices (as you would use to access an element in an array)
        :return: Either
            a) A New Duck (if the selectors specifiy a range of elements in the current Duck
            b) An element at a leaf node (if the selectors specify a partucular item)
        """
        if indices == ():
            return self
        first_selector, indices = (indices[0], indices) if isinstance(indices, tuple) else (indices, (indices,))

        if self._struct is None:
            raise Exception('No value has ever been assigned to this Duck, but you are trying to extract index {} from it.'.format(indices))

        try:
            new_substruct = self._struct[first_selector]
            if isinstance(new_substruct, UniversalCollection) and not isinstance(new_substruct, Duck):  # This will happen if the selector is a slice or something...
                new_substruct = Duck(new_substruct, recurse=False)
            if len(indices)==1:  # Case 1: Simple... this is the last selector, so we can just return it.
                return new_substruct
            else:  # Case 2:
                # assert isinstance(new_substruct, ArrayStruct), 'You are selecting with {}, which indicates a depth of {}, but the structure has no more depth.'.format(list(':' if s==slice(None) else s for s in selectors), len(selectors))
                if isinstance(first_selector, (list, np.ndarray, slice)):  # Sliced selection, with more sub-indices
                    return new_substruct.map(lambda x: x.__getitem__(indices[1:]))
                else:  # Simple selection, with more sub-indices
                    return new_substruct[indices[1:]]
        except CollectionIsEmptyException:
            raise Exception('No value has ever been assigned to this Duck, but you are trying to extract index {} from it.'.format(indices))

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
        """
        Turn the object into a numpy array.  This will fail unless all sub-branches have the same structure (ie
        the object is non-ragged)
        :return: A numpy array
        """
        _, arr = self.to_array_and_keys()
        return arr

    def arrayify_axis(self, axis, subkeys = (), inplace=False):
        """
        Take a particular axis (depth) of this object, concatenate it into an array for each leaf node.  This requires
        that all values along this axis have the same structure.  Best understood by the following example:

            a = Duck()
            a[0, 'x'] = 1
            a[0, 'y'] = 2
            a[1, 'x'] = 3
            a[1, 'y'] = 4
            b = a.arrayify_axis(axis=0)
            assert np.array_equal(b['x'], [1, 3])
            assert np.array_equal(b['y'], [2, 4])

        :param axis: The depth at which you'd like to concatenate.
        :param subkeys: A sub-branch of struct to do this along (ie. only do it on branch ('subkey', 'subsubkey'))
        :param inplace: Do the operation in-place (ie change this object (saves you from copying a new object))
        :return: A new Duck, with the given axis shifted to the end.
        """
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
        """
        We define equality as both keys and values matching for the full depth.
        :param other: Another Duck, or nested object.
        :return: True if equal, False if not.
        """
        if len(self) != len(other):
            return False
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

    def __str__(self, max_key_len=4):
        keys = list(self.keys())
        if len(keys)>max_key_len:
            inner_description = [str(k) for k in keys[:max_key_len-1]]+['...']
        else:
            inner_description = [str(k) for k in keys]
        return '<{} with {} keys: {}>'.format(self.__class__.__name__, len(keys), inner_description)

    def description(self, max_expansion=4, _skip_intro=False):
        """
        :param max_expansion: Maximum amount to break into structures
        :param _skip_intro: For internal use in recursion
        :return: A string describing the structure of this Duck.
        """
        full_string = '' if _skip_intro else (str(self) + '')
        key_value_string = ''
        for i, (k, v) in enumerate(self._struct.items()):
            if i>max_expansion:
                key_value_string += '\n(... Omitting {} of {} elements ...)'.format(len(self._struct)-max_expansion, len(self._struct))
            else:
                if isinstance(v, Duck):
                    value_string = v.description(max_expansion=max_expansion, _skip_intro=True)
                elif isinstance(v, np.ndarray):
                    value_string = arraystr(v)
                else:
                    value_string = str(v)
                key_value_string += '\n{}: {}'.format(str(k), value_string)
            full_string += key_value_string

            if i>max_expansion:
                break
        return ('' if _skip_intro else (str(self) + '')) + indent_string(key_value_string, indent='| ', include_first=False)
