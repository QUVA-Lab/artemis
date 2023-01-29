from collections import OrderedDict
from typing import Optional, TypeVar, Generic, Hashable

from more_itertools import first

from artemis.general.sequence_buffer import get_memory_footprint

KeyType = TypeVar('KeyType')
ItemType = TypeVar('ItemType')


class CacheDict(Generic[KeyType, ItemType]):
    """ A simple buffer that just keeps a cache of recent entries

    Example
        cache = ItemCache(buffer_len=3)
        cache[1]='aaa'
        cache[5]='bbb'
        cache[7]='ccc'

    """

    def __init__(self, buffer_length: Optional[int] = None, buffer_size_bytes: Optional[int] = None, calculate_size_once = True, always_allow_one_item: bool = False):

        self._buffer = OrderedDict()
        self._buffer_length = buffer_length
        self.buffer_size_bytes = buffer_size_bytes
        self._first_object_size: Optional[int] = None
        self._calculate_size_once = calculate_size_once
        self._current_buffer_size = 0
        self._always_allow_one_item = always_allow_one_item

    def _remove_oldest_item(self):
        if len(self._buffer) > 0:
            first_key = first(self._buffer.keys())
            value, itemsize = self._buffer[first_key]
            del self._buffer[first(self._buffer.keys())]
            if itemsize is not None:
                self._current_buffer_size -= itemsize

    def __setitem__(self, key: Hashable, value: ItemType) -> None:
        size = None
        if self._buffer_length is not None and len(self._buffer) == self._buffer_length:
            self._remove_oldest_item()

        # print(f'Buffer lendgth: {len(self._buffer)} Buffer size: {self._current_buffer_size}')
        if self.buffer_size_bytes is not None:
            size = get_memory_footprint(value) if not self._calculate_size_once or self._first_object_size is None else self._first_object_size
            # print(f'Memory footprint for {value.__class__} is {size} bytes')
            while len(self._buffer)>0 and self._current_buffer_size+size > self.buffer_size_bytes:
                self._remove_oldest_item()

            self._current_buffer_size += size

            if size < self.buffer_size_bytes or len(self._buffer)==0 and self._always_allow_one_item:
                self._buffer[key] = value, size
        else:
            self._buffer[key] = value, size

    def __getitem__(self, key: Hashable) -> ItemType:
        if key in self._buffer:
            value, _ = self._buffer[key]
            return value
        else:
            raise KeyError(f"{key} is not in cache")

    def __contains__(self, key: Hashable):
        return key in self._buffer
