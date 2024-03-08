import dataclasses
import sys
# from _typeshed import SupportsNext
from collections import deque
from dataclasses import dataclass, field, is_dataclass
from typing import Optional, TypeVar, Generic, Deque, Tuple, Any, Iterator
import numpy as np
ItemType = TypeVar('ItemType')


class OutOfBufferException(Exception):
    """ Raised when you request something outside the bounds of the buffer """


def get_memory_footprint(item: Any) -> int:

    if is_dataclass(item):
        return sum(get_memory_footprint(v) for v in dataclasses.asdict(item).values())
    elif isinstance(item, np.ndarray):
        return item.itemsize * item.size
    elif isinstance(item, (list, tuple, set)):
        return sum(get_memory_footprint(v) for v in item)
    elif isinstance(item, dict):
        return sum(get_memory_footprint(v) for v in item.values())
    else:
        return sys.getsizeof(item)  # Only returns pointer-size, no recursion


@dataclass
class SequenceBuffer(Generic[ItemType]):
    max_elements: Optional[int] = None
    max_memory: Optional[int] = None

    _current_index: int = 0
    _current_memory: int = 0
    _buffer: Deque[ItemType] = None

    def __post_init__(self):
        self._buffer = deque(maxlen=self.max_elements)

    def append(self, item: ItemType):
        self._buffer.append(item)
        self._current_index += 1
        if self.max_memory is not None:
            self._current_memory += get_memory_footprint(item)
            while self._current_memory > self.max_memory:
                popped_item = self._buffer.popleft()
                self._current_memory -= get_memory_footprint(popped_item)

    def get_index_bounds(self) -> Tuple[int, int]:

        return (self._current_index-len(self._buffer), self._current_index)

    def lookup(self, index: int, jump_to_edge: bool = False, new_data_source: Optional[Iterator[ItemType]] = None) -> Tuple[int, ItemType]:

        buffer_index = index - self._current_index + len(self._buffer) if index>=0 else len(self._buffer)+index

        if buffer_index >= len(self._buffer) and new_data_source is not None:
            for _ in range(buffer_index-len(self._buffer)+1):
                try:
                    self.append(next(new_data_source))
                except StopIteration:
                    if not jump_to_edge:
                        raise OutOfBufferException(f"Data source exhausted while trying to retrieve index {index}")
            return self.lookup(index, jump_to_edge=jump_to_edge)

        if jump_to_edge:
            buffer_index = max(0, min(len(self._buffer)-1, buffer_index))
        else:
            if buffer_index < 0:
                raise OutOfBufferException(f"Index {index} falls out of bounds of the buffer, which only remembers back to {self._current_index - len(self._buffer)}")
            elif buffer_index >= len(self._buffer):
                raise OutOfBufferException(f"Index {index} has not yet been assigned to buffer, which has only been filled up to index {self._current_index}")

        remapped_index = buffer_index + self._current_index - len(self._buffer)
        return remapped_index, self._buffer[buffer_index]





