import itertools
import os.path
from datetime import datetime
from functools import partial
from typing import Optional, Iterable, Callable, TypeVar, Tuple, Sequence, Iterator
import inspect
import time

def get_datetime_filename(prefix = '', suffix = '', when: Optional[datetime] = None, extension: Optional[str] = None, include_microseconds: bool = True) -> str:

    if when is None:
        when = datetime.now()
    return prefix \
           + when.strftime("%Y%m%d-%H%M%S"+('-%p' if include_microseconds else '')) \
           + suffix \
           + ('.'+extension if extension is not None else '')


def get_context_name(levels_up=1):
    context = inspect.stack()[levels_up]
    function_name = context.function
    if function_name == '<module>':
        _, filename = os.path.split(context.filename)
        return filename
    else:
        if 'self' in context.frame.f_locals:
            return f"{context.frame.f_locals['self'].__class__.__name__}.{function_name}"
        else:
            return function_name


def ensure_path(path: str) -> str:

    path = os.path.expanduser(path)

    parent, _ = os.path.split(path)
    try:
        os.makedirs(parent)
    except OSError:
        pass
    return path


def demo_get_context_name():
    print(f'Name of this context is "{get_context_name()}"')


def iter_max_rate(max_fps: Optional[float]) -> Iterable[float]:


    min_period = 1./max_fps if max_fps is not None else 0
    t_last = - float('inf')
    while True:
        t_current = time.time()
        yield t_current - t_last
        sleep_time = t_current + min_period - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)


CallReturnType = TypeVar('CallReturnType')


def timed_call(func: Callable[[], CallReturnType]) -> Tuple[float, CallReturnType]:
    start_time = time.monotonic_ns()
    result = func()
    elapsed = (time.monotonic_ns()-start_time)/1e9
    return elapsed, result


ItemType = TypeVar('ItemType')
ArgType = TypeVar('ArgType')
ResultType = TypeVar('ResultType')


def tee_and_specialize_iterator(
        iterator: Iterable[ItemType],
        specialization_func: Callable[[ItemType, ArgType], ResultType],
        args: Sequence[ArgType]
    ) -> Sequence[Iterator[ResultType]]:

    def make_sub_iterator(it_copy, arg):
        for it in it_copy:
            yield specialization_func(it, arg)

    return [make_sub_iterator(it_copy, arg) for it_copy, arg in zip(itertools.tee(iterator, len(args)), args)]


def byte_size_to_string(bytes: int, decimals_precision: int = 1) -> str:

    size = bytes
    prefix = ''
    for this_prefix in 'kMGPE':
        if size > 1024:
            prefix = this_prefix
            size = size / 1024
        else:
            break

    return f"{{:.{decimals_precision}f}} {prefix}B".format(size)


if __name__ == '__main__':
    demo_get_context_name()
