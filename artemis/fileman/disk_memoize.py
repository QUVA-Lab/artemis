import hashlib
from collections import OrderedDict
import logging
from fileman.local_dir import get_local_path, make_file_dir
import numpy as np
import pickle
import os

__author__ = 'peter'

MEMO_WRITE_ENABLED = True
MEMO_READ_ENABLED = True
MEMO_DIR = get_local_path('memoize_to_disk')


def memoize_to_disk(fcn, local_cache = False):
    """
    Save (memoize) computed results to disk, so that the same function, called with the
    same arguments, does not need to be recomputed.  This is useful if you have a long-running
    function that is often being given the same arguments.  Note: this does NOT check for the state
    of Global variables/time/whatever else the function may use, so you need to make sure your
    function is truly a function in that outputs only depend on inputs.  Otherwise, this will
    give you misleading results.
    e.g.
        @memoize_to_disk
        def fcn(a, b, c = None):
            results = ...
            return results

    You can also use this without the decorator.
    e.g.
        result = memoize_to_disk(fcn)(a, b, c=3)

    This is useful if:
    a) The decorator can/should not be visible from where the function is defined.
    b) You only want to memoize the function in one use-case, but not all.

    :param fcn: The function you're decorating
    :return: A wrapper around the function that checks for memos and loads old results if they exist.
    """

    cached_local_results = {}

    def check_memos(*args, **kwargs):
        result_computed = False

        if MEMO_READ_ENABLED:
            if local_cache:
                local_cache_signature = get_local_cache_signature(args, kwargs)
                if local_cache_signature in cached_local_results:
                    return cached_local_results[local_cache_signature]
            filepath = get_function_hash_filename(fcn, args, kwargs)
            if os.path.exists(filepath):
                with open(filepath) as f:
                    try:
                        result = pickle.load(f)
                    except ValueError as err:
                        logging.warn('Memo-file "%s" was corrupt.  (%s: %s).  Recomputing.' % (filepath, err.__class__.__name__, err.message))
                        result_computed = True
                        result = fcn(*args, **kwargs)
            else:
                result_computed = True
                result = fcn(*args, **kwargs)
        else:
            result_computed = True
            result = fcn(*args, **kwargs)

        if MEMO_WRITE_ENABLED:
            if local_cache:
                cached_local_results[local_cache_signature] = result
            if result_computed:  # Result was computed, so write it down
                filepath = get_function_hash_filename(fcn, args, kwargs)
                make_file_dir(filepath)
                with open(filepath, 'w') as f:
                    pickle.dump(result, f)

        return result

    check_memos.wrapped_fcn = fcn

    return check_memos


def memoize_to_disk_and_cache(fcn):
    """
    Memoize to disk AND keep a local cache (as you would with the @memoize decorator).
    """
    return memoize_to_disk(fcn, local_cache=True)


def get_local_cache_signature(args, kwargs):
    return args + ('5243643254_kwargs_start_here', ) + tuple((k, kwargs[k]) for k in sorted(kwargs.keys()))


def get_function_hash_filename(fcn, args, kwargs):
    args_code = compute_fixed_hash((args, kwargs))
    return os.path.join(MEMO_DIR, '%s-%s.pkl' % (fcn.__name__, args_code))


def get_all_memos():
    """
    :return: A list of file-locations
    """
    all_memos = os.listdir(MEMO_DIR) if os.path.exists(MEMO_DIR) else []
    full_paths_of_memos = [os.path.join(MEMO_DIR, m) for m in all_memos]
    return full_paths_of_memos


def get_memo_files_for_function(fcn):
    all_memos = os.listdir(MEMO_DIR) if os.path.exists(MEMO_DIR) else []
    matching_memos = [os.path.join(MEMO_DIR, m) for m in all_memos if m.startswith(fcn.wrapped_fcn.__name__)]
    return matching_memos


def clear_memo_files_for_function(fcn):
    memos = get_memo_files_for_function(fcn)
    for m in memos:
        os.remove(m)


def clear_all_memos():
    all_memos = get_all_memos()
    for m in all_memos:
        os.remove(m)
    print 'Removed %s memos.' % (len(all_memos))


def compute_fixed_hash(obj, hasher = None):
    """
    Given an object, return a hash that will always be the same (not just for the lifetime of the
    object, but for all future runs of the program too).
    :param obj: Some nested container of primitives
    :param hasher: (for internal use)
    :return:
    """

    if hasher is None:
        hasher = hashlib.md5()

    hasher.update(obj.__class__.__name__)

    if isinstance(obj, np.ndarray):
        hasher.update(pickle.dumps(obj.dtype))
        hasher.update(pickle.dumps(obj.shape))
        hasher.update(obj.tostring())
    elif isinstance(obj, (int, str, float, bool)) or (obj is None) or (obj in (int, str, float, bool)):
        hasher.update(pickle.dumps(obj))
    elif isinstance(obj, (list, tuple)):
        hasher.update(str(len(obj)))  # Necessary to distinguish ([a, b], c) from ([a, b, c])
        for el in obj:
            compute_fixed_hash(el, hasher=hasher)
    elif isinstance(obj, dict):
        hasher.update(str(len(obj)))  # Necessary to distinguish ([a, b], c) from ([a, b, c])
        keys = obj.keys() if isinstance(obj, OrderedDict) else sorted(obj.keys())
        for k in keys:
            compute_fixed_hash(k, hasher=hasher)
            compute_fixed_hash(obj[k], hasher=hasher)
    else:
        raise NotImplementedError("Don't have a method for hashing this %s" % (obj, ))

    return hasher.hexdigest()


class DisableMemoReading(object):

    def __enter__(self):
        global MEMO_READ_ENABLED
        self._old_read_state = MEMO_READ_ENABLED
        MEMO_READ_ENABLED = False

    def __exit__(self, *args):
        global MEMO_READ_ENABLED
        MEMO_READ_ENABLED = self._old_read_state


class DisableMemoWriting(object):

    def __enter__(self):
        global MEMO_WRITE_ENABLED
        self._old_write_state = MEMO_WRITE_ENABLED
        MEMO_WRITE_ENABLED = False

    def __exit__(self, *args):
        global MEMO_WRITE_ENABLED
        MEMO_WRITE_ENABLED = self._old_write_state


class DisableMemos(object):
    """
    You can disable memoization with a with.

    with DisableMemos():
        # call memoized function, but disable reading/writing.
        my_memoized_function()
    """

    def __enter__(self):
        self._reader = DisableMemoReading()
        self._writer = DisableMemoWriting()
        self._reader.__enter__()
        self._writer.__enter__()

    def __exit__(self, *args):
        self._reader.__exit__(*args)
        self._writer.__exit__(*args)


if __name__ == '__main__':

    cmd = raw_input('Type "clearall" to clear all memos: ')

    if cmd == 'clearall':
        clear_all_memos()
    else:
        raise Exception('Bad command or file name.')