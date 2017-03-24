import logging
import os
from functools import partial
from artemis.fileman.local_dir import get_local_path, make_file_dir
from artemis.general.functional import infer_arg_values
from artemis.general.hashing import compute_fixed_hash
from artemis.general.test_mode import is_test_mode

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


__author__ = 'peter'

MEMO_WRITE_ENABLED = True
MEMO_READ_ENABLED = True
MEMO_DIR = get_local_path('memoize_to_disk')


def memoize_to_disk(fcn, local_cache = False, disable_on_tests=True, use_cpickle = False):
    """
    Save (memoize) computed results to disk, so that the same function, called with the
    same arguments, does not need to be recomputed.  This is useful if you have a long-running
    function that is often being given the same arguments.  Note: this does NOT check for the state
    of Global variables/time/whatever else the function may use, so you need to make sure your
    function is truly a function in that outputs only depend on inputs.  Otherwise, this will
    give you misleading results.

    Usage:
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
    :param local_cache: Keep a cache in python (so you don't need to go to disk if you call again in the same process)
    :param disable_on_tests: Persistent memos can really screw up tests, so disable memos when is_test_mode() returns
        True.  Generally, leave this as true, unless you are testing memoization itself.
    :param use_cpickle: Use CPickle, instead of pickle, to save results.  This can be faster for complex python
        structures, but can be slower for numpy arrays.  So we recommend not using it.
    :return: A wrapper around the function that checks for memos and loads old results if they exist.
    """

    if use_cpickle:
        import cPickle as pickle
    else:
        import pickle

    cached_local_results = {}

    def check_memos(*args, **kwargs):

        if disable_on_tests and is_test_mode():
            return fcn(*args, **kwargs)

        result_computed = False
        full_args = infer_arg_values(fcn, *args, **kwargs)
        filepath = get_function_hash_filename(fcn, full_args)
        # The filepath is used as the unique identifier, for both the local path and the disk-path
        # It may be more efficient to use the built-in hashability of certain types for the local cash, and just have special
        # ways of dealing with non-hashables like lists and numpy arrays - it's a bit dangerous because we need to check
        # that no object or subobjects have been changed.
        if MEMO_READ_ENABLED:
            if local_cache:
                # local_cache_signature = get_local_cache_signature(args, kwargs)
                if filepath in cached_local_results:
                    LOGGER.info('Reading disk-memo from local cache for function %s' % (fcn.__name__, ))
                    return cached_local_results[filepath]
            if os.path.exists(filepath):
                with open(filepath) as f:
                    try:
                        LOGGER.info('Reading memo for function %s' % (fcn.__name__, ))
                        result = pickle.load(f)
                    except (ValueError, ImportError) as err:
                        if isinstance(err, ValueError):
                            LOGGER.warn('Memo-file "%s" was corrupt.  (%s: %s).  Recomputing.' % (filepath, err.__class__.__name__, str(err)))
                        elif isinstance(err, ImportError):
                            LOGGER.warn('Memo-file "{}" was tried to reference an old class and got ImportError: {}.  Recomputing.'.format(filepath, str(err)))
                        result_computed = True
                        result = fcn(*args, **kwargs)
            else:
                result_computed = True
                result = fcn(*args, **kwargs)
        else:
            result_computed = True
            result = fcn(*args, **kwargs)

        if MEMO_WRITE_ENABLED and result is not None:  # We assume result of None means you haven't done coding your function.
            if local_cache:
                cached_local_results[filepath] = result
            if result_computed:  # Result was computed, so write it down
                filepath = get_function_hash_filename(fcn, full_args)
                make_file_dir(filepath)
                with open(filepath, 'w') as f:
                    LOGGER.info('Writing disk-memo for function %s' % (fcn.__name__, ))
                    pickle.dump(result, f, protocol=2)

        return result

    check_memos.wrapped_fcn = fcn
    check_memos.clear_cache = lambda: clear_memo_files_for_function(check_memos)

    return check_memos


def memoize_to_disk_test(fcn):
    """
    Use this just when testing the memoization itself (because normally memoization is disabled when is_test_mode() is True.
    :param fcn:
    :return:
    """
    return memoize_to_disk(fcn, disable_on_tests=False)


def memoize_to_disk_and_cache(fcn):
    """
    Memoize to disk AND keep a local cache (as you would with the @memoize decorator).
    """
    return memoize_to_disk(fcn, local_cache=True)


def memoize_to_disk_and_cache_test(fcn):
    return memoize_to_disk(fcn, local_cache=True, disable_on_tests=False)


def get_function_hash_filename(fcn, argname_argvalue_list):
    args_code = compute_fixed_hash(argname_argvalue_list)
    # TODO: Include function path in hash?  Or module path, which would allow memos to be shareable.
    return os.path.join(MEMO_DIR, '%s-%s.pkl' % (fcn.__name__, args_code))


def memoize_to_disk_with_settings(**kwargs):
    return partial(memoize_to_disk, **kwargs)


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