from artemis.fileman.local_dir import get_artemis_data_path
import os
import pickle
import numpy as np
from six.moves import input
from artemis.general.hashing import fixed_hash_eq


def assert_variable_matches(var, var_name, raise_error = False):
    """
    This function allows you to check that a variable is equal between different runs of your program.  It is useful for
    rooting out unexpected randomness in your code.

    For example, suppose you have a variable my_var that for some reason is not identical between runs - but its value
    diverges only after some number of iterations.

    In your loop

        while True:

            ....
            assert_variable_matches(my_var, 'my_var')

    :param var: A variable value (preferably a python primative or numpy array)
    :param var_name: A name to identify your variable
    """

    if var_name not in _var_write_status:
        _var_write_status[var_name] = not os.path.exists(_get_tracker_file(var_name))

    if _var_write_status[var_name]:
        _write_next_var_value(var, var_name)
    else:
        if var_name not in _var_calls:
            _var_calls[var_name] = 0
        _var_calls[var_name] += 1
        loaded_var = _get_next_saved_var_value(var_name)
        if not fixed_hash_eq(var, loaded_var):
            message = '{} @ {} does not match saved value: \n  {}\n  {}'.format(var_name, _var_calls[var_name], str(var).replace('\n', '\\n'), str(loaded_var).replace('\n', '\\n'))
            if raise_error:
                raise AssertionError(message)
            else:
                print(message)

def _get_random_finder_path(relative_path = ''):
    return get_artemis_data_path(os.path.join('random_finder', relative_path), make_local_dir=True)


def _get_tracker_file(var_name):
    return _get_random_finder_path('{}.pkl'.format(var_name))


def _compare(obj1, obj2):
    assert type(obj1) is type(obj2), "Types don't match: {}, {}".format(obj1, obj2)
    if isinstance(obj1, np.ndarray):
        assert np.array_equal(obj1, obj2), "Arrays not equal: {}, {}".format(obj1, obj2)
    else:
        assert obj1 == obj2

_var_readers = {}

_var_write_status = {}

_var_calls = {}


def _get_var_reader(var_name):

    with open(_get_tracker_file(var_name), 'rb') as f:
        while True:
            yield pickle.load(f)


def _get_next_saved_var_value(var_name):

    if var_name not in _var_readers:
        print('Found log of variable {}.  Comparing...'.format(var_name))
        _var_readers[var_name] = _get_var_reader(var_name)

    return next(_var_readers[var_name])


def _write_next_var_value(var, var_name):

    filename = _get_tracker_file(var_name)

    if not os.path.exists(filename):
        print('Creating log of variable {}'.format(var_name))

    with open(filename, 'ab') as f:
        pickle.dump(var, f)


def delete_vars(vars):
    if isinstance(vars, str):
        vars = [vars]
    for var in vars:
        os.remove(os.path.join(path, var+'.pkl'))


if __name__ == '__main__':

    path = _get_random_finder_path()
    all_saved_vars = [os.path.splitext(var)[0] for var in os.listdir(path)]
    response = input('{}\nType delete <varname> to delete records for a varname, or delete --all to delete all. >>'.format('\n'.join(all_saved_vars) if len(all_saved_vars)>0 else '<No vars>'))
    command, var = response.split(' ')
    if command=='delete':
        vars = all_saved_vars if var == '--all' else [var]
        delete_vars(vars)
