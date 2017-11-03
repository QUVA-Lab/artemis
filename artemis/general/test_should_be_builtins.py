from collections import OrderedDict

import pytest

from artemis.general.should_be_builtins import itermap, reducemap, separate_common_items, remove_duplicates, \
    detect_duplicates, remove_common_prefix, all_equal, get_absolute_module, insert_at, get_shifted_key_value, \
    divide_into_subsets

__author__ = 'peter'


def test_reducemap():
    assert reducemap(lambda a, b: a+b, [1, 2, -4, 3, 6, -7], initial=0) == [1, 3, -1, 2, 8, 1]
    assert reducemap(lambda a, b: a+b, [1, 2, -4, 3, 6, -7]) == [3, -1, 2, 8, 1]
    assert reducemap(lambda a, b: a+b, [1, 2, -4, 3, 6, -7], include_zeroth=True) == [1, 3, -1, 2, 8, 1]


def test_itermap():
    # See collatz conjecture
    assert itermap(lambda a: a/2 if a % 2==0 else a*3+1, initial = 12, stop_func=lambda x: x==1, include_zeroth=True) == [12, 6, 3, 10, 5, 16, 8, 4, 2, 1]
    assert itermap(lambda a: a/2 if a % 2==0 else a*3+1, initial = 1, n_steps=5, include_zeroth=True) == [1, 4, 2, 1, 4, 2]


def test_separate_common_items():

    x={'a': 1, 'b':2, 'c':3}
    y={'a': 1, 'b':4, 'c':8}
    z={'a': 1, 'b':2, 'c':9}
    common, separate = separate_common_items([x, y, z])
    assert common == {'a': 1}
    assert separate == [{'b':2, 'c':3}, {'b':4, 'c':8}, {'b':2, 'c':9}]


def test_remove_duplicates():
    assert remove_duplicates(['a', 'b', 'a', 'c', 'c'])==['a', 'b', 'c']
    assert remove_duplicates(['a', 'b', 'a', 'c', 'c'], keep_last=True)==['b', 'a', 'c']
    assert remove_duplicates(['Alfred', 'Bob', 'Cindy', 'Alina', 'Karol', 'Betty'], key=lambda x: x[0])==['Alfred', 'Bob', 'Cindy', 'Karol']
    assert remove_duplicates(['Alfred', 'Bob', 'Cindy', 'Alina', 'Karol', 'Betty'], key=lambda x: x[0], keep_last=True)==['Cindy', 'Alina', 'Karol', 'Betty']
    assert remove_duplicates(['Alfred', 'Bob', 'Cindy', 'Alina', 'Karol', 'Betty'], key=lambda x: x[0], keep_last=True, hashable=False)==['Cindy', 'Alina', 'Karol', 'Betty']


def test_detect_duplicates():
    assert detect_duplicates(['a', 'b', 'a', 'c', 'c'])==[False, False, True, False, True]
    assert detect_duplicates(['a', 'b', 'a', 'c', 'c'], keep_last=True)==[True, False, False, True, False]


def test_remove_common_prefix():
    assert remove_common_prefix([[1, 2, 3, 4], [1, 2, 5], [1, 2, 3, 5]]) == [[3, 4], [5], [3, 5]]
    assert remove_common_prefix([[1, 2, 3, 4], [1, 2, 5], [1, 2, 3, 5]], max_elements=1) == [[2, 3, 4], [2, 5], [2, 3, 5]]


def test_all_equal():

    assert all_equal([2, 2, 2])
    assert not all_equal([2, 2, 3])
    assert all_equal([])


def test_get_absolute_module():

    assert get_absolute_module(test_get_absolute_module) == 'artemis.general.test_should_be_builtins'


def test_insert_at():

    list1 = [1, 2, 3, 4, 5, 6]
    list2 = [7, 8, 9]
    indices = [2, 2, 4]
    r = insert_at(list1, list2, indices)
    assert r == [1, 2, 7, 8, 3, 4, 9, 5, 6]

    list1 = [1, 2, 3, 4, 5, 6]
    list2 = [7, 8, 9]
    indices = [0, 2, 3]
    r = insert_at(list1, list2, indices)
    assert r == [7, 1, 2, 8, 3, 9, 4, 5, 6]

    list1 = [1, 2, 3, 4, 5, 6]
    list2 = [7, 8, 9]
    indices = [2, 2, 20]  # Index too damn high!
    with pytest.raises(AssertionError):
        r = insert_at(list1, list2, indices)


    list1 = [1, 2, 3, 4, 5, 6]
    list2 = [7, 8, 9]
    indices = [2, 3]  # len(indeces) does not match len(list2)
    with pytest.raises(AssertionError):
        r = insert_at(list1, list2, indices)

    list1 = [1, 2, 3, 4, 5, 6]
    list2 = [7, 8, 9]
    indices = [2, 2, 6]  # Index too damn high!
    r = insert_at(list1, list2, indices)
    assert r == [1, 2, 7, 8, 3, 4, 5, 6, 9]

    list1 = []
    list2 = [0, 1, 2, 3, 4]
    indices = [0, 0, 0, 0, 0]
    r = insert_at(list1, list2, indices)
    assert r == [0, 1, 2, 3, 4]


def test_get_shifted_key_value():

    dic = OrderedDict([('a', 1), ('b', 2), ('c', 3), ('d', 4)])
    assert get_shifted_key_value(dic, 'c', -2)==1
    assert get_shifted_key_value(dic, 'c', -1)==2
    assert get_shifted_key_value(dic, 'c', 0)==3
    assert get_shifted_key_value(dic, 'c', 1)==4
    assert get_shifted_key_value(dic, 'a', 1)==2


def test_divide_into_subsets():

    assert divide_into_subsets(range(10), subset_size=3) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert divide_into_subsets(range(9), subset_size=3) == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


if __name__ == '__main__':
    test_separate_common_items()
    test_reducemap()
    test_itermap()
    test_remove_duplicates()
    test_detect_duplicates()
    test_remove_common_prefix()
    test_all_equal()
    test_get_absolute_module()
    test_insert_at()
    test_get_shifted_key_value()
    test_divide_into_subsets()