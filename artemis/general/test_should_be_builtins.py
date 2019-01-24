from collections import OrderedDict

import pytest

from artemis.general.should_be_builtins import itermap, reducemap, separate_common_items, remove_duplicates, \
    detect_duplicates, remove_common_prefix, all_equal, get_absolute_module, insert_at, get_shifted_key_value, \
    divide_into_subsets, entries_to_table, natural_keys, switch

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
    assert list(remove_duplicates(['a', 'b', 'a', 'c', 'c']))==['a', 'b', 'c']
    assert list(remove_duplicates(['a', 'b', 'a', 'c', 'c'], keep_last=True))==['b', 'a', 'c']
    assert list(remove_duplicates(['Alfred', 'Bob', 'Cindy', 'Alina', 'Karol', 'Betty'], key=lambda x: x[0]))==['Alfred', 'Bob', 'Cindy', 'Karol']
    assert list(remove_duplicates(['Alfred', 'Bob', 'Cindy', 'Alina', 'Karol', 'Betty'], key=lambda x: x[0], keep_last=True))==['Cindy', 'Alina', 'Karol', 'Betty']
    assert list(remove_duplicates(['Alfred', 'Bob', 'Cindy', 'Alina', 'Karol', 'Betty'], key=lambda x: x[0], keep_last=True, hashable=False))==['Cindy', 'Alina', 'Karol', 'Betty']


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


def test_entries_to_table():

    assert entries_to_table([[('a', 1), ('b', 2)], [('a', 3), ('b', 4), ('c', 5)]]) == (['a', 'b', 'c'], [[1, 2, None], [3, 4, 5]])


def test_natural_keys():

    assert sorted(['y8', 'x10', 'x2', 'y12', 'x9'], key=natural_keys) == ['x2', 'x9', 'x10', 'y8', 'y12']


def test_switch_statement():

    responses = []
    for name in ['nancy', 'joe', 'bob', 'drew']:
        with switch(name) as case:
            if case('bob', 'nancy'):
                response = "Come in, you're on the guest list"
            elif case('drew'):
                response = "Sorry, after what happened last time we can't let you in"
            else:
                response = "Sorry, {}, we can't let you in.".format(case.value)
        responses.append(response)

    assert responses == [
        "Come in, you're on the guest list",
        "Sorry, joe, we can't let you in.",
        "Come in, you're on the guest list",
        "Sorry, after what happened last time we can't let you in"
    ]


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
    test_entries_to_table()
    test_natural_keys()
    test_switch_statement()
