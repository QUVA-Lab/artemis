from artemis.general.dict_ops import cross_dict_dicts, merge_dicts

__author__ = 'peter'


def test_cross_dict_dicts():
    assert cross_dict_dicts({'a':{'aa': 1}, 'b':{'bb': 2}}, {'c': {'cc': 3}, 'd': {'dd': 4}}) == {
        ('a','c'):{'aa':1, 'cc':3},
        ('a','d'):{'aa':1, 'dd':4},
        ('b','c'):{'bb':2, 'cc':3},
        ('b','d'):{'bb':2, 'dd':4}
        }


def test_dict_merge():

    assert merge_dicts({'a': 1, 'b': 2, 'c': 3}, {'c': 4, 'd': 5}, {'d': 6, 'e': 7}) == {
        'a': 1,
        'b': 2,
        'c': 4,
        'd': 6,
        'e': 7,
        }


if __name__ == "__main__":
    test_dict_merge()
    test_cross_dict_dicts()
