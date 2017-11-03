from artemis.fileman.local_dir import get_artemis_data_path
import os

from artemis.fileman.persistent_ordered_dict import PersistentOrderedDict


def test_persistent_ordered_dict():

    file_path = get_artemis_data_path('tests/podtest.pkl')
    if os.path.exists(file_path):
        os.remove(file_path)

    with PersistentOrderedDict(file_path) as pod:
        assert list(pod.items()) == []
        pod['a'] = [1, 2, 3]
        pod['b'] = [4, 5, 6]
        pod['c'] = [7, 8]
    pod['d'] = [9, 10]  # Should not be recorded

    with PersistentOrderedDict(file_path) as pod:
        assert list(pod.items()) == [('a', [1, 2, 3]), ('b', [4, 5, 6]), ('c', [7, 8])]
        pod['e']=11

    with PersistentOrderedDict(file_path) as pod:
        assert list(pod.items()) == [('a', [1, 2, 3]), ('b', [4, 5, 6]), ('c', [7, 8]), ('e', 11)]


if __name__ == '__main__':
    test_persistent_ordered_dict()
