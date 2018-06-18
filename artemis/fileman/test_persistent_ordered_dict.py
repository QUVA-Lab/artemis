from artemis.fileman.persistent_ordered_dict import PersistentOrderedDict
from artemis.fileman.temporary_filename import use_temporary_filename


def test_persistent_ordered_dict():

    with use_temporary_filename('tests/podtest.pkl') as file_path:

        pod = PersistentOrderedDict(file_path)
        assert list(pod.items()) == []
        pod['a'] = [1, 2, 3]
        pod['b'] = [4, 5, 6]
        pod['c'] = [7, 8]

        pod2 = PersistentOrderedDict(file_path)
        assert list(pod2.items()) == [('a', [1, 2, 3]), ('b', [4, 5, 6]), ('c', [7, 8])]
        pod['e']=11

        pod3 = PersistentOrderedDict(file_path)
        assert list(pod3.items()) == [('a', [1, 2, 3]), ('b', [4, 5, 6]), ('c', [7, 8]), ('e', 11)]


def test_catches_modifications():

    with use_temporary_filename('tests/test_catches_modifications.pkl') as file_path:

        pod1 = PersistentOrderedDict(file_path)

        pod1['a'] = 1

        pod2 = PersistentOrderedDict(file_path)
        assert pod2['a']==1

        pod1['a'] = 2
        assert pod2['a']==2

        pod2['a']=3
        assert pod1['a']==3

        pod2['b']=4
        assert list(pod1.items()) == [('a', 3), ('b', 4)]

        pod3 = PersistentOrderedDict(file_path, items=[('b', 5), ('c', 6)])
        assert list(pod1.items())==list(pod2.items())==list(pod3.items()) == [('a', 3), ('b', 5), ('c', 6)]


def test_has_changed():

    with use_temporary_filename('tests/test_catches_modifications_has_changed.pkl') as file_path:

        pod1 = PersistentOrderedDict(file_path)
        pod1['a'] = 1
        assert not pod1.has_changed()

        pod2 = PersistentOrderedDict(file_path)
        assert not pod1.has_changed()

        assert pod2['a']==1
        assert not pod1.has_changed()

        pod2['a']=2
        assert pod1.has_changed()


def test_block_change():

    with use_temporary_filename('tests/test_catches_block_change.pkl') as file_path:

        pod1 = PersistentOrderedDict(file_path)
        pod1['a'] = 1
        assert list(pod1.items()) == [('a', 1)]
        pod2 = PersistentOrderedDict(file_path)

        with pod2:
            pod2['b'] = 2
            assert list(pod1.items()) == [('a', 1)]
        assert list(pod1.items()) == [('a', 1), ('b', 2)]


if __name__ == '__main__':
    test_persistent_ordered_dict()
    test_catches_modifications()
    test_has_changed()
    test_block_change()
