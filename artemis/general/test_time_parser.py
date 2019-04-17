from datetime import timedelta

from pytest import raises

from artemis.general.time_parser import parse_time


def test_time_parser():

    assert parse_time('8h') == timedelta(hours=8)
    assert parse_time('3d8h') == timedelta(days=3, hours=8)
    assert parse_time('5s') == timedelta(seconds=5)
    assert parse_time('.25s') == timedelta(seconds=0.25)
    assert parse_time('.25d4h') == timedelta(days=0.25, hours=4)
    with raises(ValueError):
        assert parse_time('0.0.25d4h') == timedelta(days=0.25, hours=4)
    with raises(AssertionError):
        assert parse_time('5hr')
    with raises(AssertionError):
        assert parse_time('5q')
    with raises(AssertionError):
        print(parse_time('5h4q'))


if __name__ == '__main__':
    test_time_parser()
