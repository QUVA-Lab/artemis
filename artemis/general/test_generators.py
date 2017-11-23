from artemis.general.generators import multiplex_generators

import pytest
import time


def fun1():
    time.sleep(1.0)
    yield 1

def fun2():
    yield 2
    time.sleep(1.0)

def test_generator():
    multiplexed_generator = multiplex_generators([fun1(),fun2()])
    assert multiplexed_generator.next() == 2
    assert multiplexed_generator.next() == 1

def test_generator_names():
    multiplexed_generator = multiplex_generators([("fun1",fun1()), ("fun2",fun2())])
    res1 = multiplexed_generator.next()
    res2 = multiplexed_generator.next()
    assert res1[1] == 2 and res1[0] == "fun2"
    assert res2[1] == 1 and res2[0] == "fun1"


def fun3():
    yield 2

def test_generator3():
    multiplexed_generator = multiplex_generators([fun1(),fun3()], stop_at_first=False)
    assert multiplexed_generator.next() == 2
    assert multiplexed_generator.next() == 1


if __name__ == "__main__":
    test_generator()
    test_generator3()
    test_generator_names()