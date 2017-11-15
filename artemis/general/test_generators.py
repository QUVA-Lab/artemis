from artemis.general.generators import multiplex_generators

import time


def fun1():
    for i in range(10):
        yield i
        time.sleep(0.1)


def fun2():
    for i in range(10):
        time.sleep(0.1)
        yield -i

def test_generator():
    multiplexed_generator = multiplex_generators([fun1(),fun2()])
    for elem in multiplexed_generator:
        print(elem)

def test_generator_names():
    multiplexed_generator = multiplex_generators([("fun1",fun1()), ("fun2",fun2())])

    for elem in multiplexed_generator:
        print(elem)
if __name__ == "__main__":
    test_generator()
    test_generator_names()