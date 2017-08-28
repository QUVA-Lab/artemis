'''
This file is required to perform plotting server related tests. Its purpose is to be executed in a seperate process or on a remote machine, automatically.
'''

from __future__ import print_function
from argparse import ArgumentParser
import sys
import atexit
import time

def atexit_function():
    print("atexit called")
    sys.exit(0)

def sleep_function():
    atexit.register(lambda:print("Interrupted"))
    time.sleep(100)

def hanging_sleep_function():
    atexit.register(lambda:time.sleep(100))
    time.sleep(100)

def sleep_N(N):
    time.sleep(N)

def remote_graphics():
    from matplotlib import pyplot as plt
    plt.figure()
    # plt.show()
    plt.draw()
    plt.pause(0.1)
    print("Success")

def count(N):
    for i in range(N):
        print(i)
        time.sleep(1.0)

def iter_print():
    for i in range(10):
        if i%2==0:
            sys.stdout.write(str(i))
            sys.stdout.write("\n")
            # sys.stdout.flush()
        else:
            sys.stderr.write(str(i))
            sys.stderr.write("\n")
            # sys.stderr.flush()
        time.sleep(1.0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--callback',type=str,default="") # to know which test is calling this script and to setup the behaviour accordingly.
    args = parser.parse_args()
    fun = {
        "success_function":lambda: print("Success"),
        "sleep_function":sleep_function,
        "hanging_sleep_function":hanging_sleep_function,
        "remote_graphics":remote_graphics,
        "count_low":lambda: count(5),
        "count_high":lambda: count(50),
        "short_sleep":lambda: sleep_N(10),
        "iter_print":iter_print,
        }[args.callback]
    fun()

