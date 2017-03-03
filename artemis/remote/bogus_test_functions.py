'''
This file is required to perform plotting server related tests. Its purpose is to be executed in a seperate process or on a remote machine, automatically.
'''



from __future__ import print_function
import sys
import atexit
import time
def function():
    i = 1
    while i<10:
        print(i, file=sys.stderr)
        sys.stderr.flush()
        time.sleep(1.0)
        i +=1


def atexit_function():
    print("atexit called")
    sys.exit(0)


if __name__ == "__main__":
    atexit.register(atexit_function)
    function()