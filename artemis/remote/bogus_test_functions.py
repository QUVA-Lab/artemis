'''
This file is required to perform plotting server related tests. Its purpose is to be executed in a seperate process or on a remote machine, automatically.
'''



from __future__ import print_function

import json
import sys
import atexit
import time

from artemis.remote.utils import one_time_send_to

import argparse


def function(port,address):
    port = int(port)
    i = 1
    while i<400:
        print(i)
        sys.stdout.flush()
        time.sleep(0.5)
        i +=1
    d = {"a":1,"b":2,"bla":"dbu"}
    # print("Sleeping now")
    # time.sleep(1000)
    # print("sending now:")
    # one_time_send_to(address, port, json.dumps(d))

def atexit_function():
    print("atexit called")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port")
    parser.add_argument("--address")
    args = parser.parse_args()

    atexit.register(atexit_function)
    function(args.port, args.address)