#!/usr/bin/ui_code
import base64
from artemis.remote.utils import one_time_send_to
import sys
import pickle
"""
This file is called from within a ChildProcess
"""


if __name__ == '__main__':
    _, encoded_pickled_function, return_address, return_port = sys.argv
    pickled_function = base64.b64decode(encoded_pickled_function)
    try:
        gen = pickle.loads(pickled_function)
    except:
        print("Using dill to unpickle")
        import dill
        gen = dill.loads(pickled_function)

    for result in gen():
        pickled_result = pickle.dumps(result, protocol = pickle.HIGHEST_PROTOCOL)
        one_time_send_to(return_address, return_port, pickled_result)
