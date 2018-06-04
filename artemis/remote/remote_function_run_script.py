#!/usr/bin/python
import base64
import sys

from artemis.general.functional import get_partial_root
from artemis.remote.utils import one_time_send_to, is_valid_port
import pickle
"""
This file is called from within a ChildProcess
"""

if __name__ == '__main__':
    _, encoded_pickled_function, return_address, return_port = sys.argv
    pickled_function = base64.b64decode(encoded_pickled_function)
    return_port = int(return_port)

    if return_port == -1:
        send_back_results = False
    else:
        send_back_results = True
        assert is_valid_port(return_port)

    try:
        func = pickle.loads(pickled_function)
    except:
        print("Using dill to unpickle")
        import dill
        func = dill.loads(pickled_function)

    # This is so that the function can setup their own communication if desired

    get_partial_root(func).func_globals["RETURN_ADDRESS"] = return_address
    get_partial_root(func).func_globals["RETURN_PORT"] = return_port

    result = func()

    if send_back_results:

        pickled_result = pickle.dumps(result, protocol = pickle.HIGHEST_PROTOCOL)
        try:
            one_time_send_to(return_address, return_port, pickled_result)
        except:
            pass

