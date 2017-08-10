from artemis.remote.utils import one_time_send_to
import sys
import pickle
"""
This file is called from within a ChildProcess
"""

if __name__ == '__main__':
    _, pickled_function, return_address, return_port = sys.argv
    func = pickle.loads(pickled_function)
    result = func()
    pickled_result = pickle.dumps(result, protocol = pickle.HIGHEST_PROTOCOL)
    one_time_send_to(return_address, return_port, pickled_result)
