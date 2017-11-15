#!/usr/bin/python
from six.moves import queue as Queue
import base64
import threading

from artemis.remote.utils import queue_to_host
import sys
import pickle
"""
This file is called from within a ChildProcess
"""

if __name__ == '__main__':
    _, encoded_pickled_function, return_address, return_port = sys.argv
    return_port = int(return_port)
    pickled_function = base64.b64decode(encoded_pickled_function)
    try:
        gen = pickle.loads(pickled_function)
    except:
        print("Using dill to unpickle")
        import dill
        gen = dill.loads(pickled_function)

    results_queue = Queue.Queue()
    termination_event = threading.Event()
    t = threading.Thread(target=queue_to_host,args=(results_queue,return_address, return_port, termination_event))
    t.start()
    for result in gen():
        pickled_result = pickle.dumps(result, protocol = pickle.HIGHEST_PROTOCOL)
        results_queue.put(pickled_result)
    results_queue.put(pickle.dumps(StopIteration,protocol=pickle.HIGHEST_PROTOCOL))
    termination_event.set()

