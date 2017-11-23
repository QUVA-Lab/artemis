#!/usr/bin/python
import base64
import sys
import os
import pickle
def pardir(s):
    return os.path.abspath(os.path.join(s, os.pardir))
sys.path.extend(pardir(pardir(pardir(__file__))))
import socket
"""
This file is called from within a ChildProcess
"""


def get_sock_and_connect(return_address, return_port):
    from artemis.remote.utils import is_valid_port
    assert is_valid_port(return_port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((return_address, return_port))
    except Exception as e:
        sys.stderr.write("Error in connecting to port %s on address %s\n" % (return_port, return_address))
        sys.stderr.flush()
        raise e
    return sock

def gen_to_host(generator,return_address,return_port):
    from artemis.remote.utils import send_size
    sock = None
    for obj in generator:
        if sock is None:
            # Moving this in here because a wrapping slurm call will not enter this loop except for the first node
            sock = get_sock_and_connect(return_address, return_port)
        pickled_obj = pickle.dumps(obj, protocol = pickle.HIGHEST_PROTOCOL)
        send_size(sock, pickled_obj)
    pickled_obj = pickle.dumps(StopIteration, protocol=pickle.HIGHEST_PROTOCOL)
    send_size(sock, pickled_obj)
    sock.shutdown(2)
    sock.close()

if __name__ == '__main__':
    _, encoded_pickled_function, return_address, return_port = sys.argv
    return_port = int(return_port)
    pickled_function = base64.b64decode(encoded_pickled_function)
    try:
        gen = pickle.loads(pickled_function)
    except Exception,e:
        print("Caught Exception:")
        print(e.args)
        print("Using dill to unpickle")
        import dill
        gen = dill.loads(pickled_function)

    try:
        gen_to_host(gen(),return_address,return_port)
    except IOError as e:
        if e.args[0] == 32:
            print("The receiving side stopped accepting packages")
            pass
        elif e.args[0] == 104:
            print("The receiving side is not accepting connections")
        else:
            raise
    except KeyboardInterrupt as e:
        raise
    except BaseException as e:
        print("Exception caught: ")
        print(e.args)
        raise

