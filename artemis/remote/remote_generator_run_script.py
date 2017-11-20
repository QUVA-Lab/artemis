#!/usr/bin/python
import base64
import time
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

def gen_to_host(generator,return_address,return_port):
    sock = None
    for obj in generator:

        # Moving this in here because a wrapping slurm call will not enter this loop except for the first node
        if sock is None:
            from artemis.remote.utils import is_valid_port, send_size
            assert is_valid_port(return_port)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.connect((return_address, return_port))
            except:
                sys.stderr.write("Error in connecting to port %s on address %s\n" % (return_port, return_address))
                sys.stderr.flush()
                raise

        pickled_obj = pickle.dumps(obj, protocol = pickle.HIGHEST_PROTOCOL)
        send_size(sock, pickled_obj)
    sock.close()

if __name__ == '__main__':
    _, encoded_pickled_function, return_address, return_port = sys.argv
    print("Trying to reach port %s on address %s"%(return_port,return_address))
    time.sleep(10.0)
    # if return_address in get_local_ips():
    #     return_address = "127.0.0.1"
    return_port = int(return_port)
    pickled_function = base64.b64decode(encoded_pickled_function)
    try:
        gen = pickle.loads(pickled_function)
    except Exception,e:
        # raise e
        print("Caught Exception:")
        print(e.args)
        print("Using dill to unpickle")
        import dill
        gen = dill.loads(pickled_function)

    try:
        try:
            gen_to_host(gen(),return_address,return_port)
        except IOError, e:
            print(e.args)
            sys.exit(0)
        except KeyboardInterrupt:
            print("KeyboardInterrupt received")
        except BaseException, e:
            print("Exception caught: ")
            print(e.args)
            raise e
    except IOError:
        pass

