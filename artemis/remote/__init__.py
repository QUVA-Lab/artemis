'''
This package will contain routines to communicate with remote servers
'''

forward_to_server = True

def should_I_forward_to_server():
    return forward_to_server

def set_forward_to_server(val=True):
    global forward_to_server
    forward_to_server = val

