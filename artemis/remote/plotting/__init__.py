forward_to_server = True


def should_I_forward_to_server():
    return forward_to_server


def set_forward_to_server(val=True):
    '''
    If set to false, makes sure that the process calling dbplot does not forward to a different plotting server.
    :param val:
    :return:
    '''
    global forward_to_server
    forward_to_server = val