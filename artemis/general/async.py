from multiprocessing import Process, Queue, Manager, Lock
import time


class PoisonPill:
    pass


def _async_queue_manager(gen_func, queue):
    for item in gen_func():
        queue.put(item)
    queue.put(PoisonPill)


def iter_asynchronously(gen_func):
    """ Given a generator function, make it asynchonous.  """
    q = Queue()
    p = Process(target=_async_queue_manager, args=(gen_func, q))
    p.start()
    while True:
        item = q.get()
        if item is PoisonPill:
            break
        else:
            yield item


def _async_value_setter(gen_func, namespace, lock):
    for item in gen_func():
        with lock:
            namespace.time_and_data = (time.time(), item)
    with lock:
        namespace.time_and_data = (time.time(), PoisonPill)


class Uninitialized:
    pass


def iter_latest_asynchonously(gen_func, timeout = None, empty_value = None, use_forkserver = False, uninitialized_wait = None):
    """
    Given a generator function, make an iterator that pulls the latest value yielded when running it asynchronously.
    If a value has never been set, or timeout is exceeded, yield empty_value instead.

    :param gen_func: A generator function (a function returning a generator);
    :return:
    """
    if use_forkserver:
        from multiprocessing import set_start_method  # Only Python 3.X
        set_start_method('forkserver')  # On macos this is necessary to start camera in separate thread

    m = Manager()
    namespace = m.Namespace()

    lock = Lock()

    with lock:
        namespace.time_and_data = (-float('inf'), Uninitialized)

    p = Process(target=_async_value_setter, args=(gen_func, namespace, lock))
    p.start()
    while True:
        with lock:
            lasttime, item = namespace.time_and_data
        if item is PoisonPill:  # The generator has terminated
            break
        elif item is Uninitialized:
            if uninitialized_wait is not None:
                time.sleep(uninitialized_wait)
                continue
            else:
                yield empty_value
        elif timeout is not None and (time.time() - lasttime) > timeout:  # Nothing written or nothing recent enough
            yield empty_value
        else:
            yield item
