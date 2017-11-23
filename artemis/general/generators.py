import inspect

from six.moves import queue
import threading

def gen_to_queue(generator, obj_queue):
    for obj in generator:
        obj_queue.put(obj)

def wrap_generator_with_event(generator, event):
    obj_queue = queue.Queue()
    t = threading.Thread(target=gen_to_queue, args=(generator,obj_queue))
    t.setDaemon(True)
    t.start()
    while True:
        try:
            obj = obj_queue.get(timeout=0.1)
            yield obj
        except queue.Empty:
            if event.is_set():
                raise StopIteration

def multiplex_generators(generators, stop_at_first=True):
    '''
    generators is either a list of generators or a list of tuples (name,generator)
    :param generators:either a list of generators or a list of tuples (name,generator)
    :param stop_at_first: If true, raises a StopIteration if one generator raised a StopIteration. If false, empties all generators
    :return:
    '''
    assert isinstance(generators,list)
    assert len(generators) > 0, "No generators passed to the multiplexer"

    if isinstance(generators[0],tuple):
        assert inspect.isgenerator(generators[0][1])
        use_names=True
        names,generators = zip(*generators)
    else:
        assert inspect.isgenerator(generators[0])
        use_names=False
        names = [None,]*len(generators)

    item_q = queue.Queue()
    def run_one(source, name=None):
        for item in source:
            if use_names:
                item_q.put((name,item))
            else:
                item_q.put(item)

    def run_all():
        thrlist = []
        for source,name in zip(generators,names):
            t = threading.Thread(target=run_one,args=(source,name), name="Multiplex Sub Thread for CP %s"%(name))
            t.start()
            thrlist.append(t)
        for t in thrlist:
            t.join()

        if use_names:
            item_q.put(("", StopIteration))
        else:
            item_q.put(StopIteration)

    t = threading.Thread(target=run_all,name="Multiplex OverThread")
    t.start()
    while True:
        item = item_q.get()
        if use_names:
            if item[1] == StopIteration:
                raise StopIteration
        else:
            if item == StopIteration:
                raise StopIteration
        yield item