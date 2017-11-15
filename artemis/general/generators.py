import inspect

from six.moves import queue
import threading

def multiplex_generators(generators):
    '''
    generators is either a list of generators or a list of tuples (name,generator)
    :param self:
    :param generators:
    :return:
    '''
    assert isinstance(generators,list)
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
            t = threading.Thread(target=run_one,args=(source,name))
            t.start()
            thrlist.append(t)
        for t in thrlist:
            t.join()
        item_q.put(StopIteration)

    threading.Thread(target=run_all).start()
    while True:
        item = item_q.get()
        if item == StopIteration:
            raise StopIteration
        yield item