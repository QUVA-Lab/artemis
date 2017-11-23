import socket
import sys
import time

import os
import threading
import pytest

from artemis.experiments import decorators
import subprocess

from artemis.remote.remote_execution import ParamikoPrintThread

from artemis.config import get_artemis_config_value
from artemis.remote.child_processes import RemotePythonProcess, SlurmPythonProcess
from artemis.remote.nanny import Nanny
from artemis.remote.utils import get_local_ips, get_remote_artemis_path, am_I_in_slurm_environment
from functools import partial


ip_addresses = [get_artemis_config_value(section="tests", option="remote_server", default_generator=lambda: "127.0.0.1")]
if ip_addresses[0] not in get_local_ips():
    ip_addresses.append("127.0.0.1")


class ThreadDiff(object):
    def __enter__(self):
        self.pre_exec_threads = threading.enumerate()

    def __exit__(self, type, value, traceback):
        for t in [j for j in threading.enumerate() if j not in self.pre_exec_threads]:
            print("Alive: Thread %s" % t.name)


class AssertNoThreadDiff(object):
    def __enter__(self):
        self.pre_exec_threads = threading.enumerate()

    def __exit__(self, type, value, traceback):
        time.sleep(0.5)
        alive_threads = [j for j in threading.enumerate() if j not in self.pre_exec_threads]

        assert len(alive_threads) == 0, "Some threads are still alive: %s" % (str([th.name for th in alive_threads]))


def get_test_functions_path(ip_address):
    if ip_address in get_local_ips():
        return os.path.join(os.path.dirname(__file__), "remote_test_functions.py")
    else:
        remote_path = get_remote_artemis_path(ip_address)
        return os.path.join(remote_path, "remote/remote_test_functions.py")


def remote_test_func(a, b):
    print('a+b={}'.format(a + b))
    return a + b


def my_func(a, b):
    print('hello hello hello')
    return a + b


def test_remote_python_process():
    in_debug_mode = sys.gettrace() is not None

    p = RemotePythonProcess(
        function=partial(my_func, a=1, b=2),
        ip_address='localhost',
    )

    stdin, stdout, stderr = p.execute_child_process()
    time.sleep(.1)  # That autta be enough

    errtext = stderr.read()
    if in_debug_mode:
        assert errtext.startswith('pydev debugger: ')
    else:
        assert errtext == '', errtext
    out = stdout.read()
    assert 'hello hello hello\n' == out, "was %s instead" % (out)
    res = p.get_return_value()
    assert res == 3, "was %s instead" % (str(res))


def my_gen_func(a, b):
    print("hello hello hello")
    yield a + b


def test_remote_generator_python_process():
    print("test_remote_generator_python_process")
    in_debug_mode = sys.gettrace() is not None

    p = RemotePythonProcess(
        function=partial(my_gen_func, a=1, b=2),
        ip_address='localhost',
    )

    stdin, stdout, stderr = p.execute_child_process()
    time.sleep(.1)  # That autta be enough

    errtext = stderr.read()
    if in_debug_mode:
        assert errtext.startswith('pydev debugger: ')
    else:
        assert errtext == '', errtext
    out = stdout.read()
    assert 'hello hello hello\n' == out, "was %s instead" % (out)
    for res in p.get_return_generator():
        assert res == 3, "was %s instead" % (str(res))


def test_remote_generator_python_process2():
    print("test_remote_generator_python_process2")
    in_debug_mode = sys.gettrace() is not None

    p = RemotePythonProcess(
        function=partial(my_func, a=1, b=2),
        ip_address='localhost',
    )

    stdin, stdout, stderr = p.execute_child_process()
    time.sleep(.1)  # That autta be enough

    errtext = stderr.read()
    if in_debug_mode:
        assert errtext.startswith('pydev debugger: ')
    else:
        assert errtext == '', errtext
    out = stdout.read()
    assert 'hello hello hello\n' == out, "was %s instead" % (out)
    res = p.get_return_value()
    assert res == 3, "was %s instead" % (str(res))


@pytest.mark.skipif(am_I_in_slurm_environment(), reason="Not in SLURM environment")
def test_slurm_process():
    p = SlurmPythonProcess(
        function=partial(my_gen_func, a=1, b=2),
        ip_address="127.0.0.1",
    )

    (stdin, stdout, stderr) = p.execute_child_process()
    time.sleep(.1)  # That autta be enough

    out = stdout.read()
    assert 'hello hello hello\n' == out, "was %s instead" % (out)
    for res in p.get_return_generator():
        assert res == 3, "was %s instead" % (str(res))


@pytest.mark.skipif(am_I_in_slurm_environment(), reason="Not in SLURM environment")
def test_slurm_process2():
    p = SlurmPythonProcess(
        function=partial(my_func, a=1, b=2),
        ip_address="127.0.0.1",
    )

    (stdin, stdout, stderr) = p.execute_child_process()
    time.sleep(.1)  # That autta be enough

    out = stdout.read()
    assert 'hello hello hello\n' == out, "was %s instead" % (out)
    res = p.get_return_value()
    assert res == 3, "was %s instead" % (str(res))


@pytest.mark.skipif(am_I_in_slurm_environment(), reason="Not in SLURM environment")
def test_slurm_nanny1():
    print("test_slurm_nanny1")
    p = SlurmPythonProcess(
        function=partial(my_gen_func, a=1, b=2),
        ip_address="127.0.0.1",
    )

    nanny = Nanny()
    nanny.register_child_process(p)
    mixed_results = nanny.execute_all_child_processes_yield_results()
    res = next(mixed_results)
    assert isinstance(res, tuple)
    assert res[1] == 3, "next(mixed_results) returned %s instead" % (res[1])
    with pytest.raises(StopIteration):
        next(mixed_results)


@pytest.mark.skipif(am_I_in_slurm_environment(), reason="Not in SLURM environment")
def test_slurm_nanny1_2():
    print("test_slurm_nanny1")
    p = SlurmPythonProcess(
        function=partial(my_func, a=1, b=2),
        ip_address="127.0.0.1",
    )

    nanny = Nanny()
    nanny.register_child_process(p)
    nanny.execute_all_child_processes_block_return()
    res = p.get_return_value()
    assert res == 3, "p.get_return_value() returned %s instead" % (res)


def my_blocking_gen_func(a, b):
    print("Sleeping now")
    time.sleep(10000)
    yield a + b


@pytest.mark.skipif(am_I_in_slurm_environment(), reason="Not in SLURM environment")
def test_slurm_nanny2():
    print("test_slurm_nanny2")
    p = SlurmPythonProcess(
        function=partial(my_blocking_gen_func, a=1, b=2),
        ip_address="127.0.0.1",
    )

    nanny = Nanny()
    nanny.register_child_process(p)
    mixed_results = nanny.execute_all_child_processes_yield_results(result_time_out=5)
    with pytest.raises(StopIteration):
        res = next(mixed_results)


def my_blocking_func(a, b):
    print("Sleeping now")
    time.sleep(10000)
    return a + b


def test_nanny3():
    print("test_slurm_nanny3")
    p1 = RemotePythonProcess(
        function=partial(my_blocking_gen_func, a=1, b=2),
        ip_address="127.0.0.1",
    )
    p2 = RemotePythonProcess(
        function=partial(my_gen_func, a=1, b=2),
        ip_address="127.0.0.1",
    )
    nanny = Nanny()
    nanny.register_child_process(p1)
    nanny.register_child_process(p2)
    mixed_results = nanny.execute_all_child_processes_yield_results(result_time_out=15)
    res = next(mixed_results)
    assert res[1] == 3, "next(mixed_results) returned %s instead" % (res[1])
    # Not the first cp is done, all others have terminated
    with pytest.raises(StopIteration):
        res = next(mixed_results)


def test_nanny4():
    print("test_nanny4")
    p1 = RemotePythonProcess(
        function=partial(my_blocking_gen_func, a=1, b=2),
        ip_address="127.0.0.1",
    )
    nanny = Nanny()
    nanny.register_child_process(p1)
    mixed_results = nanny.execute_all_child_processes_yield_results(result_time_out=5)
    with pytest.raises(StopIteration):
        res = next(mixed_results)


def test_nanny5():
    print("test_nanny5")
    p1 = RemotePythonProcess(
        function=partial(my_blocking_gen_func, a=1, b=2),
        # function=partial(my_gen_func, a=1, b=2),
        ip_address="127.0.0.1",
    )
    nanny = Nanny()
    nanny.register_child_process(p1, monitor_if_stuck_timeout=5)
    mixed_results = nanny.execute_all_child_processes_yield_results()
    with pytest.raises(StopIteration):
        res = next(mixed_results)


def test_nanny5_2():
    print("test_nanny5")
    p1 = RemotePythonProcess(
        function=partial(my_blocking_func, a=1, b=2),
        ip_address="127.0.0.1",
    )
    nanny = Nanny()
    nanny.register_child_process(p1, monitor_if_stuck_timeout=5)
    nanny.execute_all_child_processes_block_return()
    assert p1.get_return_value() is None


def my_triggering_gen_function():
    for _ in range(5):
        print(_)
        yield _
        time.sleep(0.1)
    print("my_trigger")
    for _ in range(5):
        print(_ + 5)
        yield _ + 5
        time.sleep(0.1)


def my_triggering_function():
    for _ in range(5):
        print(_)
        time.sleep(0.1)
    print("my_trigger")
    for _ in range(5):
        print(_ + 5)
        time.sleep(0.1)


def test_nanny6():
    print("test_nanny6")
    p1 = RemotePythonProcess(
        function=my_triggering_gen_function,
        ip_address="127.0.0.1",
        name="test_nanny6_cp"
    )
    nanny = Nanny()
    nanny.register_child_process(p1)
    mixed_results = nanny.execute_all_child_processes_yield_results(stdout_stopping_criterium=lambda line: "my_trigger" in line)
    for res in mixed_results:
        pass
        # print(res)
    assert res[1] < 9


def test_nanny6_2():
    print("test_nanny6")
    p1 = RemotePythonProcess(
        function=my_triggering_function,
        ip_address="127.0.0.1",
        name="test_nanny6_cp"
    )
    nanny = Nanny()
    nanny.register_child_process(p1)
    nanny.execute_all_child_processes_block_return(stdout_stopping_criterium=lambda line: "my_trigger" in line)
    assert p1.get_return_value() is None


def return_function():
    for i in range(10):
        print(i)
        time.sleep(0.1)
    return 9


def f():
    for i in range(10):
        yield i
        time.sleep(0.1)
    print("Inner Function Over")


def cp_function():
    print("test_nanny6")
    p = RemotePythonProcess(
        function=f,
        ip_address="127.0.0.1",
        name="Inner_Process",
    )

    nanny = Nanny()
    nanny.register_child_process(p)
    mixed_results = nanny.execute_all_child_processes_yield_results()
    for res in mixed_results:
        print(res)
    print("Outer Function Over")
    return True


def cp_function2():
    print("test_nanny6")
    p = SlurmPythonProcess(
        function=f,
        ip_address="127.0.0.1",
        name="Inner_Process",
    )

    nanny = Nanny()
    nanny.register_child_process(p)
    mixed_results = nanny.execute_all_child_processes_yield_results()
    for res in mixed_results:
        print(res)
    print("Outer Function Over")
    return True


@pytest.mark.skipif(am_I_in_slurm_environment(), reason="Not in SLURM environment")
def test_nanny7():
    print("RemotePythonProcess")
    for fun in [cp_function, cp_function2]:
        p = RemotePythonProcess(
            function=fun,
            ip_address="127.0.0.1",
            name="Outer_Process"
        )

        nanny = Nanny()
        nanny.register_child_process(p)
        nanny.execute_all_child_processes_block_return()
        res = p.get_return_value()
    print("Test over")

@pytest.mark.skipif(am_I_in_slurm_environment(), reason="Not in SLURM environment")
def test_nanny8():
    print("test_nanny8")
    nanny = Nanny()
    p = SlurmPythonProcess(
        function=cp_function,
        ip_address="127.0.0.1",
        name="SlurmProcess",
        slurm_kwargs={"-N": 1}
    )
    nanny.register_child_process(p)
    nanny.execute_all_child_processes_block_return()
    res = p.get_return_value()


@decorators.experiment_function
def remote_func():
    if "SLURM_JOB_NODELIST" in os.environ:
        print("Running within a slurm process")
        command = "scontrol show hostname %s" % os.environ["SLURM_NODELIST"]
        sub = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        node_list = sub.stdout.readlines()
        nodes = [n.strip() for n in node_list]
        hosts = [socket.gethostbyname(h.strip()) for h in nodes]
        print(hosts)
        nanny = Nanny()
        for i, host in enumerate(hosts):
            p = RemotePythonProcess(
                function=inner_most_function,
                ip_address=host,
                name="Innermost_Process_%i" % i,
            )
            nanny.register_child_process(p)
        mixed_results = nanny.execute_all_child_processes_yield_results()
        for res in mixed_results:
            print(res)
            yield res

    else:
        print("Not running within a slurm process")
        hosts = ["127.0.0.1", "127.0.0.1"]
        nanny = Nanny()
        for i, host in enumerate(hosts):
            p = RemotePythonProcess(
                function=inner_most_function,
                ip_address=host,
                name="Innermost_Process_%i" % i,
            )
            nanny.register_child_process(p)
        mixed_results = nanny.execute_all_child_processes_yield_results()
        for res in mixed_results:
            print(res)
            yield res


@pytest.mark.skipif(True, reason="Not in SLURM environment and need to make sure specific node has been requested")
def test_remote_process():
    p = RemotePythonProcess(
        function=inner_most_function,
        ip_address=socket.gethostbyname("node010"),
        name="RemoteProcess"
    )
    (stdin, stdout, stderr) = p.execute_child_process()
    print_thread = ParamikoPrintThread(source_pipe=stdout, target_pipe=sys.stdout, name="Print Thread")
    print_thread.start()
    for i, res in enumerate(p.get_return_generator()):
        assert i == res


def long_func():
    print("Print from long_func()")
    for i in range(50):
        time.sleep(0.1)
        yield i


def inner_most_function():
    for i in range(10):
        time.sleep(0.1)
        yield i


@pytest.mark.skipif(True, reason="Not in SLURM environment and need to make sure specific nodes has been requested")
def test_remote_nanny():
    p = RemotePythonProcess(
        function=long_func,
        ip_address=socket.gethostbyname("node011"),
        name="RemoteProcess1"
    )
    q = RemotePythonProcess(
        function=inner_most_function,
        ip_address=socket.gethostbyname("node010"),
        name="RemoteProcess2"
    )
    nanny = Nanny()
    nanny.register_child_process(p)
    nanny.register_child_process(q)
    results = nanny.execute_all_child_processes_yield_results()
    for i, res in enumerate(results):
        print(res)
        assert i <= 30
    print("Stop")


@pytest.mark.skipif(True, reason="Not in SLURM environment and need to make sure specific nodes has been requested")
def test_remote_nanny2():
    p = RemotePythonProcess(
        function=long_func,
        ip_address=socket.gethostbyname("node011"),
        name="RemoteProcess1"
    )
    q = RemotePythonProcess(
        function=inner_most_function,
        ip_address=socket.gethostbyname("node010"),
        name="RemoteProcess2"
    )
    nanny = Nanny()
    nanny.register_child_process(p, monitor_for_termination=False)
    nanny.register_child_process(q)
    results = nanny.execute_all_child_processes_yield_results()
    for i, res in enumerate(results):
        print(res)
        pass
    # Making sure it went through through the end when not monitoring for termination
    assert res[1] == 49


@pytest.mark.skipif(True, reason="Not in SLURM environment and need to make sure specific nodes has been requested")
def test_remote_nanny3():
    print("test_remote_nanny3")
    p = RemotePythonProcess(
        function=inner_most_function,
        ip_address=socket.gethostbyname("node012"),
        name="RemoteProcess1"
    )
    q = RemotePythonProcess(
        function=long_func,
        ip_address=socket.gethostbyname("node010"),
        name="RemoteProcess2"
    )
    r = RemotePythonProcess(
        function=long_func,
        ip_address=socket.gethostbyname("node011"),
        name="RemoteProcess3"
    )
    nanny = Nanny()
    nanny.register_child_process(p)
    nanny.register_child_process(q)
    nanny.register_child_process(r, monitor_for_termination=False)
    results = nanny.execute_all_child_processes_yield_results()
    senders = []
    for i, res in enumerate(results):
        print(res)
        senders.append(res[0])
        pass
    # Making sure only RemoteProcess3 went through through the end
    assert set(senders[-10:]) == set(["RemoteProcess3"])


def main():
    test_remote_python_process()

    test_remote_generator_python_process()
    test_remote_generator_python_process2()

    test_slurm_process()
    test_slurm_process2()

    test_slurm_nanny1()
    test_slurm_nanny1_2()

    test_slurm_nanny2()
    test_nanny3()
    test_nanny4()
    test_nanny5()
    test_nanny5_2()
    test_nanny6()
    test_nanny6_2()
    test_nanny7()
    test_nanny8()

    test_remote_nanny()
    test_remote_nanny2()
    test_remote_nanny3()
    test_remote_process()

    if am_I_in_slurm_environment():
        remote_func.browse(slurm_kwargs={"-N": 2}, command="0 -s")
    else:
        remote_func.browse(slurm_kwargs={"-N": 2}, command="0")

    print("Tests finished")


if __name__ == "__main__":
    main()
