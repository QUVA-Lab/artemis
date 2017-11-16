import signal
import sys
import time

import os
from six.moves import queue
import pytest

from artemis.config import get_artemis_config_value
from artemis.remote.child_processes import PythonChildProcess, RemotePythonProcess, SlurmPythonProcess
from artemis.remote.nanny import Nanny
from artemis.remote.utils import get_local_ips, get_remote_artemis_path, am_I_in_slurm_environment
from functools import partial

ip_addresses = [get_artemis_config_value(section="tests", option="remote_server",default_generator=lambda: "127.0.0.1")]
# ip_addresses=["127.0.0.1"]
if ip_addresses[0] not in get_local_ips():
    ip_addresses.append("127.0.0.1")

def get_test_functions_path(ip_address):
    if ip_address in get_local_ips():
        return os.path.join(os.path.dirname(__file__), "remote_test_functions.py")
    else:
        remote_path = get_remote_artemis_path(ip_address)
        return os.path.join(remote_path, "remote/remote_test_functions.py")


def test_simple_pcp():
    for ip_address in ip_addresses:
        command = "python %s --callback=%s"%(get_test_functions_path(ip_address),"success_function")
        pyc = PythonChildProcess(ip_address=ip_address,command=command)
        (stdin, stdout, stderr) = pyc.execute_child_process()
        stderr_out =stderr.readlines()
        stderr_out = [line.strip() for line in stderr_out if "pydev debugger" not in line]
        stdout_out = stdout.readlines()
        assert len("".join(stderr_out)) == 0, "Stderr not empty. Received: %s"%stderr_out
        assert len(stdout_out) == 1 and stdout_out[0].strip() == "Success", "Stdout not as expected. Received: %s"%stdout_out

def test_simple_pcp_list():
    for ip_address in ip_addresses:
        command = ["python", get_test_functions_path(ip_address), "--callback=success_function"]
        pyc = PythonChildProcess(ip_address=ip_address,command=command)
        (stdin, stdout, stderr) = pyc.execute_child_process()
        stderr_out = stderr.readlines()
        stderr_out = [line.strip() for line in stderr_out if "pydev debugger" not in line]
        stdout_out = stdout.readlines()
        assert len("".join(stderr_out)) == 0, "Stderr not empty. Received: %s"%stderr_out
        assert len(stdout_out) == 1 and stdout_out[0].strip() == "Success", "Stdout not as expected. Received: %s"%stdout_out

def test_interrupt_process_gently():
    for ip_address in ip_addresses:
        command = ["python", get_test_functions_path(ip_address), "--callback=count_high"]

        cp = PythonChildProcess(ip_address, command)
        stdin , stdout, stderr = cp.execute_child_process()
        time.sleep(5)
        cp.kill()
        time.sleep(1)
        stdout_out = stdout.readlines()
        stderr_out = stderr.readlines()
        if cp.is_local():
            assert stderr_out[-1].strip() == "KeyboardInterrupt"
        else:
            assert stdout_out[-1].strip() == "KeyboardInterrupt"
        assert not cp.is_alive(), "Process is still alive, killing it did not work"

def test_kill_process_gently():
    for ip_address in ip_addresses:
        command = ["python", get_test_functions_path(ip_address), "--callback=sleep_function"]

        cp = PythonChildProcess(ip_address, command)
        stdin , stdout, stderr = cp.execute_child_process()
        time.sleep(1)
        cp.kill()
        time.sleep(1)
        stdout_out = stdout.readlines()
        assert stdout_out[-1].strip() == "Interrupted"
        assert not cp.is_alive(), "Process is still alive, killing it did not work"

def test_kill_process_strongly():
    for ip_address in ip_addresses:
        command = ["python", get_test_functions_path(ip_address), "--callback=hanging_sleep_function"]

        cp = PythonChildProcess(ip_address, command)
        stdin , stdout, stderr = cp.execute_child_process()
        time.sleep(1)
        cp.kill()
        time.sleep(1)
        assert cp.is_alive(), "Process terminated too soon. Check remote_test_functions.py implementation!"
        cp.kill(signal.SIGKILL)
        time.sleep(1)
        assert not cp.is_alive(), "Process is still alive, killing it did not work"



def test_remote_graphics():
    for ip_address in ip_addresses:
        command = ["python", get_test_functions_path(ip_address), "--callback=remote_graphics"]

        cp = PythonChildProcess(ip_address=ip_address,command=command,take_care_of_deconstruct=True)
        i, stdout, stderr = cp.execute_child_process()
        time.sleep(1)
        stderr_out = stderr.readlines()
        stdout_out = stdout.readlines()
        # assert len(stderr_out) == 0, "Stderr not empty. Received: %s"%stderr_out
        assert stdout_out[-1].strip() == "Success", "Stdout not as expected. Received: %s"%stdout_out

        # Do not call decons
        # cp.deconstruct(signum=signal.SIGINT)

def remote_test_func(a, b):
    print('a+b={}'.format(a+b))
    return a+b


def test_remote_child_function():

    for ip_address in ip_addresses:
        pyc = PythonChildProcess(ip_address=ip_address,command=partial(remote_test_func, a=1, b=2))
        (stdin, stdout, stderr) = pyc.execute_child_process()
        stderr_out =stderr.readlines()
        stderr_out = [line.strip() for line in stderr_out if "pydev debugger" not in line]
        stdout_out = stdout.readlines()


def my_func(a, b):
    print('hello hello hello')
    time.sleep(0.01)
    return a+b



def test_remote_python_process():

    in_debug_mode = sys.gettrace() is not None

    p = RemotePythonProcess(
        function=partial(my_func, a=1, b=2),
        ip_address='localhost',
        )

    stdin , stdout, stderr = p.execute_child_process()
    time.sleep(.1)  # That autta be enough

    errtext = stderr.read()
    if in_debug_mode:
        assert errtext.startswith('pydev debugger: ')
    else:
        assert errtext == '', errtext
    assert stdout.read() == 'hello hello hello\n'
    assert p.get_return_value()==3

def my_gen_func(a,b):
    print("Executing Gen Func")
    yield a+b

def test_remote_generator_python_process():

    in_debug_mode = sys.gettrace() is not None

    p = RemotePythonProcess(
        function=partial(my_gen_func, a=1, b=2),
        ip_address='localhost',
        )

    stdin , stdout, stderr = p.execute_child_process()
    time.sleep(.1)  # That autta be enough

    errtext = stderr.read()
    if in_debug_mode:
        assert errtext.startswith('pydev debugger: ')
    else:
        assert errtext == '', errtext
    assert stdout.read() == 'Executing Gen Func\n'
    assert p.get_return_generator().next()==3

@pytest.mark.skipif(am_I_in_slurm_environment(),reason="Not in SLURM environment")
def test_slurm_process():
    p = SlurmPythonProcess(
        function=partial(my_gen_func, a=1, b=2),
        ip_address="127.0.0.1",
    )

    (stdin, stdout, stderr) = p.execute_child_process()
    time.sleep(.1)  # That autta be enough

    for _ in stderr.readlines():
        sys.stdout.write(_)
        sys.stdout.flush()
    assert p.get_return_generator().next()==3

@pytest.mark.skipif(am_I_in_slurm_environment(),reason="Not in SLURM environment")
def test_slurm_nanny1():
    print("test_slurm_nanny1")
    p = SlurmPythonProcess(
        function=partial(my_gen_func, a=1, b=2),
        ip_address="127.0.0.1",
    )

    nanny = Nanny()
    nanny.register_child_process(p)
    mixed_results = nanny.execute_all_child_processes_yield_results()
    res = mixed_results.next()
    assert isinstance(res,tuple)
    assert res[1]==3, "mixed_results.next() returned %s instead"%(res[1])

def my_blocking_gen_func(a,b):
    print("Sleeping now")
    time.sleep(10000)
    yield a+b

@pytest.mark.skipif(am_I_in_slurm_environment(),reason="Not in SLURM environment")
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
        res = mixed_results.next()

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
    res = mixed_results.next()
    assert res[1] == 3, "mixed_results.next() returned %s instead" % (res[1])
    # Not the first cp is done, all others have terminated
    with pytest.raises(StopIteration):
        res = mixed_results.next()

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
        res = mixed_results.next()

def test_nanny5():
    print("test_nanny5")
    p1 = RemotePythonProcess(
        function=partial(my_blocking_gen_func, a=1, b=2),
        # function=partial(my_gen_func, a=1, b=2),
        ip_address="127.0.0.1",
    )
    nanny = Nanny()
    nanny.register_child_process(p1,monitor_if_stuck_timeout=5)
    mixed_results = nanny.execute_all_child_processes_yield_results(result_time_out=None)
    with pytest.raises(StopIteration):
        res = mixed_results.next()

def my_triggering_function():
    for _ in range(5):
        yield _
        time.sleep(1.0)
    print("my_trigger")
    for _ in range(5):
        yield _+5
        time.sleep(1.0)


def test_nanny6():
    print("test_nanny6")
    p1 = RemotePythonProcess(
        function=my_triggering_function,
        ip_address="127.0.0.1",
    )
    nanny = Nanny()
    nanny.register_child_process(p1)
    mixed_results = nanny.execute_all_child_processes_yield_results(result_time_out=None,stdout_stopping_criterium=lambda line: "my_trigger" in line)
    for res in mixed_results:
        assert res[1] < 8




if __name__ == "__main__":
    test_remote_generator_python_process()
    test_nanny3()
    test_nanny4()
    test_nanny5()
    test_nanny6()
    time.sleep(1.0)
    if am_I_in_slurm_environment():
        test_slurm_nanny2()
        test_slurm_nanny1()
        test_slurm_process()
    # test_simple_pcp()
    # test_simple_pcp_list()
    # test_interrupt_process_gently()
    # test_kill_process_gently()
    # test_kill_process_strongly()
    # test_remote_graphics()
    # test_remote_child_function()
    # test_remote_python_process()
    print("Tests finished")
