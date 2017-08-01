import pytest
from artemis.config import get_artemis_config_value

from artemis.remote.child_processes import PythonChildProcess
from artemis.remote.nanny import Nanny
from artemis.remote.utils import get_local_ips, get_remote_artemis_path
import os,sys
from contextlib import contextmanager
from StringIO import StringIO

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



@contextmanager
def captured_output():
    '''
    Source: https://stackoverflow.com/a/17981937/2068168
    :return:
    '''
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def test_simple_process():
    for ip_address in ip_addresses:
        nanny = Nanny()
        command = "python %s --callback=%s"%(get_test_functions_path(ip_address),"success_function")
        pyc = PythonChildProcess(name="Process", ip_address=ip_address,command=command)
        nanny.register_child_process(pyc,)
        with captured_output() as (out,err):
            nanny.execute_all_child_processes()
        assert "%s: Success"%(pyc.get_name()) == out.getvalue().strip()

@pytest.mark.parametrize("N", [2, 10])
def test_several_simple_processes(N):
    for ip_address in ip_addresses:
        nanny = Nanny()
        command = "python %s --callback=%s"%(get_test_functions_path(ip_address),"success_function")
        for i in range(N):
            pyc = PythonChildProcess(name="Process%i"%i, ip_address=ip_address,command=command)
            nanny.register_child_process(pyc,)
        with captured_output() as (out,err):
            nanny.execute_all_child_processes()
        out_value = out.getvalue().strip()
        for pyc in nanny.managed_child_processes.values():
            assert "%s: Success"%(pyc.get_name()) in out_value

def test_process_termination():
    for ip_address in ip_addresses:
        nanny = Nanny()
        command = "python %s --callback=%s"%(get_test_functions_path(ip_address),"count_low")
        pyc = PythonChildProcess(name="Process1", ip_address=ip_address,command=command)
        nanny.register_child_process(pyc,)
        command = "python %s --callback=%s"%(get_test_functions_path(ip_address),"count_high")
        pyc = PythonChildProcess(name="Process2", ip_address=ip_address,command=command)
        nanny.register_child_process(pyc,)
        with captured_output() as (out,err):
            nanny.execute_all_child_processes(time_out=1)
        check_text = "Child Process Process2 at %s did not terminate 1 seconds after the first process in cluster terminated. Terminating now."%ip_address
        assert check_text in out.getvalue() or check_text in err.getvalue()

def test_output_monitor():
    for ip_address in ip_addresses:
        nanny = Nanny()
        command = "python %s --callback=%s"%(get_test_functions_path(ip_address),"short_sleep")
        pyc = PythonChildProcess(name="Process1", ip_address=ip_address,command=command)
        nanny.register_child_process(pyc,monitor_if_stuck_timeout=5)
        command = "python %s --callback=%s"%(get_test_functions_path(ip_address),"count_high")
        pyc = PythonChildProcess(name="Process2", ip_address=ip_address,command=command)
        nanny.register_child_process(pyc,monitor_if_stuck_timeout=3)
        with captured_output() as (out,err):
            nanny.execute_all_child_processes(time_out=1)
        check_text1 = "Timeout occurred after 0.1 min, process Process1 stuck"
        check_text = "Child Process Process2 at %s did not terminate 1 seconds after the first process in cluster terminated. Terminating now."%ip_address
        assert check_text in out.getvalue() or check_text in err.getvalue()
        assert check_text1 in out.getvalue() or check_text1 in err.getvalue()


def test_iter_print():
    for ip_address in ip_addresses:
        nanny = Nanny()
        command = ["python","-u", get_test_functions_path(ip_address), "--callback=iter_print"]
        pyc = PythonChildProcess(name="P1",ip_address=ip_address,command=command)
        nanny.register_child_process(pyc)
        with captured_output() as (out, err):
            nanny.execute_all_child_processes(time_out=1)
        if pyc.is_local():
            assert str(out.getvalue().strip()) == "\n".join(["P1: %i"%i for i in [0,2,4,6,8]])
            assert str(err.getvalue().strip()) == "\n".join(["P1: %i"%i for i in [1,3,5,7,9]])
        else:
            assert "\r\n".join(["P1: %i" % i for i in range(10)]) == str(out.getvalue().strip())


if __name__ == "__main__":
    test_simple_process()
    test_several_simple_processes(2)
    test_several_simple_processes(10)
    test_output_monitor()
    test_iter_print()
