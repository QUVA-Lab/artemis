import SocketServer
import os
import sys
import time
import signal
from artemis.plotting.plotting_backend import get_plotting_server_address
from artemis.remote.child_processes import check_ssh_connection, check_if_port_is_free, execute_command, ChildProcess, \
    ParamikoPrintThread, Nanny
from artemis.remote.utils import get_local_ips, get_socket, get_remote_artemis_path
from pytest import raises


ip_addresses = [get_plotting_server_address()]
if ip_addresses[0] not in get_local_ips():
    ip_addresses.append("127.0.0.1")


def test_check_ssh_connections():
    for ip_address in ip_addresses:
        if ip_address not in get_local_ips():
            check_ssh_connection(ip_address)


def test_check_if_port_is_free():
    for ip_address in ip_addresses:
        if ip_address not in get_local_ips():
            with raises(AssertionError):
                check_if_port_is_free(ip_address,80)
        else:
            sock, port = get_socket(ip_address,7005)
            # port is now not free:
            with raises(SocketServer.socket.error):
                check_if_port_is_free(ip_address, port)
            sock.close()
            # now no error to be expected
            check_if_port_is_free(ip_address,port)


def test_communications():
    for ip_address in ip_addresses:
        execute_command(ip_address=ip_address,
                    blocking=True,
                    command = "python -c 'from __future__ import print_function\nimport sys,time\nfor i in range(10): print(i, file=sys.stderr if i%2==0 else sys.stdout);sys.stdout.flush();time.sleep(0.3)'")


def test_kill_process_gently():

    for ip_address in ip_addresses:
        if ip_address in get_local_ips():
            command = "python %s"%os.path.join(os.path.dirname(__file__), "bogus_test_functions.py")
        else:
            remote_path = get_remote_artemis_path()
            command = "python %s"%os.path.join(remote_path, "remote/bogus_test_functions.py")

        cp = ChildProcess(ip_address, command)
        stdin , stdout, stderr = cp.execute_child_process()
        # stdin , stdout, stderr = ssh_conn.exec_command(command)

        pid=cp.get_pid()
        print("Pid: %s"%pid)
        #stdout
        t1 = ParamikoPrintThread(source_pipe=stdout, target_pipe=sys.stdout, prefix="stdout: ")
        t1.start()
        # stderr
        t2 = ParamikoPrintThread(source_pipe=stderr, target_pipe=sys.stderr, prefix="stderr: ")
        t2.start()

        print("Waiting 4 seconds")
        time.sleep(4)

        print ("Killing Process %s:" %(pid) )
        cp.kill()
        print("Waiting 4 seconds")
        time.sleep(4)
        if cp.is_alive():
            print("Process still alive")
        else:
            print("Process dead")


        print("Terminating")



def test_remote_graphics():
    for ip_address in ip_addresses:
        # command = 'export DISPLAY=:0.0; python -c "from matplotlib import pyplot as plt;import time; plt.figure();plt.show();time.sleep(10)"'
        if ip_address not in get_local_ips():
            command =["export DISPLAY=:0.0;", "python","-u", "-c", "from matplotlib import pyplot as plt;import time; plt.figure();plt.show();time.sleep(10)"]
        else:
            command =["python","-u", "-c", "from matplotlib import pyplot as plt;import time; plt.figure();plt.show();time.sleep(10)"]

        cp = ChildProcess(ip_address=ip_address,command=command)
        i, o, e = cp.execute_child_process()
        time.sleep(5)
        cp.deconstruct(message=signal.SIGTERM)
        print(o.read())
        print(e.read())


def test_is_alive():
    for ip in ip_addresses:
        command ="python -u -c 'from __future__ import print_function\nimport sys,time\nfor i in range(10): print(i, file=sys.stderr if i%2==0 else sys.stdout);time.sleep(0.3)'"
        cp = ChildProcess(ip_address=ip, command=command)
        i,o,e = cp.execute_child_process()
        t = ParamikoPrintThread(o,sys.stdout).start()
        t = ParamikoPrintThread(e,sys.stderr).start()
        while cp.is_alive():
            print("alive")
            time.sleep(0.3)
        print("dead")


if __name__ == "__main__":
    test_check_ssh_connections()
    test_check_if_port_is_free()
    test_communications()
    test_kill_process_gently()
    test_remote_graphics()
    test_is_alive()
