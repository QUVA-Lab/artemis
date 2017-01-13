import os
import socket
import sys
import time

from artemis.remote.child_processes import check_ssh_connection, check_if_port_is_free, execute_command, ChildProcess, \
    ParamikoPrintThread, Nanny
from artemis.remote.utils import get_local_ips


def test_check_ssh_connections():
    connections = ["146.50.28.6"]
    check_ssh_connection(connections)


def test_check_if_port_is_free():
    connection = "146.50.28.6:9005"
    ip, port = connection.split(":")
    check_if_port_is_free(ip,port)


def test_communications():
    ip_address = "146.50.28.6"#"python -c $'from __future__ import print_function\nimport sys,time\nfor i in range(7): print(i, file=sys.stdout if i%2==0 else sys.stdout);time.sleep(10.0)'"
    ip_address = socket.gethostbyname(socket.gethostname())
    execute_command(ip_address=ip_address,
                    blocking=True,
                    command = "python -c 'from __future__ import print_function\nimport sys,time\nfor i in range(20): print(i, file=sys.stderr if i%2==0 else sys.stdout);sys.stdout.flush();time.sleep(1.0)'")


def test_kill_process_gently():
    ip_address = "146.50.28.6"
    # ip_address = socket.gethostbyname(socket.gethostname())

    command = "python %s"%(os.path.expanduser("%s/PycharmProjects/Distributed-VI/distributed_vi/remote/bogus_test_functions.py" % ("/Users/matthias" if ip_address in get_local_ips() else "/home/mreisser/")))

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


def test_nanny_lifetime():
    ip1 = "146.50.28.6"
    ip2 = socket.gethostbyname(socket.gethostname())
    command1 = "python %s"%(os.path.expanduser("%s/PycharmProjects/Distributed-VI/distributed_vi/remote/bogus_test_functions.py" % ("/Users/matthias" if ip1 in get_local_ips() else "/home/mreisser/")))
    command2 = "python %s"%(os.path.expanduser("%s/PycharmProjects/Distributed-VI/distributed_vi/remote/bogus_test_functions.py" % ("/Users/matthias" if ip2 in get_local_ips() else "/home/mreisser/")))
    command2 = "python %s"%(os.path.expanduser("%s/PycharmProjects/Distributed-VI/distributed_vi/distributed/parameter_server.py" % ("/Users/matthias" if ip2 in get_local_ips() else "/home/mreisser/")))

    # cp1 = ChildProcess(ip_address=ip1,command=command1)
    cp2 = ChildProcess(ip_address=ip2,command=command2)

    nanny = Nanny()
    # nanny.register_child_process(cp1)
    nanny.register_child_process(cp2)

    nanny.execute_all_child_processes()


def test_remote_graphics():
    ip_address = "146.50.28.6"
    command = 'export DISPLAY=:0.0; python -c "from matplotlib import pyplot as plt;import time; plt.figure();plt.show();time.sleep(10)"'
    # command = "python %s"%(os.path.expanduser("%s/PycharmProjects/Distributed-VI/distributed_vi/remote/bogus_test_functions.py" % ("/Users/matthias" if ip_address in get_local_ips() else "/home/mreisser/")))
    cp = ChildProcess(ip_address="146.50.28.6",command=command)
    i, o, e = cp.execute_child_process()
    time.sleep(5)
    cp.deconstruct()
    print(o.read())
    print(e.read())


def test_is_alive():
    ip = "146.50.28.6"
    command ="python -u -c'from __future__ import print_function\nimport sys,time\nfor i in range(20): print(i, file=sys.stderr if i%2==0 else sys.stdout);time.sleep(1.0)'"
    cp = ChildProcess(ip_address=ip, command=command)
    i,o,e = cp.execute_child_process()
    t = ParamikoPrintThread(o,sys.stdout).start()
    t = ParamikoPrintThread(e,sys.stderr).start()
    while True:
        print(cp.is_alive())
        time.sleep(1)


if __name__ == "__main__":
    test_check_if_port_is_free()
    communication_test()
    test_kill_process_gently()
    test_nanny_lifetime()
    test_remote_graphics()
    test_is_alive()
