from __future__ import print_function

import os
from ConfigParser import NoSectionError, NoOptionError

import paramiko
from artemis.config import get_artemis_config_value

from artemis.fileman.config_files import get_config_value
from artemis.fileman.local_dir import get_local_dir
from artemis.remote.utils import get_ssh_connection


def check_config_file(ip_address, file_path=".artemisrc"):
    '''
    Makes sure all required fields are present in ~./artemisrc.
    Also performs test for the different options if applicable
    :param ip_address: The section to look for. Remote ip is assumed. Makes no sense for local ip.
    :return:
    '''
    mandatory_options = ["username", "python"]
    artemisrc_path = os.path.expanduser("~/%s" % file_path)
    for option in mandatory_options:
        try:
            get_artemis_config_value(section=ip_address, option=option)
        except NoSectionError:
            print("Section %s could not be found in %s. Please provide it." % (ip_address, artemisrc_path))
            raise
        except NoOptionError:
            print("Section %s does not contain option %s. Please provide it in %s" % (ip_address, option, artemisrc_path))
            raise

    # optional_options = ["private_key"]
    try:
        private_key_path = get_artemis_config_value(section=ip_address, option="private_key")
        assert os.path.isfile(private_key_path), "The path to the private_key for %s you specified in %s is not valid. You provided %s" % (
        ip_address, artemisrc_path, private_key_path)
    except NoOptionError:
        pass
    # username & private key setup tests:
    try:
        get_ssh_connection(ip_address)
    except paramiko.ssh_exception.AuthenticationException as e:
        if "Authentication failed" in e.message:
            print("An AuthenticationException is being raised. Make sure you have your private key set up correctly")
        else:
            print("An AuthenticationException is being raised. Did you specify the correct username for %s in %s? You provided the username %s" % (
            ip_address, artemisrc_path, get_artemis_config_value(section=ip_address, option="username")))
        raise
    except paramiko.ssh_exception.SSHException:
        try:
            private_key_path = get_artemis_config_value(section=ip_address, option="private_key")
            print("Something is wrong with the private_key you specified in %s for %s . You provided %s" % (artemisrc_path, ip_address, private_key_path))
            raise
        except NoOptionError:
            private_key_path = os.path.join(os.path.expanduser("~"), ".ssh/id_rsa")
            print("You did not provide a private_key path in %s. The default path %s appears to be wrongly set up. "
                  "Please make sure you have correctly set up your private key for %s " % (artemisrc_path, private_key_path, ip_address))

    # python tests:
    python_path = get_artemis_config_value(section=ip_address, option="python")

    command = "python -c 'import os; print(os.path.isfile(os.path.expanduser(\"%s\")))'" % python_path
    ssh_conn = get_ssh_connection(ip_address)
    _, stdout, stderr = ssh_conn.exec_command(command)
    assert stdout.read().strip() == "True", "The provided path to the remote python installation on %s does not exist. You provided %s" % (
    ip_address, python_path)

    command = "%s -c 'print(\"Success\")'" % python_path
    _, stdout, stderr = ssh_conn.exec_command(command)
    err = stderr.read().strip()
    assert stdout.read().strip() == "Success" and not err, "The provided python path on %s does not seem to point to a python executable. " \
                                                           "You provided %s, which resulted in the following error on the remote machine: " % (
                                                           ip_address, python_path, err)


def simple_rsync(local_path, remote_path, ip_address, verbose=False):
    '''
    This method synchronizes local_path and all subfolders with remote_path at the given address.
    This method executes a system rsync call. This is not a general wrapper for rsync. The call is blocking.
    :param local_path:
    :param remote_path: Assumed to be relative to the home dir
    :param ip_address:
    :return:
    '''
    options = "-ah"
    if verbose:
        options += "v"

    local_path = os.path.expanduser(local_path)
    username = get_artemis_config_value(section=ip_address, option="username")
    if remote_path.startswith("~"):
        remote_path = remote_path[1:]
    if remote_path.startswith(("/")):
        remote_path = remote_path[1:]
    # to_path = "%s@%s:/home/%s/%s" % (username, address, username, remote_path)
    to_path = "%s@%s:~/%s" % (username, ip_address, remote_path)
    return rsync(options, from_path=local_path, to_path=to_path)


def rsync(options, from_path, to_path):
    '''
    basic rsync wrapper
    :param options:
    :param from_path:
    :param to_path:
    :return:
    '''
    import subprocess
    print("Starting: rsync %s %s %s" % (options, from_path, to_path))
    if not type(options) is list:
        options = [options]
    command = subprocess.Popen(["rsync"] + options + [from_path, to_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1)
    if "v" in options:
        while True:
            line = command.stdout.readline()
            if line != '':
                print(line.rstrip())
            else:
                break
    err = command.stderr.read().strip()
    if err:
        msg = "rsync received messages on stderr. This might indicate that the command failed or, if you transferred to a remote server," \
              " it might just be some message received by the remote server. \n" \
              "This is because rsync automatically forwards all messages by the remote server to stderr. \n" \
              "If you are confident that the call succeeded although stderr received messages, then catch the RuntimeError accordingly.\n " \
              "The messages received are: \n %s" % err
        raise RuntimeError(msg)
    print("rsync finished")
    return True


def mount_directory(user, ip, local_dir, remote_dir, options=["cache=yes", "kernel_cache", "compression=no", "large_read", "Ciphers=arcfour"],
                    raise_exception=False):
    '''
    This method performs a system call to 'sshfs' in order to mount a remote directory locally. This can be useful if your experiment directory lies
    on you local machine, while your experiment runs on a cluster somewhere. You then do not need to manually synchronize experiment directories across
    machines. Make sure all keys are configured for the mounting operation to succeed without requiring password entry (confer or google ssh-copy-id for instructions).
    Also be aware that this might result in slower reading/writing speed compared to writing to a local directory.
    Tests in our dutch computing cluster show that writing speeds are comparable when the experiment directory is located in the home directory. Since
    the GPU node mounts the home directory within the cluster's network anyways, the writing speeds are identical to when mounting the experiment directory
    on my local desktop machine (which is in the same university WAN). In order for the experiment directory to be available to artemis, make sure you configure your experiment directory to be
    identical to 'local_dir'. todo(matthias): A method to directly mount the experiment directory will be implemented shortly.

    The method registers unmounting of the local directory upon exit of the process (using atexit)

    This method has been tested on Linux only. Mac supports sshfs for example through Homebrew.

    Parameters of this method will be used to compose the sshfs call as:
    sshfs [options] username@ip:remote_dir local_dir
    :param user:
    :param ip:
    :param local_dir: Local directory. Is created if it does not exist. After unmounting, this folder will still be there, but empty.
    :param remote_dir:
    :param options: A list of strings. Each option will be appended as '-o option' to the sshfs call. Call sshfs -h for a list of options.
    When not specified, options ["cache=yes","kernel_cache","compression=no", "large_read", "Ciphers=arcfour"] will be used in an attempt to maximize
    speed at the cost of very weak encryption with arcfour. (Following http://www.admin-magazine.com/HPC/Articles/Sharing-Data-with-SSHFS)
    :param raise_exception: If True, will raise a RunTime error with the stderr printout of the sshfs system call. If False, will only raise a warning.
    :return:
    '''
    import subprocess, warnings, atexit
    local_dir = get_local_dir(local_dir, True)[:-1]
    mounting_command = "sshfs " + " -o ".join(["", ] + options)[1:] + " {}@{}:{} {}".format(user, ip, remote_dir, local_dir)
    sub = subprocess.Popen(mounting_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stderr_pipe = sub.stderr
    stderr_out = stderr_pipe.read()
    if stderr_out:
        if raise_exception:
            raise RuntimeError("Mounting of {} failed with error message: \n {}".format(local_dir, stderr_out))
        else:
            warnings.warn("Mounting of {} failed with error message: \n {} \n Continuing".format(local_dir, stderr_out))
    atexit.register(lambda x: unmount_directory(local_dir))


def unmount_directory(local_dir, raise_exception=False):
    '''
    This method performs the system call 'fusermount -u local_dir' in order to unmount the 'local_dir'.
    todo(matthias): Perform 'umount' on Mac OS instead
    :param local_dir: The directory to unmount
    :param raise_exception: If True, will raise a RunTime error with the stderr printout of the system call. If False, will only raise a warning.
    :return:
    '''
    import subprocess, warnings
    unmounting_command = "fusermount -u {}".format(local_dir)
    sub = subprocess.Popen(unmounting_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stderr_pipe = sub.stderr
    stderr_out = stderr_pipe.read()
    if stderr_out:
        if raise_exception:
            raise RuntimeError("Unmounting of {} failed. Error stack: {}".format(stderr_out))
        else:
            warnings.warn("Unmounting of {} failed with error message: \n{}\n Continuing".format(local_dir, stderr_out))


def mount_experiment_directory(user, ip, local_dir, remote_dir, options=["cache=yes", "kernel_cache", "compression=no", "large_read", "Ciphers=arcfour"],
                               raise_exception=False):
    '''
    Call this method before performing any methods that read the location of the experiment directory. The default for the experiment directory lies in
    $HOME/.artemis/experiments. This method reconfigures this session such that the experiment directory lies instead in 'local_dir'. We then mount the 'remote_dir'
    in the location of 'local_dir'. Make sure 'remote_dir' points to the experiment directory on the machine at 'ip', otherwise artemis will not detect experiment records.
    'local_dir' might point to '/local/USERNAME/experiments' for example, which is located on the local drive of the GPU node within your cluster. Make sure you
    have permission to create this directory.
    Please refer to 'mount_directory' for detailed explanations and parameters.
    :return:
    '''

    pass

def sync_directory():
    '''
    todo(matthias):
    This method performs a 'rsync' system call to copy data from '.artemis/data'
    :return:
    '''
    pass

