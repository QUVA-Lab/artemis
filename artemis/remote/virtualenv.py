import json
import os
import sys
from collections import OrderedDict
import numpy as np
import pip
from six.moves import input

from artemis.fileman.config_files import get_config_value
from artemis.config import get_artemis_config_value
from artemis.remote.utils import get_ssh_connection


def get_remote_installed_packages(ip_address):
    '''
    This method queries a remote python installation about the installed packages.
    All necessary information is extracted from ~/.artemisrc
    :param address: Ip address of remote server
    :return:
    '''
    python_executable = get_artemis_config_value(section=ip_address, option="python")
    function = "%s -c 'import pip; import json; print json.dumps({i.key: i.version  for i in pip.get_installed_distributions() })' "%python_executable

    ssh_conn = get_ssh_connection(ip_address)
    stdin , stdout, stderr = ssh_conn.exec_command(function)
    err = stderr.read()
    if err:
        msg="Quering %s python installation at %s sent a message on stderr. If you are confident that the error can be ignored, catch this RuntimeError" \
            "accordingly. The error is: %s"%(ip_address, python_executable, err)
        raise RuntimeError(msg)

    installed_packages = json.loads(stdout.read())
    ssh_conn.close()
    return installed_packages


def install_packages_on_remote_virtualenv(ip_address, packages):
    '''
    This function installs every package in packages on the remote virtual environment specified by the ip_address in ~/.artemisrc.
    In case the remote pip install -U command returns anything that is not "Successfully installed" or "Requirement already up-to-date" on stdout,
    the user is informed and an error that is not a SNIMissingWarning or InsecurePlatformWarning is printed.
    :param ip_address: ip_address, whose virtualenv is being modified
    :param packages: a dict {key:version} of package name and associated version
    :return:
    '''
    if len(packages) == 0:
        return
    print("installing/upgrading remote packages ...")
    python_path = get_artemis_config_value(ip_address,"python")
    activate_path = os.path.join(os.path.dirname(python_path),"activate") # TODO: Make this work without the user using virtualenv
    activate_command = "source %s"%activate_path
    ssh_conn = get_ssh_connection(ip_address)
    for key,version in packages.items():
        install_command = "pip install -U %s==%s" %(key,version)
        function = "; ".join([activate_command, install_command])
        stdin , stdout, stderr =ssh_conn.exec_command(function)
        out = stdout.read()
        if "Successfully installed" in out or "Requirement already up-to-date" in out:
            pass
        else:
            print(("Error in installing %s==%s:" % (key,version)))
            err = stderr.read()
            err = "\n".join([s for s in err.strip().split("\n") if "SNIMissingWarning" not in s and "InsecurePlatformWarning" not in s])
            print(err)
    ssh_conn.close()
    print("... Done")


def check_diff_local_remote_virtualenv(ip_address, auto_install=None, auto_upgrade=None, ignore_warnings=True):
    '''

    This method compares a remote virtual environment with the currently active virtual environment. Any local package that is not installed at
    the remote virtual environment or is installed in a different version at the remote virtual environment is identified. If auto_install is left at None,
    the user is prompted to specify which missing package he/she wants to install. If it is set to True, all missing packges are installed. If set to False,
    the missing packages are ignored. The same holds for auto_upgrade for different versions of installed packages. If some packages failed to install on
    the remote virtual environment, the user is asked to continue or exit the program unless 'ignore_warnings' is set to True, in which case the program
    continues even when some packages failed to install/upgrade.
    By setting any boolean value for auto_install and auto_upgrade, as well as setting ignore_warnings to True, no user interaction will be prompted.
    :param ip_address:
    :param auto_install:
    :param auto_upgrade:
    :param ignore_warnings:
    :return:
    '''
    print(("="*10 + " Checking remote virtualenv %s "%ip_address + "="*10))
    remote_packages = get_remote_installed_packages(ip_address)
    local_packages = {i.key: i.version  for i in pip.get_installed_distributions(include_editables=False)}
    missing_packages = OrderedDict()
    different_versions = OrderedDict()
    for (local_key, local_version) in local_packages.items():
        if local_key not in remote_packages.keys():
            missing_packages[local_key] = local_version
        elif local_version != remote_packages[local_key]:
            different_versions[local_key] = (local_version, remote_packages[local_key])

    # Install missing packages
    chosen_packages_to_install_and_update = None
    if len(missing_packages) > 0:
        missing_packages_string = ", ".join(missing_packages.keys())
        chosen_packages_to_install_and_update = {}
        if auto_install is not None:
            if auto_install:
                chosen_packages_to_install_and_update = missing_packages
            else:
                print(("Some locally installed packages are missing in the remote virtualenv at %s. You chose not to install them. "
                  "The missing packages are: \n %s " % (ip_address,missing_packages_string)))
        else:
            print("The following locally installed packages are missing in the virtualenv at %s:" % ip_address)
            print('\n'.join(['%s: %s (%s)' % (i, key, missing_packages[key]) for i, key in enumerate(missing_packages.keys())]))
            valid = False
            while not valid:
                ix = input("Please specify in the format '1, 3, 4' all packages you want to install. Can be empty or 'all' ")
                ix=ix.strip(",")
                numbers = ix.split(",")
                if numbers[0] == "all" and len(numbers) == 1:
                    chosen_packages_to_install_and_update = missing_packages
                    valid = True
                elif not numbers[0] and len(numbers) == 1:
                    valid = True
                else:
                    try:
                        numbers = [int(n) for n in numbers]
                    except:
                        print(("Please format input correctly. The input recieved was: \n%s"%ix))
                        continue
                    if not np.all([n >= 0 and n < len(missing_packages) for n in numbers]):
                        print(("Please make sure every number is valid. The input recieved was: \n%s"%ix))
                        continue
                    for i, key in enumerate(missing_packages.keys()):
                        if i in numbers:
                            chosen_packages_to_install_and_update[key] = missing_packages[key]
                    valid=True

    # Upgrade different version packages
    if len(different_versions) > 0:
        if auto_upgrade is not None:
            if auto_upgrade:
                for key, versions in different_versions.items():
                    chosen_packages_to_install_and_update[key] = versions[0]
            else:
                print(("Some locally installed packages are installed in a different version than on the remote virtualenv at %s. You chose not to upgrade them. "
                  "The different packages are:" % (ip_address)))
                for key, versions in different_versions.items():
                    print(("%s: local %s, reote %s") % (key, versions[0],versions[1]))
        else:
            print("The following locally installed packages are installed in a different version than on the virtualenv at %s:" % ip_address)
            print('\n'.join(['%s: %s (local: %s) => (remote: %s)' % (i, key, different_versions[key][0], different_versions[key][1]) for i, key in enumerate(different_versions.keys())]))
            valid = False
            while not valid:
                ix = input("Please specify in the format '1, 3, 4' all remote packages you want to upgrade (downgrade) to the local version. Can be empty or 'all' ")
                ix=ix.strip(",")
                numbers = ix.split(",")
                if numbers[0] == "all" and len(numbers) == 1:
                    for key, versions in different_versions.items():
                        chosen_packages_to_install_and_update[key] = versions[0]
                    valid = True
                elif not numbers[0] and len(numbers) == 1:
                    valid = True
                else:
                    try:
                        numbers = [int(n) for n in numbers]
                    except:
                        print(("Please format input correctly. The input recieved was: \n%s"%ix))
                        continue
                    if not np.all([n >= 0 and n < len(different_versions) for n in numbers]):
                        print(("Please make sure every number is valid. The input recieved was: \n%s"%ix))
                        continue
                    for i, key in enumerate(different_versions.keys()):
                        if i in numbers:
                            chosen_packages_to_install_and_update[key] = different_versions[key][0]
                    valid=True
    if chosen_packages_to_install_and_update is not None:
        if len(chosen_packages_to_install_and_update) == 0:
            print ("virtualenv up-to-date")
        else:
            install_packages_on_remote_virtualenv(ip_address, chosen_packages_to_install_and_update)
            if not ignore_warnings:
                remote_packages = get_remote_installed_packages(ip_address)
                missing_packages = OrderedDict()
                different_versions = OrderedDict()
                for (local_key, local_version) in local_packages.items():
                    if local_key not in remote_packages.keys():
                        missing_packages[local_key] = local_version
                    elif local_version != remote_packages[local_key]:
                        different_versions[local_key] = (local_version, remote_packages[local_key])
                if len(missing_packages)>0 or len(different_versions)>0:
                    valid = False
                    while not valid:
                        ix=input("The following packages could not be installed or upgraded: \n %s \nDo you want to continue? (y/n)"% (",".join(missing_packages.keys() + different_versions.keys())))
                        ix=ix.strip().lower()
                        if ix in ("y", "n"):
                            valid=True

                    if ix == "n":
                        print("Exiting...")
                        sys.exit(0)
                    else:
                        print("Ignoring discrepancies...")
    
    print(("="*10 + " Done remote virtualenv %s "%ip_address + "="*10))
