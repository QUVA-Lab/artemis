#Artemis plotting Server Edition#
If you want to use the srver edition of artemis' plotting capability, you need to do a few things: 

##Edit your .artemisrc in the homedirectory:##
In order to activate the server edition, add the option 'plotting_server' under the section 'plotting'. The option contains the ip-address of the server you want to use. The following is an example:
```
[plotting]
backend = matplotlib
plotting_server = 8.8.8.8
mode = safe

[fileman]
data_dir = /home/user/.artemis

[8.8.8.8]
username = dave
python = ~/virtualenvs/artemis_env/bin/python
private_key = ~/.ssh/id_rsa
```

If you don't want to bother with remote plotting servers, you are all set now. If you want to have your plots appear on a remote machine, you need to do the following:
In case this ip-address is a remote address (and not 127.0.0.1), this ip-address then needs to also be available as a section in the same file. 
In the section that belongs to the remote address, you need to specify the username that will be used to establish a ssh connection to the remote machine. Also, you need to specify wich python executable you want to use on the remote machine. 

##Make sure artemis is available remotely##
At the moment, you must make sure that artemis is installed in the same version on all the machines, both server(s) and client(s), that you want to use. 

##Make sure that your keys are set##
We do try to take privacy seriously (but are not experts in any way, please use at own risk). The way the connection is established is by using the functionality of the excellent [paramiko](http://www.paramiko.org) library. We didn't want to have the user input username and password every time a remote connection is established. Therefore we chose to rely on the private/public keys being correctly set. The one location where your private keys is being used, is in [artemis/remote/child_processes.py](artemis/remote/child_processes.py), in the function get_ssh_connection(). 

That's it. You are ready to go.



