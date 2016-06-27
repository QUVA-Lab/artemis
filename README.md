# Artemis

Artemis is a collection of tools that make it easier to run experiments in Python.  These include:

- An easy-to-use system for making live plots, to monitor variables in a running experiment.
- A browser-based plotter for displaying live plots.
- A framework for defining experiments and logging their results (text output and figures) so that they can be reviewed later and replicated easily.
- A system for downloading/caching files, to a local directory, so the same code can work on different machines.

## Installation
To use artemis from within your project, use the following to install Artemis and its dependencies: (You probably want to do this in a virtualenv - go `virtualenv venv` to make one).
```
pip install -e git+http://github.com/QUVA-Lab/artemis.git#egg=artemis 
pip install -r venv/src/artemis/requirements.txt
```
Or if your project has a `requirements.txt` file, add the line `-e git+http://github.com/QUVA-Lab/artemis.git#egg=artemis` before running `pip install -r requirements.txt`

Note: During installation, the settings file `.artemisrc` is created in your home directory. In it you can specify the plotting backend to use, and other settings.
## Quick Demos

**Live Plotting**: [artemis.plotting.demo_dbplot](/quva-lab/artemis/blob/master/artemis/plotting/demo_dbplot.py)  

**Recording Experiment Results**:  [artemis.fileman.demo_experiments.py]](/quva-lab/artemis/blob/master/artemis/fileman/demo_experiments.py)  

 ![](https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Tizian_015.jpg/800px-Tizian_015.jpg)




