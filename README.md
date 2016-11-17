# Artemis

Artemis is a collection of tools that make it easier to run experiments in Python.  These include:

- An easy-to-use system for making live plots, to monitor variables in a running experiment.
- A browser-based plotter for displaying live plots.
- A framework for defining experiments and logging their results (text output and figures) so that they can be reviewed later and replicated easily.
- A system for downloading/caching files, to a local directory, so the same code can work on different machines.

## Installation
To use artemis from within your project, use the following to install Artemis and its dependencies: (You probably want to do this in a virtualenv with the latest version of pip - run `virtualenv venv; source venv/bin/activate; pip install --upgrade pip;` to make one).
```
pip install -e git+http://github.com/QUVA-Lab/artemis.git#egg=artemis 
```
<!--- To verify that the installation worked, go:
```
cd venv/src/artemis
py.test
```
All tests should pass.
(pytest for some reason cant find modules when you do this alone)--->
Note: During installation, the settings file `.artemisrc` is created in your home directory. In it you can specify the plotting backend to use, and other settings.


## Quick Demos

**Live Plotting**: [/artemis/plotting/demo_dbplot.py](/artemis/plotting/demo_dbplot.py)  

**Recording Experiment Results**:  [artemis/fileman/demo_experiments.py](/artemis/fileman/demo_experiments.py)  

 ![](https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Tizian_015.jpg/800px-Tizian_015.jpg)

## Using Browser-plotting
After installing, you should have a file ~/.artemis rc.  To use the web backend, edit the `backend` field to `matplotlib-web`.  To try it you can run [/artemis/plotting/demo_dbplot.py](/artemis/plotting/demo_dbplot.py)  



