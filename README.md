# Artemis

 ![The deer represents dull, repetitive coding tasks, and Artemis represents Artemis.  As you can see, once Artemis comes along, the future is not bright for dull, repetitive coding tasks.](https://raw.githubusercontent.com/petered/data/master/images/artemis.jpeg)

Artemis is a collection of tools that make it easier to run experiments in Python.  These include:

- An easy-to-use system for making live plots, to monitor variables in a running experiment.
- A browser-based plotter for displaying live plots.
- A framework for defining experiments and logging their results (text output and figures) so that they can be reviewed later and replicated easily.
- A system for downloading/caching files, to a local directory, so the same code can work on different machines.

## Tutorials

- [A simple example showing how you can use Artemis to organize your Machine Learning experiments](https://github.com/QUVA-Lab/artemis/blob/master/artemis/examples/demo_mnist_logreg.py)
- [Step-by-step tutorial on using Artemis to organize your Experiments](https://rawgit.com/petered/data/master/gists/experiment_tutorial.html)

## Quick Demos

**Live Plotting**: [/artemis/plotting/demo_dbplot.py](/artemis/plotting/demo_dbplot.py)  

**Recording Experiment Results**:  [artemis/experiments/demo_experiments.py](/artemis/experiments/demo_experiments.py)  

**A demo repo showing how to use Artemis from your code**: [https://github.com/QUVA-Lab/demo_repo](https://github.com/QUVA-Lab/demo_repo)


## Installation
To use artemis from within your project, use the following to install Artemis and its dependencies: (You probably want to do this in a virtualenv with the latest version of pip - run `virtualenv venv; source venv/bin/activate; pip install --upgrade pip;` to make one and enter it).


**Option 1: (Recommended) Install as source (allows you to edit Artemis).**

```
pip install -e git+http://github.com/QUVA-Lab/artemis.git#egg=artemis 
```
This will install it in `(virtual env or system python root)/src/artemis`.  You can edit the code and submit pull requests to our git repo.  To install with the optional [remote plotting](https://github.com/QUVA-Lab/artemis/blob/master/artemis/remote/README.md) mode enabled, add the `[remote_plotting]` option, as in: `pip install -e git+http://github.com/QUVA-Lab/artemis.git#egg=artemis[remote_plotting]`


**Option 2: Simple install:**

```
pip install artemis-ml
```

**Verifying that it works**

To verify that the plotting works, run:
```
python -m artemis.plotting.demo_dbplot
```
A bunch of plots should come up and start updating live. 


<!--- To verify that the installation worked, go:
```
cd venv/src/artemis
py.test
```
All tests should pass.
(pytest for some reason cant find modules when you do this alone)--->
Note: During installation, the settings file `.artemisrc` is created in your home directory. In it you can specify the plotting backend to use, and other settings.

# Info on Extras


**Using Browser-plotting**
After installing, you should have a file ~/.artemis rc.  To use the web backend, edit the `backend` field to `matplotlib-web`.  To try it you can run [/artemis/plotting/demo_dbplot.py](/artemis/plotting/demo_dbplot.py)  
