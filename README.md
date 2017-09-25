# Artemis

 ![The deer represents dull, repetitive coding tasks, and Artemis represents Artemis.  As you can see, once Artemis comes along, the future is not bright for dull, repetitive coding tasks.](https://raw.githubusercontent.com/petered/data/master/images/artemis.jpeg)

Artemis is a collection of tools that make it easier to run experiments in Python.  These include:

- An easy-to-use system for making live plots, to monitor variables in a running experiment.
- A browser-based plotter for displaying live plots.
- A [framework for defining experiments](http://artemis-ml.readthedocs.io/en/latest/experiments.html) and logging their results (text output and figures) so that they can be reviewed later and replicated easily.
- A system for downloading/caching files, to a local directory, so the same code can work on different machines.

For examples of how to use artemis, read the [Artemis Documentation](http://artemis-ml.readthedocs.io)

## Installation

**Note: Artemis was build for Python 2.7.  The Master branch now supports Python 3, so if you want to use on Python 3, you can directly install from master: `pip install git+http://github.com/QUVA-Lab/artemis.git#egg=artemis`.  A more official release supporting Python 3 is coming soon.**

To use artemis from within your project, use the following to install Artemis and its dependencies: (You probably want to do this in a virtualenv with the latest version of pip - run `virtualenv venv; source venv/bin/activate; pip install --upgrade pip;` to make one and enter it).


**Option 1: Simple install:**

```
pip install artemis-ml
```

**Option 2: Install as source.**

```
pip install -e git+http://github.com/QUVA-Lab/artemis.git#egg=artemis 
```
This will install it in `(virtual env or system python root)/src/artemis`.  You can edit the code and submit pull requests to our git repo.  To install with the optional [remote plotting](https://github.com/QUVA-Lab/artemis/blob/master/artemis/remote/README.md) mode enabled, add the `[remote_plotting]` option, as in: `pip install -e git+http://github.com/QUVA-Lab/artemis.git#egg=artemis[remote_plotting]`

**(Note, this doesn't work if you have Anaconda installed, as it does not work with the `-e` option)**.  Use `pip install artemis-ml` in this case instead.



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

Now that you have Artemis installed, see [this Tutorial](http://artemis-ml.readthedocs.io/en/latest/experiments.html) on how to use Artemis to organize your experiments.
