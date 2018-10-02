# Artemis

 ![The deer represents dull, repetitive coding tasks, and Artemis represents Artemis.  As you can see, once Artemis comes along, the future is not bright for dull, repetitive coding tasks.](https://raw.githubusercontent.com/petered/data/master/images/artemis.jpeg)

Artemis is a collection of tools that make it easier to run experiments in Python.  These include:

### A [framework for defining experiments](http://artemis-ml.readthedocs.io/en/latest/experiments.html) and logging their results (text output and figures) so that they can be reviewed later and replicated easily.

e.g.
```
from artemis.experiments import experiment_function

@experiment_function  # Decorate your main function to turn it into an Experiment object
def multiply_3_numbers(a=1, b=2, c=3):
    answer = a*b*c
    print('{} x {} x {} = {}'.format(a, b, c, answer))
    return answer
    
record = multiply_3_numbers.run()  # Run experiment, save arguments, console output, return value to disk
print(record.get_log())  # Pring console output of last run      
print(record.get_result())  # Print return value of last run
ex = multiply_3_numbers.add_variant(a=4, b=5)  # Make a new experiment with different paremters.
multiply_3_numbers.browse()  # Open a UI to browse through all experiments and results.
```

### An easy-to-use system for making live plots, to monitor variables in a running experiment.

e.g.
```
from artemis.plotting.db_plotting import dbplot
import numpy as np
for t in np.linspace(0, 10, 100):
    dbplot(np.sin(t), 'sin of the times')  # Detects data type and makes appropriate plot
    dbplot(np.sin(-4*t+sum(xi**2 for xi in np.meshgrid(*[np.linspace(-20, 20, 200)]*2))), "Instaaaaall Arrrteeeemis")
```
(this can also be set up in the browser for remote live plotting)

### Functions for easy download and loading of numerical data.

e.g.
```
from artemis.plotting.db_plotting import dbplot
from artemis.fileman.smart_io import smart_load
img = smart_load('https://cdn.britannica.com/s:700x450/54/13354-004-2F9AE1B2.jpg')  # Detects data type and loads into numpy array
dbplot(im, 'artemis', hang=True)
```

### A system for downloading/caching files to a local directory, so the same code can work on different machines.

```
from artemis.fileman.file_getter import get_file
import os
local_path = get_file(url = 'https://cdn.britannica.com/s:700x450/54/13354-004-2F9AE1B2.jpg')  # Downloads first time, caches after 
print('Image "{}" has a size of {:.2g}kB'.format(local_path, os.path.getsize(local_path)/1000.))
```
For more examples of how to use artemis, read the [Artemis Documentation](http://artemis-ml.readthedocs.io)


## Installation

**As of release 2.0.0 on November 13, 2017, Artemis now supports Python 3**

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
