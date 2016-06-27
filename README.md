# Artemis

Artemis is a plotting library which can use either Bokeh or Matplotlib as a backend.

It is made mainly for creating live plots to monitor things.

To import:

```
cd ~/my/projects/folder
git clone https://github.com/quva-lab/artemis.git
cd artemis
```

To use artemis from within your project: (You probably want to do this in a virtualenv).
```
pip install -e git+http://github.com/QUVA-Lab/artemis.git#egg=artemis 
```
You'll need to make sure your virtualenv already has artemis dependencies (numpy, matplotlib or bokeh).

Or if your project has a `requirements.txt` file, add the line `-e git+http://github.com/QUVA-Lab/artemis.git#egg=artemis` before running `pip install -r requirements.txt`

During installation, the file `.artemisrc` is created in your home directory. In it you can specify the backend to use

![](https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Tizian_015.jpg/800px-Tizian_015.jpg)
