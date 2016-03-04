# ML Template Repo

This is a useful template with which you can start Python Machine Learning Projects.

It includes a setup script which will install the basics (numpy, scipy, matplotlib, ipython notebook, scikit-learn, pytest).

To use:

## Step 1: Create your new repo

Create your own repo on github.  Lets say it's called my-repo, and your name is helga

```
cd ~/my/projects/folder
git clone https://github.com/helga/my-repo.git
cd my-repo
```

## Step 2: Merge in the template

Now, to initialize the repo with this templayte, you can copy the following lines into your terminal:

```
git remote add ml_template git@github.com:petered/ml-template.git
git fetch ml_template
git merge -s ours --no-commit ml_template/master
git read-tree --prefix=ml_template/ -u ml_template/master
git commit -m "Started repo from ML template."
```

## Step 3: Customize and setup

Take a look at `requirements.txt`.  A list of default requirements (numpy, matplotlib, etc) are included.  Add or remove requirements as necessary.  

Now run `setup.sh`.  This should run without error and set up your virtualenv.

## Step 4: (Optional) PyCharm setup

Download (PyCharm).  Open it:
`File > New Project`, select your the folder of your repo.  For the interpreter, select `</my/projects/folder>/venv/bin/activate`.  You should not be ready to go.
