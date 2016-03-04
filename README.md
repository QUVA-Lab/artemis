# ML Template Repo

This is a useful template with which you can start Python Machine Learning Projects.

It includes a setup script which will install the basics (numpy, scipy, matplotlib, ipython notebook, scikit-learn, pytest).

To use:

Create your own repo on github.  Lets say it's called my-repo, and your name is helga

```
cd ~/my/projects/folder
git clone https://github.com/helga/my-repo.git
cd my-repo

```


git remote add ml_template git@github.com:petered/ml-template.git
git fetch ml_template
git merge -s ours --no-commit ml_template/master
git read-tree --prefix=ml_template/ -u ml_template/master
git commit -m "Started repo from ML template."
