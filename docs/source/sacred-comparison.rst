
+------------------------------------------------+--------------------------------------------+---------------------------------------------------------------------------+
| **Script to train an SVM on the iris dataset** | **The same script as a Sacred experiment** | **And as an Artemis experiment**                                          |
+------------------------------------------------+--------------------------------------------+---------------------------------------------------------------------------+
| .. code:: python                               | .. code:: python                           | .. code:: python                                                          | 
|                                                |                                            |                                                                           |
|  from numpy.random import permutation          |   from numpy.random import permutation     |   from numpy.random import permutation                                    |
|  from sklearn import svm, datasets             |   from sklearn import svm, datasets        |   from sklearn import svm, datasets                                       |
|                                                |   from sacred import Experiment            |   from artemis.experiments import experiment_function                     |
|                                                |   ex = Experiment('iris_rbf_svm')          |                                                                           |
|                                                |                                            |                                                                           |
|                                                |   @ex.config                               |                                                                           |
|                                                |   def cfg():                               |                                                                           |
|  C = 1.0                                       |     C = 1.0                                |                                                                           |
|  gamma = 0.7                                   |     gamma = 0.7                            |                                                                           |
|                                                |                                            |                                                                           |
|                                                |   @ex.automain                             |   @experiment_function                                                    |
|                                                |   def run(C, gamma):                       |   def demo_iris_svm(C=1.0, gamma=0.7):                                    |
|  iris = datasets.load_iris()                   |     iris = datasets.load_iris()            |       iris = datasets.load_iris()                                         |
|  perm = permutation(iris.target.size)          |     per = permutation(iris.target.size)    |       perm = permutation(iris.target.size)                                |
|  iris.data = iris.data[perm]                   |     iris.data = iris.data[per]             |       iris.data = iris.data[perm]                                         |
|  iris.target = iris.target[perm]               |     iris.target = iris.target[per]         |       iris.target = iris.target[perm]                                     |
|  clf = svm.SVC(C, 'rbf', gamma=gamma)          |     clf = svm.SVC(C, 'rbf', gamma=gamma)   |       clf = svm.SVC(C, 'rbf', gamma=gamma)                                |
|  clf.fit(iris.data[:90],                       |     clf.fit(iris.data[:90],                |       clf.fit(iris.data[:90], iris.target[:90])                           |
|          iris.target[:90])                     |             iris.target[:90])              |       return clf.score(iris.data[90:], iris.target[90:])                  |
|  print(clf.score(iris.data[90:],               |     return clf.score(iris.data[90:],       |                                                                           |
|                  iris.target[90:]))            |                      iris.target[90:])     |   if __name__ == '__main__':                                              |
|                                                |                                            |       demo_iris_svm.browse()                                              |
+------------------------------------------------+--------------------------------------------+---------------------------------------------------------------------------+
