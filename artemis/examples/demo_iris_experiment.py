# from artemis.experiments.decorators import experiment_function
# from numpy.random import permutation
# from sklearn import svm, datasets
#
#
# @experiment_function
# def demo_iris_svm(C=1.0, gamma=0.7):
#
#     iris = datasets.load_iris()
#     perm = permutation(iris.target.size)
#     iris.data = iris.data[perm]
#     iris.target = iris.target[perm]
#     clf = svm.SVC(C, 'rbf', gamma=gamma)
#     clf.fit(iris.data[:90], iris.target[:90])
#     return clf.score(iris.data[90:], iris.target[90:])
#
#
# X=demo_iris_svm.add_variant(C=0.5)
# X.add_variant(gamma=0.9)
#
# if __name__ == '__main__':
#     demo_iris_svm.browse()


from numpy.random import permutation
from sklearn import svm, datasets
from sacred import Experiment
ex = Experiment('iris_rbf_svm')

@ex.config
def cfg():
  C = 1.0
  gamma = 0.7

@ex.automain
def run(C, gamma):
  iris = datasets.load_iris()
  per = permutation(iris.target.size)
  iris.data = iris.data[per]
  iris.target = iris.target[per]
  clf = svm.SVC(C, 'rbf', gamma=gamma)
  clf.fit(iris.data[:90],
          iris.target[:90])
  return clf.score(iris.data[90:],
                   iris.target[90:])