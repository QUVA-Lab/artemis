from numpy.random import permutation
from sklearn import svm, datasets
from artemis.experiments import experiment_function


@experiment_function
def demo_iris_svm(C=1.0, gamma=0.7):

    iris = datasets.load_iris()
    perm = permutation(iris.target.size)
    iris.data = iris.data[perm]
    iris.target = iris.target[perm]
    clf = svm.SVC(C, 'rbf', gamma=gamma)
    clf.fit(iris.data[:90], iris.target[:90])
    return clf.score(iris.data[90:], iris.target[90:])


X=demo_iris_svm.add_variant(C=0.5)
X.add_variant(gamma=0.9)

if __name__ == '__main__':
    demo_iris_svm.browse()
