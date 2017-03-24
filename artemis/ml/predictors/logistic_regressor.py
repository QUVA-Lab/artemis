from artemis.general.mymath import softmax
from artemis.ml.predictors.i_predictor import IPredictor
import numpy as np


class LogisticRegressor(IPredictor):

    def __init__(self, w, learning_rate=0.1):
        self.w = w
        self.learning_rate = learning_rate

    def predict(self, x):
        """
        :param x: An (n_samples, n_inputs) input
        :return: An (n_samples, n_classes) class probability
        """
        return softmax(x.dot(self.w), axis=1)

    def train(self, x, y):
        """
        :param x: An (n_samples, n_inputs) input
        :param y: An integer class label
        :return:
        """
        probs = self.predict(x)
        d_l_d_u = probs - onehot(y, n_classes=self.w.shape[1])
        self.w -= self.learning_rate * x.T.dot(d_l_d_u)

    @staticmethod
    def from_init(n_in, n_out, **kwargs):
        return LogisticRegressor(np.zeros((n_in, n_out)), **kwargs)


def onehot(labels, n_classes):
    """
    Turn a vector of labels into a onehot-encoding.
    :param labels:
    :param n_classes:
    :return:
    """
    values = np.zeros((len(labels), n_classes))
    values[np.arange(len(values)), labels] = 1
    return values