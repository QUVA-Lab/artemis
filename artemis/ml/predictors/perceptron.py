import numpy as np

from artemis.ml.predictors.i_predictor import IPredictor


class Perceptron(IPredictor):

    def __init__(self, w, b = None, alpha = 0.1):
        self._w = w
        self._b = np.zeros((1, w.shape[1])) if b is None else b
        self._alpha = alpha

    def train(self, x, target_data):
        y = (self.predict(x) > 0).astype(int)
        self._w += self._alpha*x.T.dot(target_data-y)
        self._b += self._alpha*np.sum(target_data-y, axis = 0)

    def predict(self, x):
        return x.dot(self._w)+self._b

    def to_categorical(self, **kwargs):
        from artemis.ml.predictors.i_predictor import CategoricalPredictor
        return CategoricalPredictor(self, **kwargs)
