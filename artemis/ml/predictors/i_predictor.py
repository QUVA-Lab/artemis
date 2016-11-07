from abc import abstractmethod

import numpy as np

from artemis.ml.tools.processors import OneHotEncoding


__author__ = 'peter'


class IPredictor(object):

    @abstractmethod
    def train(self, input_data, target_data):
        """
        :param input_data:
        :param target_data:
        :return:
        """

    @abstractmethod
    def predict(self, input_data):
        """
        :return: The output given the input data
        """


class CategoricalPredictor(IPredictor):
    """
    A wrapper that transforms a predictor that outputs a vector into
    a predictor that outputs an integer "category" label.
    """
    def __init__(self, predictor, n_categories = None, argmax_outputs = True):
        self._predictor = predictor
        self._n_categories = n_categories
        self._encoder = None if n_categories is None else OneHotEncoding(n_categories)
        self._argmax_outputs = argmax_outputs

    def train(self, input_data, target_data):
        if self._encoder is None:
            raise Exception('If you call train before predict, you must provide the number of categories.')
        new_target_data = self._encoder(target_data)
        return self._predictor.train(input_data, new_target_data)

    def predict(self, input_data):
        out = self._predictor.predict(input_data)
        if self._argmax_outputs:
            if self._encoder is None:
                assert out.ndim==2
                self._n_categories = out.shape[1]
                self._encoder = OneHotEncoding(self._n_categories)
            return np.argmax(out, axis = 1)
        else:
            return out
