from scipy.stats.stats import mode
import numpy as np

from artemis.ml.predictors.i_predictor import IPredictor


__author__ = 'peter'


class MockPredictor(IPredictor):

    def __init__(self, prediction_function):
        self._prediction_function = prediction_function

    def fit(self, input_data, target_data):
        pass

    def predict(self, input_data):
        return self._prediction_function(input_data)


class MostFrequentPredictor(IPredictor):

    def train(self, input_data, target_data):
        (self._most_frequent_value, ), _ = mode(target_data, axis = 0)
        self._target_type = target_data.dtype

    def predict(self, input_data):
        return np.zeros((len(input_data), )+self._most_frequent_value.shape, dtype = self._most_frequent_value.dtype)+self._most_frequent_value


class DistributionPredictor(IPredictor):

    def train(self, input_data, target_data):
        distro = np.sum(target_data, axis = 0).astype(float)
        self._distro = distro/np.sum(distro)

    def predict(self, input_data):
        return np.zeros((len(input_data), )+self._distro.shape, dtype = self._distro.dtype)+self._distro
