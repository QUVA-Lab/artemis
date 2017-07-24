import numpy as np
from artemis.experiments.experiment_record import experiment_function
from artemis.general.mymath import softmax
from artemis.ml.datasets.mnist import get_mnist_dataset
from artemis.ml.predictors.logistic_regressor import onehot
from artemis.ml.tools.iteration import minibatch_index_info_generator
from matplotlib import pyplot as plt


class OnlineLogisticRegressor:

    def __init__(self, n_in, n_out, learning_rate = 0.01):
        self.w = np.zeros((n_in, n_out))
        self.learning_rate = learning_rate

    def train(self, x, targ):  # x: (n_samples, n_in), targ: (n_samples, ) target label
        y_pred = self.predict(x)
        dl_dy = y_pred - onehot(targ, n_classes=y_pred.shape[1])
        self.w -= self.learning_rate * (x.T.dot(dl_dy))

    def predict(self, x):  # x: (n_samples, n_in)
        return softmax(x.dot(self.w), axis=1)


@experiment_function
def demo_mnist_logreg(minibatch_size=20, learning_rate=0.01, max_training_samples = None, n_epochs=10, test_epoch_period=0.2):

    x_train, y_train, x_test, y_test = get_mnist_dataset(flat=True, n_training_samples=max_training_samples).xyxy

    predictor = OnlineLogisticRegressor(n_in=784, n_out=10, learning_rate=learning_rate)

    # Train and periodically record scores.
    epoch_scores = []
    for ix, iteration_info in minibatch_index_info_generator(n_samples = len(x_train), minibatch_size=minibatch_size, n_epochs=n_epochs, test_epochs=('every', test_epoch_period)):
        if iteration_info.test_now:
            training_error = 100*(np.argmax(predictor.predict(x_train), axis=1)==y_train).mean()
            test_error = 100*(np.argmax(predictor.predict(x_test), axis=1)==y_test).mean()
            print 'Epoch {epoch}: Test Error: {test}%, Training Error: {train}%'.format(epoch=iteration_info.epoch, test=test_error, train=training_error)
            epoch_scores.append((iteration_info.epoch, training_error, test_error))
        predictor.train(x_train[ix], y_train[ix])

    # Plot
    plt.figure()
    epochs, training_costs, test_costs = zip(*epoch_scores)
    plt.plot(epochs, np.array([training_costs, test_costs]).T)
    plt.xlabel('Epoch')
    plt.ylabel('% Error')
    plt.legend(['Training Error', 'Test Error'])
    plt.title("Learning Curve")
    plt.ylim(80, 100)
    plt.ion()  # Don't hang on plot
    plt.show()

    return {'train': epoch_scores[-1][1], 'test': epoch_scores[-1][2]}


demo_mnist_logreg.add_variant(minibatch_size=10)
demo_mnist_logreg.add_variant(learning_rate=0.1)
X=demo_mnist_logreg.add_variant(learning_rate=0.001)
X.add_variant(minibatch_size=10)

if __name__ == '__main__':
    demo_mnist_logreg.browse()
