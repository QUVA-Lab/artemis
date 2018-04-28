import numpy as np
from artemis.experiments import experiment_function
from artemis.general.mymath import softmax
from artemis.ml.datasets.mnist import get_mnist_dataset
from artemis.ml.predictors.logistic_regressor import onehot
from artemis.ml.tools.iteration import minibatch_index_info_generator
from matplotlib import pyplot as plt

"""
This demo shows how you can use the Artemis Experiment Framework.

Using the Experiment framework (via the UI) requires only 2 additional lines of code:
1) Add the (@experiment_function) decorator onto your main function (lets call it my_main_fcn).
2) Open up the User Interface by calling my_main_funcion.browse()

There are two ways to interact with the experiment framework:
1) Through the User Interface.
2) Through the experiment API.

In the bottom of this file, you an see example code for running either the UI or API.  You can try the different versions
by either running this file as "python demo_mnist_logreg.py ui" or "python demo_mnist_logreg.py api"
"""


class OnlineLogisticRegressor:

    def __init__(self, n_in, n_out, learning_rate = 0.01):
        self.w = np.zeros((n_in, n_out))
        self.learning_rate = learning_rate

    def train(self, x, targ):  # x: (n_samples, n_in), targ:
        """
        :param x: A (n_samples, n_in) input array
        :param targ: A (n_samples, ) array of interger target labels
        """
        y_pred = self.predict(x)
        dl_dy = y_pred - onehot(targ, n_classes=y_pred.shape[1])
        self.w -= self.learning_rate * (x.T.dot(dl_dy))

    def predict(self, x):  # x: (n_samples, n_in)
        """
        :param x: A (n_samples, n_in) input array
        :return: A (n_samples, n_out) array of output probabilities for each sample.
        """
        return softmax(x.dot(self.w), axis=1)


@experiment_function  # This decorator turns the function into an Experiment object.
def demo_mnist_logreg(minibatch_size=20, learning_rate=0.01, max_training_samples = None, n_epochs=10, test_epoch_period=0.2):
    """
    Train a Logistic Regressor on the MNIST dataset, report training/test scores throughout training, and return the
    final scores.
    """

    x_train, y_train, x_test, y_test = get_mnist_dataset(flat=True, n_training_samples=max_training_samples).xyxy

    predictor = OnlineLogisticRegressor(n_in=784, n_out=10, learning_rate=learning_rate)

    # Train and periodically record scores.
    epoch_scores = []
    for ix, iteration_info in minibatch_index_info_generator(n_samples = len(x_train), minibatch_size=minibatch_size, n_epochs=n_epochs, test_epochs=('every', test_epoch_period)):
        if iteration_info.test_now:
            training_error = 100*(np.argmax(predictor.predict(x_train), axis=1)==y_train).mean()
            test_error = 100*(np.argmax(predictor.predict(x_test), axis=1)==y_test).mean()
            print('Epoch {epoch}: Test Error: {test}%, Training Error: {train}%'.format(epoch=iteration_info.epoch, test=test_error, train=training_error))
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

    return {'train': epoch_scores[-1][1], 'test': epoch_scores[-1][2]}  # Return final scores


demo_mnist_logreg.add_variant(learning_rate=0.1)                        # You can define a variant by providing keyword arguments to the function
X=demo_mnist_logreg.add_variant(learning_rate=0.001)                    # You can assign a variant to a variable ...
X.add_variant(minibatch_size=10)                                        # ... which lets you make variants of variants
demo_mnist_logreg.add_variant('small-set', max_training_samples=1000)   # You can optionally give variants names to refer to via experiment.get_variant(name)


if __name__ == '__main__':
    import sys
    demo_version = sys.argv[1] if len(sys.argv) > 1 else 'ui'

    if demo_version == 'ui':
        # Open the experiment browser UI, from where you can run and view experiments:
        demo_mnist_logreg.browse(raise_display_errors=False, display_format='nested')
        # Commands you can try (or press h to see a list of all commands):
        # run all          # Runs all experiments
        # show 2-3         # Show the output and figures from experiments 2 and 3
        # delete all       # Delete all records
        # delete old       # Delete all records except the newest from each experiment
        # compare 1-3      # Generate a table comparing the arguments and results of experiments 1,2,3

    elif demo_version == 'api':
        # Demonstrate some commands in the api.  Here we will collect the results from several experiments and compare them.

        # Collect a list of demo_mnist_logreg and all its variants.
        exp_list = demo_mnist_logreg.get_all_variants(include_self=True)

        # Run all experiments that have not yet been run to completion:
        for ex in exp_list:
            if len(ex.get_records(only_completed=True))==0:
                ex.run()

        # Print a table containing the full list of arguments and results, and show all figures.
        for ex in exp_list:
            record = ex.get_latest_record(only_completed=True)   # Get an object representing the latest completed run of the experiment
            args = record.get_args()  # This is a list of tuples of (arg_name, arg_value)
            result = record.get_result()  # This is the return value {'train': train score, 'test': test_score}
            (fig, ) = record.load_figures()
            fig.gca().set_title(ex.name)
            print(ex.name.ljust(60) \
                  + ' '.join('{}:{}'.format(arg_name, arg_value).ljust(26) for arg_name, arg_value in args.items()) \
                  + '  ||  ' + ' '.join('{}:{}'.format(subset, score).rjust(15) for subset, score in result.items()))
        plt.show()
    else:
        raise NotImplementedError('No Demo Version {}'.format(demo_version))

