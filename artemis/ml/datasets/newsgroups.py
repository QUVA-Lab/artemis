import numpy as np

from artemis.fileman.file_getter import get_file
from artemis.general.should_be_builtins import memoize
from artemis.ml.datasets.datasets import DataSet


__author__ = 'peter'


@memoize
def get_20_newsgroups_dataset(filter_most_common = 2000, numeric = False, shuffling_seed = 1234, bag_of_words = False, count_scaling = None):
    """
    The 20 newsgroups dataset.  In this dataset, you try to predict the topic of a forum from the words contained in
    posts in the forums.

    Words have been preprocessed to the "stemmed" version, as explained on the website:
    http://ana.cachopo.org/datasets-for-single-label-text-categorization

    :param filter_most_common: Can be:
        None: Don't filter out words
        int N: Filter out words that are not in the N most common workds
        (int N, int M): Filter out words that are not between the Nth and Mth most common words.
    :param numeric: Convert everything from words to numbers
    :param shuffling_seed: Random seed for shuffling (you want to shuffle, because everything's sorted by topic)
    :param bag_of_words: Return count vectors for each word
    :param count_scaling: If using bag_of_words, apply the transformation:
        vector = log(1+word_counts)
        To generate the input data (this scaling makes it more suitable for some types of classifiers).
    :return: A DataSet object
    """

    training_set_file = get_file(
        relative_name = 'data/20ng-train-stemmed.txt',
        url = 'http://ana.cachopo.org/datasets-for-single-label-text-categorization/20ng-train-stemmed.txt'
        )

    test_set_file = get_file(
        relative_name = 'data/20ng-test-stemmed.txt',
        url = 'http://ana.cachopo.org/datasets-for-single-label-text-categorization/20ng-test-stemmed.txt'
        )

    train_words, train_labels = _read_formatted_file(training_set_file)
    test_words, test_labels = _read_formatted_file(test_set_file)

    # Shuffle it up...
    rng = np.random.RandomState(shuffling_seed)
    train_words, train_labels =_shuffle((train_words, train_labels), rng)
    test_words, test_labels =_shuffle((test_words, test_labels), rng)

    # Filter out most-common-but-not-too-common-words
    all_train_words = np.concatenate(train_words)
    filtered_vocab, counts = _find_most_common(all_train_words, filter_most_common)
    train_words = _filter_lists_of_words(train_words, filtered_vocab)
    test_words = _filter_lists_of_words(test_words, filtered_vocab)

    if numeric or bag_of_words:
        train_ixs_list = _list_of_posts_to_list_of_ixs(train_words, filtered_vocab)
        test_ixs_list = _list_of_posts_to_list_of_ixs(test_words, filtered_vocab)
        label_vocab = {lab: i for i, lab in enumerate(np.unique(train_labels))}
        train_labels = _words_to_ints(train_labels, label_vocab)
        test_labels = _words_to_ints(test_labels, label_vocab)

        if bag_of_words:
            train_counts = _list_of_ixs_to_count_matrix(train_ixs_list, n_words=len(filtered_vocab))
            test_counts = _list_of_ixs_to_count_matrix(test_ixs_list, n_words=len(filtered_vocab))
            if count_scaling == 'log':
                train_counts = np.log(1+train_counts)
                test_counts = np.log(1+test_counts)
            return DataSet.from_xyxy(training_inputs = train_counts, training_targets = train_labels, test_inputs = test_counts, test_targets = test_labels)
        else:
            return DataSet.from_xyxy(training_inputs = train_ixs_list, training_targets = train_labels, test_inputs = test_ixs_list, test_targets = test_labels)
    else:
        return DataSet.from_xyxy(training_inputs = train_words, training_targets = train_labels, test_inputs = test_words, test_targets = test_labels)



def _read_formatted_file(file_relative_path):

    with open(get_file(file_relative_path)) as f:
        text = f.read()
    pairs = [line.split('\t') for line in text.split('\n')[:-1]]
    labels = [group for group, _ in pairs]
    words = [sentence.split(' ') for _, sentence in pairs]
    return words, labels


def _find_most_common(elements, n_most_common):

    unique_elements, counts = np.unique(elements, return_counts=True)
    # ixs = np.argpartition(-counts, kth = n_most_common)
    ixs = np.argsort(counts)[::-1]
    if isinstance(n_most_common, int):
        most_common_element_ixs = ixs[:n_most_common]
    else:
        assert isinstance(n_most_common, tuple) and len(n_most_common) == 2, 'eh?'
        start, stop = n_most_common
        most_common_element_ixs = ixs[start:stop]
    most_common_elements = unique_elements[most_common_element_ixs]
    return most_common_elements, counts[most_common_element_ixs]


def _filter_words(word_list, filter_set):
    if not isinstance(filter_set, set):
        filter_set = set(filter_set)
    return [w for w in word_list if w in filter_set]


def _filter_lists_of_words(lists_of_words, filter_set):
    return np.array([_filter_words(word_list, filter_set) for word_list in lists_of_words])


def _words_to_ints(word_list, lookup):
    return np.array([lookup[w] for w in word_list])


def _list_of_posts_to_list_of_ixs(list_of_posts, vocabulary):
    div_ixs = np.cumsum([len(post) for post in list_of_posts])[:-1]
    all_filtered_words = np.concatenate(list_of_posts)
    ixs = np.zeros(len(all_filtered_words), dtype = int)
    for i, w in enumerate(vocabulary):
        ixs[all_filtered_words==w] = i
    list_of_ixs = np.split(ixs, div_ixs)
    return np.array(list_of_ixs)


def _list_of_ixs_to_count_matrix(list_of_ixs, n_words):
    n_samples = len(list_of_ixs)
    counts = np.zeros((n_samples, n_words), dtype = int)
    for c, ixs in zip(counts, list_of_ixs):
        np.add.at(c, ixs, 1)
    return counts


def _shuffle(arrays, rng):
    n_samples = len(arrays[0])
    assert all(n_samples == len(arr) for arr in arrays)
    ixs = np.array(rng.permutation(n_samples))
    return tuple(np.array(arr)[ixs] for arr in arrays)


if __name__ == '__main__':

    data = get_20_newsgroups_dataset(numeric=False, filter_most_common = (100, 500), bag_of_words=False)
    for _, (inputs, ), (targets, ) in data.training_set.shorten(20).minibatch_iterator(minibatch_size = 1):
        print('%s: %s' % (targets, inputs))
