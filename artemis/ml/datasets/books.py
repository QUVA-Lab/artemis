from artemis.fileman.file_getter import get_file
import re

__author__ = 'peter'


def read_book(code, max_characters = None):

    text = {
        'bible': read_the_bible,
        'fifty_shades_of_grey': read_fifty_shades_of_grey
        }[code](max_characters=max_characters)
    return text


def read_the_bible(max_characters = None):
    """
    Returns the King James Bible as a single string.
    Thanks to Janel (http://janelwashere.com/pages/bible_daily_reading.html) for compiling it.
    :param max_characters: You have the option to truncate it to a length of max_characters
        (If you're Jewish, for instance)
    :return: A string.
    """

    filename = get_file(
        relative_name = 'data/king_james_bible.txt',
        url = 'http://janelwashere.com/files/bible_daily.txt',
        )

    with open(filename) as f:
        text = f.read(-1 if max_characters is None else max_characters)

    return text


def read_fifty_shades_of_grey(max_characters = None):
    """
    Returns Fifty Shades of Gray, by EL James.
    :param max_characters:
    :return:
    """

    filename = get_file(
        relative_name = 'data/fifty_shades_of_grey.txt',
        url = None
        )

    with open(filename) as f:
        text = f.read(-1 if max_characters is None else max_characters)

    # Need to remove some weird non-ascii stuff.
    # http://stackoverflow.com/questions/20078816/replace-non-ascii-characters-with-a-single-space
    text = re.sub(r'[^\x00-\x7F]+',' ', text)

    return text


if __name__ == '__main__':

    book = 'fifty_shades_of_grey'
    n_characters = 10000

    print(read_book(book, n_characters))