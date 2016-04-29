from artemis.general.newline_writer import TextWrappingPrinter

__author__ = 'peter'


def test_newline_writer():

    text = """
fdsafdsafdsa gfsdgbfd fd svfdsb vfdsvfds dsacdascdas  vfdsbfdsbfdsbfdsbfdsbd

gfdsgfdsgfds

gfdsgfdsgfdsgfdgfdsgfdsgfdsbfdsgfds
"""

    t = TextWrappingPrinter()
    t.write(text)


if __name__ == '__main__':

    test_newline_writer()
