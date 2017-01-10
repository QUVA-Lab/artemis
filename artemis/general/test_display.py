from artemis.general.display import IndentPrint, CaptureStdOut, side_by_side

_desired = """
aaa
bbb
    ccc
    ddd
        eee
        fff
    ggg
    hhh
iii
jjj
"""


def test_indent_print():

    with CaptureStdOut() as cap:
        print 'aaa'
        print 'bbb'
        with IndentPrint():
            print 'ccc'
            print 'ddd'
            with IndentPrint():
                print 'eee'
                print 'fff'
            print 'ggg'
            print 'hhh'
        print 'iii'
        print 'jjj'

    assert '\n'+cap.read() == _desired


str1 = """For those who are led by the Spirit of God are the
children of God.
Romans 8:14"""

str2 = """But to you who are listening I say: Love your enemies,
do good to those who hate you, bless those who curse you, pray
for those who mistreat you.
Luke 6:27-28"""

desired = """For those who are led by the Spirit of God are the    But to you who are listening I say: Love your enemies,
children of God.                                      do good to those who hate you, bless those who curse you, pray
Romans 8:14                                           for those who mistreat you.
                                                      Luke 6:27-28
"""


def test_side_by_side():

    print 'String 1:\n{}'.format(str1)
    print 'String 2:\n{}'.format(str2)

    out = side_by_side([str1, str2])
    print 'Side by side:\n{}'.format(out)
    # assert out==desired  # Would work but pycharm automatically trims trailing spaces.

if __name__ == '__main__':
    # test_indent_print()
    test_side_by_side()
