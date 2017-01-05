from artemis.general.display import IndentPrint, CaptureStdOut


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

if __name__ == '__main__':
    test_indent_print()
