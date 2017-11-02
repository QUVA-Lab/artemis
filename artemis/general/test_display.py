import textwrap
from collections import OrderedDict

from artemis.general.display import IndentPrint, CaptureStdOut, side_by_side, DocumentWrapper, deepstr, \
    str_with_arrayopts, surround_with_header, sensible_str, format_duration
import numpy as np

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
        print('aaa')
        print('bbb')
        with IndentPrint():
            print('ccc')
            print('ddd')
            with IndentPrint():
                print('eee')
                print('fff')
            print('ggg')
            print('hhh')
        print('iii')
        print('jjj')

    assert '\n'+cap.read() == _desired


str1 = '' + \
    'For those who are led by the Spirit of God are the\n' \
    'children of God.\n' \
    'Romans 8:14'

str2 = '' \
       'But to you who are listening I say: Love your enemies,\n' \
       'do good to those who hate you, bless those who curse you, pray\n' \
       'for those who mistreat you.\n' \
       'Luke 6:27-28'

desired = '' \
    'For those who are led by the Spirit of God are the    But to you who are listening I say: Love your enemies,        \n' \
    'children of God.                                      do good to those who hate you, bless those who curse you, pray\n' \
    'Romans 8:14                                           for those who mistreat you.                                   \n' \
    '                                                      Luke 6:27-28                                                  '


def test_side_by_side():

    print('String 1:\n{}'.format(str1))
    print('String 2:\n{}'.format(str2))

    out = side_by_side([str1, str2])
    print('Side by side:\n{}'.format(out))
    assert out==desired  # Would work but pycharm automatically trims trailing spaces from the strings defined av


def test_document_wrapper():
    """
    Document Wrapper is deals with wrapping text with new lines already present.
    :return:
    """

    str3="0123456789\n0123456789\n01234567890123456789"

    desired3="0123456789\n0123456789\n012345678901\n23456789"

    w1 = textwrap.TextWrapper(width=12, replace_whitespace=False)
    r1 = w1.fill(str3)
    assert r1 != desired3
    assert r1 == "0123456789\n0123456789\n0\n123456789012\n3456789"

    w2 = DocumentWrapper(width=12, replace_whitespace=False)
    r2 = w2.fill(str3)
    assert r2 == desired3


def test_deepstr():

    obj = {'a': np.arange(100).reshape(10, 10), 'bbbb': [1, 3, np.arange(6).reshape(2, 3), ('xx', 'yy')]}
    obj['c'] = obj['bbbb']
    string_desc = deepstr(obj)
    print(string_desc)

    obj = {}
    string_desc = deepstr(obj)
    print(string_desc)
    # For now, no assertions, because string contains IDS which will always change.  We can come up with some way to do this later with regular experessions if needed.


def test_str_with_arrayopts():

    a = np.arange(1, 6, 1./3)
    assert str_with_arrayopts(a, float_format='.3g', threshold=None) == '[1 1.33 1.67 2 2.33 2.67 3 3.33 3.67 4 4.33 4.67 5 5.33 5.67]'
    assert str_with_arrayopts(a, float_format='.3g', threshold=5) == '[1 1.33 1.67 ..., 5 5.33 5.67]'  # Yeah there's a minimum of 3 on each end.

    b = a.reshape(3, 5)
    assert str_with_arrayopts(b, float_format='.3g', threshold=5) == '[[1 1.33 1.67 2 2.33]\n [2.67 3 3.33 3.67 4]\n [4.33 4.67 5 5.33 5.67]]'


def test_surround_with_header():

    a = surround_with_header('abcd', width=40)
    assert len(a)==40
    b = surround_with_header('abcde', width=40)
    assert len(b)==40

    a = surround_with_header('abcd', width=41)
    assert len(a)==41
    b = surround_with_header('abcde', width=41)
    assert len(b)==41

    a = surround_with_header('abcd', width=2)
    assert len(a)==6


def test_nested_capture():

    with CaptureStdOut() as cap1:
        print('a')
        with CaptureStdOut() as cap2:
            print('b')
        print('c')

    assert cap2.read()=='b\n'
    assert cap1.read()=='a\nb\nc\n'


def test_capture_prefix():

    with CaptureStdOut() as cap1:
        print('a')
        with CaptureStdOut(prefix='abc:') as cap2:
            print('b')
        print('c')

    print ('Done')
    assert cap2.read()=='b\n'
    assert cap1.read()=='a\nabc:b\nc\n'


def test_sensible_str():

    a = [1, 2, 3]
    stra = sensible_str(a)
    assert stra=='[1,2,3]'

    a = OrderedDict([('a', [1,2,3]), ('b', (3, list(range(20)))), ('c', np.arange(20).reshape(4, 5)), ('d', np.arange(4).reshape(2, 2))])
    stra = sensible_str(a, size_limit=4, compact=True)
    assert stra=='OrderedDict([a:[1,2,3],b:(3,<len20-list>),c:<(4,5)ndarray>,d:ndarray([[01],[23]])])'


def test_format_duration():
    assert format_duration(0.0025) == '2.5ms'
    assert format_duration(0.25) == '250.0ms'
    assert format_duration(25) == '25.0s'
    assert format_duration(250) == '0:04:10'
    assert format_duration(2500) == '0:41:40'
    assert format_duration(25000) == '6:56:40'
    assert format_duration(250000) == '2d,21:26:40'
    assert format_duration(2500000) == '28d,22:26:40'
    assert format_duration(25000000) == '289d,8:26:40'
    assert format_duration(250000000) == '2893d,12:26:40'


if __name__ == '__main__':
    test_indent_print()
    test_side_by_side()
    test_document_wrapper()
    test_deepstr()
    test_str_with_arrayopts()
    test_surround_with_header()
    test_nested_capture()
    test_capture_prefix()
    test_sensible_str()
    test_format_duration()
