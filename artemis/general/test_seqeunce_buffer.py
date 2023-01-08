from artemis.general.sequence_buffer import SequenceBuffer, OutOfBufferException
from pytest import raises


def test_sequence_buffer():

    seqbuf = SequenceBuffer(max_elements=3)
    seqbuf.append('a')
    assert seqbuf.lookup(-1) == (0, 'a')
    seqbuf.append('b')
    assert seqbuf.lookup(-1) == (1, 'b')
    seqbuf.append('c')
    assert seqbuf.lookup(-1) == (2, 'c')
    seqbuf.append('d')
    assert seqbuf.lookup(-1) == seqbuf.lookup(3) == (3, 'd')
    assert seqbuf.lookup(-2) == seqbuf.lookup(2) == (2, 'c')
    assert seqbuf.lookup(-3) == seqbuf.lookup(1) == (1, 'b')
    with raises(OutOfBufferException):
        seqbuf.lookup(-4)
    with raises(OutOfBufferException):
        seqbuf.lookup(0)
    assert seqbuf.lookup(-4, jump_to_edge=True) == (1, 'b')

    with raises(OutOfBufferException):
        seqbuf.lookup(4)
    assert seqbuf.lookup(4, jump_to_edge=True) == (3, 'd')
    data_source = iter(['e', 'f', 'g', 'h'])
    assert seqbuf.lookup(4, new_data_source=data_source) == (4, 'e')
    assert seqbuf.lookup(6, new_data_source=data_source) == (6, 'g')
    with raises(OutOfBufferException):
        seqbuf.lookup(8, new_data_source=data_source)
    assert seqbuf.lookup(8, new_data_source=data_source, jump_to_edge=True) == (7, 'h')


if __name__ == '__main__':
    test_sequence_buffer()
