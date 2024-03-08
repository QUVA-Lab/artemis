from artemis.plotting.easy_window import rpack_from_aspect_ratio


def test_pack_with_aspect_ratio():
    corners = rpack_from_aspect_ratio(sizes = [(6, 6), (5, 5), (4, 4)], aspect_ratio=1.)
    assert corners == [(0, 0), (6, 0), (6, 5)]
    corners = rpack_from_aspect_ratio(sizes = [(6, 6), (5, 5), (4, 4)], aspect_ratio=10)
    assert corners == [(0, 0), (6, 0), (11, 0)]
    corners = rpack_from_aspect_ratio(sizes = [(6, 6), (5, 5), (4, 4)], aspect_ratio=0.1)
    assert corners == [(0, 0), (0, 6), (0, 11)]


if __name__ == "__main__":
    test_pack_with_aspect_ratio()

