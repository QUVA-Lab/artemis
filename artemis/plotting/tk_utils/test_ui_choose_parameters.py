from dataclasses import dataclass, replace
from typing import Sequence, Optional, Tuple, Any, Union

import pytest

from artemis.plotting.tk_utils.tk_basic_utils import hold_tkinter_root_context
from artemis.plotting.tk_utils.ui_choose_parameters import ui_choose_parameters, ParameterSelectionFrame
import tkinter as tk


@dataclass
class Book:
    title: str
    pages: int


@dataclass
class Author:
    name: str
    age: int
    books: Sequence[Book]


def test_ui_choose_parameters():
    steven_king = Author(name="Steven King", age=72, books=[Book(title="The Shining", pages=447), Book(title="The Stand", pages=1153)])
    steven_king_edited = ui_choose_parameters(initial_params=steven_king, timeout=0.2)
    assert steven_king_edited == steven_king


def test_edit_params():

    steven_king = Author(name="Steven King", age=72, books=[Book(title="The Shining", pages=447), Book(title="The Stand", pages=1153)])

    callback_called = False

    with hold_tkinter_root_context() as root:
        window = tk.Toplevel(root)

        def callback(key_path: Tuple[Union[str, int], ...], new_value: Any) -> None:
            nonlocal callback_called
            callback_called = True
            assert key_path == ('age', )
            assert new_value.get() == 73
            print(f"Callback called with {key_path} and {new_value.get()}")

        ps_frame = ParameterSelectionFrame(window, on_change_callback=callback)
        ps_frame.set_parameters(initial_params=steven_king)
        ps_frame.pack()
        ps_frame.get_variables()[('age', )].set(73)

        window.after(200, window.destroy)
        window.wait_window()
        assert callback_called
        steven_king_edited = ps_frame.get_filled_parameters()
        assert steven_king_edited == replace(steven_king, age=73)


@pytest.mark.skip(reason="Requires manual intervention")
def test_change_subfield():

    steven_king = Author(name="Steven King", age=72, books=[Book(title="The Shining", pages=447), Book(title="The Stand", pages=1153)])
    steven_king_edited = ui_choose_parameters(initial_params=steven_king, title='Add one page to The Shining')
    print(steven_king_edited)
    assert steven_king_edited == replace(steven_king, books=[Book(title="The Shining", pages=448), Book(title="The Stand", pages=1153)])


@pytest.mark.skip(reason="Requires manual intervention")
def test_nullable_field():

    @dataclass
    class Book:
        title: str
        chapter_names: Optional[Sequence[str]] = None  # A book may not have chapters, in which case this field is None.  It may also be an emty book, in which case it is an empty list.

    book = Book(title="The Shining", chapter_names=['Chapter 1', 'Chapter 2', 'Chapter 3'])
    steven_king_edited = ui_choose_parameters(initial_params=book, title='Remove chapters')
    print(steven_king_edited)
        # assert steven_king_edited == replace(steven_king, books=[Book(title="The Shining", pages=448), Book(title="The Stand", pages=1153)])


@pytest.mark.skip(reason="Requires manual intervention")
def test_nested_parameter_selection():

    @dataclass
    class Author:
        name: str
        age: int
        books: Sequence[Book]
        favourite_foods: Optional[Sequence[str]] = None  # None means we don't know, empty list means they don't have any.

    new_params = ui_choose_parameters(
        params_type=Author,
        initial_params=Author(name="Steven King", age=72, books=[Book(title="The Shining", pages=447), Book(title="The Stand", pages=1153)]),
        title='Edit Author',
        depth = 1,
        timeout=0.2,
        # editable_fields=['nameame', 'age', 'books'],
        # editable_fields=False,
    )
    print(new_params)



if __name__ == "__main__":
    # test_ui_choose_parameters()
    test_edit_params()
    # test_change_subfield()
    # test_nullable_field()
    # test_nested_parameter_selection()
