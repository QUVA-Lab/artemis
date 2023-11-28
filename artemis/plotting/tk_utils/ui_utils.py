import os
import platform
import re
import subprocess
import tkinter as tk
import traceback
from contextlib import contextmanager
from typing import Callable, Any, TypeVar, Union, Tuple, Dict
from typing import Sequence, Optional

import cv2
from PIL import Image, ImageTk


from artemis.general.custom_types import BGRImageArray
from artemis.image_processing.image_builder import ImageBuilder
from artemis.image_processing.image_utils import BGRColors
from artemis.plotting.tk_utils.constants import ThemeColours
from artemis.plotting.tk_utils.tk_error_dialog import ErrorDetail
from artemis.plotting.tk_utils.tooltip import create_tooltip


def bgr_image_to_pil_image(image: BGRImageArray) -> Image:
    cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2image)


def bgr_image_to_tk_image(image: BGRImageArray) -> Image:
    return ImageTk.PhotoImage(image=bgr_image_to_pil_image(image))


_CACHED_STANDBY_IMAGES = {}


def get_awaiting_input_image(size_xy = (640, 480), text = f'Awaiting input\nClick on a video to view') -> BGRImageArray:
    w, h = size_xy
    global _AWAITING_INPUT_IMAGE
    arg_tuple = (w, h, text)
    if arg_tuple not in _CACHED_STANDBY_IMAGES:
        if len(_CACHED_STANDBY_IMAGES) > 10:  # just clear
            _CACHED_STANDBY_IMAGES.clear()

        builder = ImageBuilder.from_blank((w, h), color=BGRColors.BLACK)
        for i, line_text in enumerate(text.split('\n')):
            builder = builder.draw_text(text=line_text, anchor_xy=(0.5, 0.5), loc_xy=(w//2, h//2-40+40*i), colour=BGRColors.WHITE)
        _CACHED_STANDBY_IMAGES[arg_tuple] = builder.image
    return _CACHED_STANDBY_IMAGES[arg_tuple]


def open_file_or_folder_in_system(path: str):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


GLOBAL_TERMINATION_REQUEST = False


def set_global_termination_request():
    global GLOBAL_TERMINATION_REQUEST
    GLOBAL_TERMINATION_REQUEST = True


def is_global_termination_request():
    return GLOBAL_TERMINATION_REQUEST


class RespectableLabel(tk.Label):

    def __init__(self,
                 master: tk.Frame,
                 text: str,
                 command: Optional[Callable[[], Any]] = None,
                 shortcut: Optional[str] = None,
                 tooltip: Optional[str] = None,
                 button_id: Optional[str] = None,
                 add_shortcut_to_tooltip: bool = True,
                 **kwargs
                 ):
        tk.Label.__init__(self, master, text=text, **kwargs)
        if command is not None:
            self.bind("<Button-1>", lambda event: command())
            self.config(cursor="hand2", relief=tk.RAISED)

        if button_id is not None and command is not None:
            register_button(button_id, command)

        self._command = command
        if shortcut is not None:
            master.winfo_toplevel().bind(shortcut, self._execute_shortcut)
        if tooltip is not None or (shortcut is not None and add_shortcut_to_tooltip):
            if add_shortcut_to_tooltip and shortcut is not None:
                shortcut_stroke = shortcut.strip('<>')
                shortcut_stroke = re.sub(r'([A-Z])', r'Shift-\1', shortcut_stroke)
                tooltip = f"({shortcut_stroke})" if tooltip is None else f"{tooltip} ({shortcut_stroke})"
            create_tooltip(widget=self, text=tooltip, background=ThemeColours.TOOLTIP_BACKGROUND)

    def _execute_shortcut(self, event: tk.Event):
        if not isinstance(event.widget, (tk.Text, tk.Entry)):
            # Block keystrokes from something is being typed into a text field
            self._command()

class ToggleLabel(RespectableLabel):

    def __init__(self,
                 master: tk.Frame,
                 off_text: str,
                 on_text: Optional[str] = None,
                 on_bg: Optional[str] = None,
                 off_bg: Optional[str] = None,
                 state_switch_pre_callback: Optional[Callable[[bool], bool]] = None,
                 state_switch_callback: Optional[Callable[[bool], Any]] = None,
                 call_switch_callback_immidiately: bool = True,
                 initial_state: bool = False,
                 tooltip: Optional[str] = None,
                 **kwargs
                 ):

        RespectableLabel.__init__(self, master, text='', command=self.toggle, tooltip=tooltip, borderwidth=1, relief=tk.RAISED, **kwargs)
        self._state = False
        self._state_switch_pre_callback = state_switch_pre_callback
        self._state_switch_callback = state_switch_callback
        self._on_text = on_text or off_text
        self._off_text = off_text
        self._on_bg = on_bg
        self._off_bg = off_bg
        self.set_toggle_state(initial_state, call_callback=call_switch_callback_immidiately)

    def set_toggle_state(self, state: bool, call_callback: bool = True):
        if self._state_switch_pre_callback is not None:
            state = self._state_switch_pre_callback(state)
        self._state = state
        self.config(text=self._on_text if self._state else self._off_text, background=self._on_bg if self._state else self._off_bg, relief=tk.SUNKEN if self._state else tk.RAISED)
        if self._state_switch_callback is not None and call_callback:
            self._state_switch_callback(self._state)

    def get_toggle_state(self) -> bool:
        return self._state

    def toggle(self):
        self.set_toggle_state(not self._state)


_BUTTON_CALLBACK_ACCESSORS: Optional[Dict[str, Callable]] = None


@contextmanager
def hold_button_registry():
    global _BUTTON_CALLBACK_ACCESSORS
    old = _BUTTON_CALLBACK_ACCESSORS
    try:
        _BUTTON_CALLBACK_ACCESSORS = {}
        yield
    finally:
        _BUTTON_CALLBACK_ACCESSORS = None


def press_button_by_id(button_id: str):
    assert _BUTTON_CALLBACK_ACCESSORS is not None, "You need to do this from within hold_button_callback_accessors"
    _BUTTON_CALLBACK_ACCESSORS[button_id]()


def register_button(button_id: str, callback: Callable):
    if _BUTTON_CALLBACK_ACCESSORS is not None:
        _BUTTON_CALLBACK_ACCESSORS[button_id] = callback


class RespectableButton(tk.Button):

    def __init__(self,
                 master: tk.Frame,
                 text: str,
                 command: Callable[[], Any],
                 error_handler: Optional[Callable[[ErrorDetail], Any]] = None,
                 tooltip: Optional[str] = None,
                 shortcut: Optional[Union[str, Sequence[str]]] = None,
                 button_id: Optional[str] = None,  # Use this in hold_button_callback_accessors with register_button
                 **kwargs
                 ):
        tk.Button.__init__(self, master, text=text, **kwargs)

        if button_id is not None:
            register_button(button_id, command)

        # Add callback when clicked
        self.bind("<Button-1>", lambda event: self._call_callback_with_safety())

        self._original_text = text
        # self._original_command = command
        # self._original_tooltip = tooltip
        # self._highlight = highlight

        self._original_config = {k: v[4] for k, v in self.config().items() if len(v)>4}  # Get all "current" values
        self._original_tooltip = tooltip
        self._original_callback = self._callback = command
        self._error_handler = error_handler

        if tooltip is not None:
            create_tooltip(widget=self, text=tooltip, background=ThemeColours.TOOLTIP_BACKGROUND)
        if shortcut is not None:
            for s in [shortcut] if isinstance(shortcut, str) else shortcut:
                master.winfo_toplevel().bind(s, lambda event: self._call_callback_with_safety())

    def restore(self):
        self.config(**self._original_config)
        self._callback = self._original_callback
        create_tooltip(widget=self, text=self._original_tooltip, background=ThemeColours.TOOLTIP_BACKGROUND)

    def modify(self, tooltip: str, command: Optional[Callable[[], Any]], **kwargs):
        self.config(**kwargs)
        if command is not None:
            self._callback = command
        create_tooltip(widget=self, text=tooltip, background=ThemeColours.TOOLTIP_BACKGROUND)

    def _call_callback_with_safety(self):
        try:
            self._callback()
        except Exception as e:
            err = e
            traceback_str = traceback.format_exc()
            print(traceback_str)
            if self._error_handler:
                self._error_handler(ErrorDetail(error=err, traceback=traceback_str, additional_info=f"Button: '{self._original_text}'"))
            raise e


FrameType = TypeVar('FrameType', bound=tk.Frame)

@contextmanager
def populate_frame(frame: Optional[FrameType] = None) -> FrameType:
    """
    This context manager doesn't really do anything, but it helps to structure the code
    with indentaiton so you see what belongs to what frame.
    :param frame:
    :return:
    """

    if frame is None:
        frame = tk.Frame()
    # for child in frame.winfo_children():
    #     child.destroy()
    yield frame
    # frame.pack()
    # frame.update()


class ButtonPanel(tk.Frame):

    def __init__(self,
                 master: tk.Frame,
                 error_handler: Optional[Callable[[ErrorDetail], Any]] = None,
                 as_row: bool = True,  # False = column
                 font: Optional[Union[str, Tuple[str, int]]] = (None, 14),
                 **kwargs):
        tk.Frame.__init__(self, master, **kwargs)
        self._error_handler = error_handler
        self._buttons = []
        self._as_row = as_row
        self._font = font
        if as_row:
            self.rowconfigure(0, weight=1)
        else:
            self.columnconfigure(0, weight=1)
        self._count = 0

    def add_button(self,
                   text: str,
                   command: Callable[[], Any],
                   tooltip: Optional[str] = None,
                   shortcut: Optional[str] = None,
                   highlight: bool = False,
                   weight: int = 1,
                   **kwargs
                   ) -> RespectableButton:
        button = RespectableButton(
            self,
            text=text,
            tooltip=tooltip,
            shortcut=shortcut,
            command=command,
            padx=0,
            pady=0,
            font=self._font,
            # width=1,
            highlightbackground=ThemeColours.HIGHLIGHT_COLOR if highlight else None,
            error_handler=self._error_handler,
            **kwargs
        )
        if self._as_row:
            button.grid(column=self._count, row=0, sticky=tk.NSEW)
            self.columnconfigure(self._count, weight=weight)
        else:
            button.grid(column=0, row=self._count, sticky=tk.NSEW)
            self.rowconfigure(self._count, weight=weight)
        self._count += 1
        return button

# def add_respectable_button(
#         frame: tk.Frame,
#         error_handler: Optional[Callable[[ErrorDetail], Any]],
#         text: str,
#         command: Callable[[], Any()],
#         tooltip: Optional[str] = None,
#         shortcut: Optional[str] = None,
#         highlight: bool = False,
#         add_restore_method: bool = False
# ) -> tk.Button:
#     nonlocal count
#
#     def call_callback_with_safety(callback: Callable[[], None]):
#         try:
#             callback()
#         except Exception as e:
#             err = e
#             traceback_str = traceback.format_exc()
#             if error_handler:
#                 error_handler(ErrorDetail(error=err, traceback=traceback_str, additional_info=f"Button: '{text}'"))
#             raise e
#
#             # TODO: Error handling
#             # self.after(50, lambda: self._handle_error(err, command=f"Button {text}", traceback_str=traceback_str))
#
#     button = tk.Button(button_row_frame, text=text, command=partial(call_callback_with_safety, command), highlightbackground=ThemeColours.HIGHLIGHT_COLOR if highlight else None)
#     button.grid(row=0, column=count, sticky=tk.NSEW)
#     count += 1
#     if tooltip is not None:
#         create_tooltip(widget=button, text=tooltip, background=ThemeColours.TOOLTIP_BACKGROUND)
#     if shortcut is not None:
#         self.bind(shortcut, lambda event: command())
#
#     if add_restore_method:
#         def restore():
#             button.config(text=text, command=partial(call_callback_with_safety, command), highlightbackground=ThemeColours.HIGHLIGHT_COLOR if highlight else None)
#             create_tooltip(widget=button, text=tooltip, background=ThemeColours.TOOLTIP_BACKGROUND)
#
#         button.restore = restore
#
#     return button