import os
import platform
import re
import subprocess
import tkinter as tk
import traceback
from contextlib import contextmanager
from enum import Enum
from typing import Callable, Any, TypeVar, Union, Tuple, Dict, Generic
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


def get_shortcut_string(shortcut: str) -> str:
    """ Formats the shortcut as a string to display in a tooltip
    Replaces upper-case letters with 'Shift-<letter>' unless Shift is already in the shortcut
    E.g. 'Control-A' -> 'Ctrl-Shift-A'
    """
    shortcut_stroke = shortcut.strip('<>')
    # display_stroke = re.sub(r'([A-Z])', r'Shift-\1', shortcut_stroke) if 'Shift' not in shortcut_stroke else shortcut_stroke
    # Change above so that it only subs lone capital lettes
    display_stroke = re.sub(r'(?<![A-Za-z])([A-Z])(?![A-Za-z])', r'Shift-\1', shortcut_stroke)
    return display_stroke


class EmphasizableMixin:

    def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self._original_highlighting = {}
        if isinstance(self, tk.Label):
            self._original_highlighting.update({k: v[4] for k, v in self.config().items() if len(v)>4 and k in ['highlightbackground', 'highlightthickness', 'fg']})
    def set_emphasis(self, emphasis: bool):

        if emphasis:
            self.configure(highlightbackground=ThemeColours.HIGHLIGHT_COLOR, highlightthickness=2, fg=ThemeColours.HIGHLIGHT_COLOR, relief=tk.RAISED)
        else:
            self.configure(**self._original_highlighting)


class RespectableLabel(tk.Label, EmphasizableMixin):

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
        EmphasizableMixin.__init__(self, master, text=text, **kwargs)
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
                shortcut_stroke = get_shortcut_string(shortcut)
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

    def set_state_switch_callback(self, callback: Optional[Callable[[bool], Any]]):
        self._state_switch_callback = callback

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


class ReparentableWidgetMixin(tk.Widget):

    def __init__(self, **kwargs):
        self._copy_kwargs = kwargs

    def adopt_to_new_parent(self, parent: tk.Widget, **kwargs) -> tk.Widget:
        return self.__class__(parent, **{**self._copy_kwargs, **kwargs})


class RespectableButton(tk.Button, ReparentableWidgetMixin, EmphasizableMixin):

    def __init__(self,
                 master: tk.Frame,
                 text: str,
                 command: Callable[[], Any],
                 error_handler: Optional[Callable[[ErrorDetail], Any]] = None,
                 tooltip: Optional[str] = None,
                 shortcut: Optional[Union[str, Sequence[str]]] = None,
                 add_shortcut_to_tooltip: bool = True,
                 button_id: Optional[str] = None,  # Use this in hold_button_callback_accessors with register_button
                 as_label: bool = False,
                 **kwargs
                 ):
        if as_label:
            tk.Label.__init__(self, master, text=text, relief=tk.RAISED if command else None, **kwargs)
        else:
            tk.Button.__init__(self, master, text=text, **kwargs)
        ReparentableWidgetMixin.__init__(self, text=text, command=command, error_handler=error_handler, tooltip=tooltip,
                          shortcut=shortcut, button_id=button_id, **kwargs)
        EmphasizableMixin.__init__(self, **kwargs)

        if add_shortcut_to_tooltip and shortcut is not None:
            shortcut_stroke = get_shortcut_string(shortcut)
            tooltip = f"({shortcut_stroke})" if tooltip is None else f"{tooltip} ({shortcut_stroke})"

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
                s = f"<{s.strip('<>')}>"
                master.winfo_toplevel().bind(s, lambda event: self._call_callback_with_safety())

    def get_command(self) -> Callable[[], Any]:
        return self._callback

    def get_tooltip(self) -> Optional[str]:
        return self._original_tooltip

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
                 max_buttons_before_expand: Optional[int] = None,
                 **kwargs):
        tk.Frame.__init__(self, master, **kwargs)
        self._error_handler = error_handler
        self._buttons = []
        self._as_row = as_row
        self._font = font
        self._max_buttons_before_expand = max_buttons_before_expand
        self._is_adding_expand_button = False
        if as_row:
            self.rowconfigure(0, weight=1)
        else:
            self.columnconfigure(0, weight=1)

    def add_label(self, text: str, weight: int = 1):
        label = RespectableLabel(self, text=text)
        self._add_next_widget(label, weight=weight)
        return label

    def add_button(self,
                   text: str,
                   command: Callable[[], Any],
                   tooltip: Optional[str] = None,
                   shortcut: Optional[str] = None,
                   highlight: bool = False,
                   weight: int = 1,
                   as_label: bool = False,
                   padx=0,
                   pady=0,
                   surround_padding: int = 0,
                   **kwargs
                   ) -> RespectableButton:
        button = RespectableButton(
            self,
            text=text,
            tooltip=tooltip,
            shortcut=shortcut,
            command=command,
            padx=padx,
            pady=pady,
            font=self._font,
            as_label=as_label,
            # width=1,
            highlightbackground=ThemeColours.HIGHLIGHT_COLOR if highlight else None,
            error_handler=self._error_handler,
            **kwargs
        )
        # if self._as_row:
        #     button.grid(column=self._count, row=0, sticky=tk.NSEW)
        #     self.columnconfigure(self._count, weight=weight)
        # else:
        #     button.grid(column=0, row=self._count, sticky=tk.NSEW)
        #     self.rowconfigure(self._count, weight=weight)
        # self._count += 1
        self._add_next_widget(button, weight=weight, padding=surround_padding)
        return button

    def _expand(self):

        def on_button_press(original_callback: Callable[[], None]):
            def callback():

                original_callback()
                print('Calling destroy')
                new_window.destroy()
            return callback

        new_window = tk.Toplevel(self.master)
        new_window.title("Additional Buttons")
        new_window.geometry(f"600x{min(500, 50*len(self._buttons)+50)}")
        new_window.resizable(False, False)

        # Keep it on top until clicked
        new_window.attributes("-topmost", True)

        for button in self._buttons[self._max_buttons_before_expand:]:
            new_button = button.adopt_to_new_parent(new_window, command=on_button_press(button.get_command()), text=button.cget('text')+(" : "+t if (t:=button.get_tooltip()) is not None else ''))
            new_button.pack(fill=tk.BOTH, expand=True)

        # Add a cancel button
        cancel_button = RespectableButton(new_window, text='Cancel', command=new_window.destroy)
        cancel_button.pack(fill=tk.BOTH, expand=True)

        # Clicking anywhere outside the window will close it
        new_window.bind("<FocusOut>", lambda *args: new_window.destroy())

        new_window.update()
        new_window.wait_window()

        # self.destroy()

    def _add_next_widget(self, widget: tk.Widget, weight: int = 1, padding: int = 0):
        count = len(self._buttons)
        if self._max_buttons_before_expand is not None and count+1 == self._max_buttons_before_expand and not self._is_adding_expand_button:
            # If we're at the limit - add the expand button
            self._is_adding_expand_button = True
            self.add_button('...', self._expand, tooltip="Show other buttons", weight=weight)
            self._is_adding_expand_button = False

        if self._max_buttons_before_expand is not None and count+1 > self._max_buttons_before_expand:
            pass
        elif self._as_row:
            widget.grid(column=count, row=0, sticky=tk.NSEW, padx=padding, pady=padding)
            self.columnconfigure(count, weight=weight, uniform='button')
        else:
            widget.grid(column=0, row=count, sticky=tk.NSEW, padx=padding, pady=padding)
            self.rowconfigure(count, weight=weight, uniform='button')
        if not self._is_adding_expand_button:
            self._buttons.append(widget)

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


MultiStateEnumType = TypeVar('MultiStateEnumType', bound=Enum)


class MultiStateToggle(ButtonPanel, Generic[MultiStateEnumType]):

    def __init__(self,
                 master: tk.Frame,
                 enum_type: type(MultiStateEnumType),
                 error_handler: Optional[Callable[[ErrorDetail], Any]] = None,
                 initial_state: Optional[MultiStateEnumType] = None,
                 on_state_change_callback: Optional[Callable[[MultiStateEnumType], None]] = None,
                 call_callback_immediately: bool = True,
                 pad: int = 5,
                 surround_padding: int = 0,
                 on_button_config: Optional[dict] = None,
                 off_button_config: Optional[dict] = None,
                 tooltip_maker: Optional[Callable[[MultiStateEnumType], str]] = None,
                 **kwargs
                 ):
        ButtonPanel.__init__(self, master, error_handler=error_handler, **kwargs)
        self._active_state: Optional[MultiStateEnumType] = None
        self._on_button_config = on_button_config or {}
        self._off_button_config = off_button_config or {}
        if initial_state is None:
            initial_state = list(enum_type)[0]
        self._on_state_change_callback = on_state_change_callback if call_callback_immediately else None
        for state in enum_type:
            if tooltip_maker is not None:
                tooltip = tooltip_maker(state)
            else:
                tooltip = f"Switch to '{state.value}'"
            self.add_button(text=state.value, command=lambda s=state: self.set_state(s), tooltip=tooltip,
                            as_label=True, padx=pad, pady=pad, surround_padding=surround_padding)
        self.set_state(initial_state)
        self._on_state_change_callback = on_state_change_callback

    def set_state(self, state: MultiStateEnumType):
        old_state = self._active_state
        state_index = list(type(state)).index(state)
        for i, button in enumerate(self._buttons):
            if i == state_index:
                button.config(relief=tk.SUNKEN, **self._on_button_config)
            else:
                button.config(relief=tk.RAISED, **self._off_button_config)
        self._active_state = state
        if self._on_state_change_callback is not None and old_state != state:  # Avoid recursion
            self._on_state_change_callback(state)

    def get_state(self) -> MultiStateEnumType:
        assert self._active_state is not None, "No state set"
        return self._active_state
