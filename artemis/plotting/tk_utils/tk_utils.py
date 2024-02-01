import tkinter as tk
import traceback
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from tkinter import Widget
from typing import Iterable, Optional, Tuple, Mapping, TypeVar
from typing import Mapping, Callable, Union, Any, Optional, Sequence, Tuple, List

from artemis.general.measuring_periods import PeriodicChecker
from artemis.plotting.tk_utils.tk_error_dialog import ErrorDetail
from artemis.plotting.tk_utils.ui_utils import RespectableButton
from artemis.plotting.tk_utils.constants import UIColours, ThemeColours
import platform
import subprocess

# from artemis.plotting.tk_utils.bitmap_view import TkImageWidget
# from video_scanner.utils import ResourceFiles


# class TkKeys(enum.Enum):
#     # List of Tkinter key names (Thank you ChatGPT)
#     KEY_1 = '1'
#     KEY_2 = '2'
#     KEY_3 = '3'
#     KEY_4 = '4'
#     KEY_5 = '5'
#     KEY_6 = '6'
#     KEY_7 = '7'
#     KEY_8 = '8'
#     KEY_9 = '9'
#     KEY_0 = '0'
#     A = 'a'
#     B = 'b'
#     C = 'c'
#     D = 'd'
#     E = 'e'
#     F = 'f'
#     G = 'g'
#     H = 'h'
#     I = 'i'
#     J = 'j'
#     K = 'k'
#     L = 'l'
#     M = 'm'
#     N = 'n'
#     O = 'o'
#     P = 'p'
#     Q = 'q'
#     R = 'r'
#     S = 's'
#     T = 't'
#     U = 'u'
#     V = 'v'
#     W = 'w'
#     X = 'x'
#     Y = 'y'
#     Z = 'z'
#     SPACE = 'space'
#     LEFT = 'left'
#     RIGHT = 'right'
#     UP = 'up'
#     DOWN = 'down'
#     RETURN = 'return'
#     ESCAPE = 'escape'
#     BACKSPACE = 'backspace'
#     TAB = 'tab'
#     CAPSLOCK = 'capslock'
#     SHIFT = 'shift'
#     CONTROL = 'control'
#     OPTION = 'option'
#     COMMAND = 'command'
#     FN = 'fn'
#     F1 = 'f1'
#     F2 = 'f2'
#     F3 = 'f3'
#     F4 = 'f4'
#     F5 = 'f5'
#     F6 = 'f6'
#     F7 = 'f7'
#     F8 = 'f8'
#     F9 = 'f9'
#     F10 = 'f10'
#     F11 = 'f11'
#     F12 = 'f12'
#
#     def __add__(self, other):
#         return other.__radd__(self)
#
#     def __radd__(self, other):
#         if not all(o in (self.COMMAND, self.CONTROL, self.SHIFT, self.OPTION) for o in other.split('-')):
#             raise BadModifierError(f"Key {self} is not a modifier")
#         return f"{other.value}-{self.value}"


class BadModifierError(Exception):
    """ Raised when you try to use an invalidmodifier """


def wrap_ui_callback_with_handler(
        callback: Callable[[tk.Event], Any],
        error_handler: Callable[[ErrorDetail], None],
        info_string: Optional[str] = None,
        reraise: bool = True
) -> Callable[[tk.Event], Any]:
    def wrapped_callback(event: tk.Event) -> Any:
        try:
            callback(event)
        except Exception as err:
            traceback_str = traceback.format_exc()
            details = ErrorDetail(error=err, traceback=traceback_str, additional_info=info_string)
            error_handler(details)
            if reraise:
                raise err

    return wrapped_callback


def bind_callbacks_to_widget(
        callbacks: Mapping[Union[str], Callable[[tk.Event], Any]],
        widget: tk.Widget,
        bind_all: bool = False,
        error_handler: Optional[Callable[[ErrorDetail], None]] = None):
    for key, callback in callbacks.items():
        # widget.bind(f"<{key.strip('<>')}>", callback)
        key_binding_str = f"<{key.strip('<>')}>"
        if error_handler is not None:
            callback = wrap_ui_callback_with_handler(callback, error_handler, info_string=f"Error while pressing {key_binding_str}")
        widget.bind_all(key_binding_str, callback) if bind_all else widget.bind(f"<{key.strip('<>')}>", callback)


def destroy_all_descendents(widget: Widget):
    for c in widget.winfo_children():
        destroy_all_descendents(c)
        c.destroy()


class OptionDialog(tk.Toplevel):
    """
        This dialog accepts a list of options.
        If an option is selected, the results property is to that option value
        If the box is closed, the results property is set to zero
    """

    def __init__(self, title, question, options):
        parent = tk.Toplevel()
        tk.Toplevel.__init__(self, parent)
        self.title(title)
        self.question = question
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.options = options
        self.result = '_'
        self.createWidgets()
        self.grab_set()
        ## wait.window ensures that calling function waits for the window to
        ## close before the result is returned.
        self.wait_window()

    def createWidgets(self):
        frmQuestion = tk.Frame(self)
        tk.Label(frmQuestion, text=self.question).grid()
        frmQuestion.grid(row=1)
        frmButtons = tk.Frame(self)
        frmButtons.grid(row=2)
        column = 0
        for option in self.options:
            btn = tk.Button(frmButtons, text=option, command=lambda x=option: self.setOption(x))
            btn.grid(column=column, row=0)
            column += 1

    def setOption(self, optionSelected):
        self.result = optionSelected
        self.destroy()

    def cancel(self):
        self.result = None
        self.destroy()


# from tkinter import *

@dataclass
class OptionInfo:
    text: str
    help_text: Optional[str]
    shortcut: Optional[str]
    callback: Optional[Callable[[], Any]]

    def get_additional_info(self) -> str:
        description = ''
        if self.help_text:
            description += f"{self.help_text}"
        if self.shortcut:
            description += f" ({self.shortcut})"
        return description


_OPTION_SELECT_OVERRIDE_QUEUE: Optional[List[str]] = None

@contextmanager
def hold_option_select_override(*option_select_queue: str):
    global _OPTION_SELECT_OVERRIDE_QUEUE
    old_queue = _OPTION_SELECT_OVERRIDE_QUEUE
    _OPTION_SELECT_OVERRIDE_QUEUE = deque((list(old_queue) if old_queue is not None else [])+list(option_select_queue))
    try:
        yield
    finally:
        _OPTION_SELECT_OVERRIDE = old_queue


@dataclass
class TkOptionListBuilder:
    _options: List[OptionInfo] = field(default_factory=list)
    _default_option: Optional[OptionInfo] = None  # Must be one of options or will be ignored

    def add_option(self, text: str, help_text: Optional[str] = None, shortcut: Optional[str] = None,
                   is_default: bool = False, callback: Optional[Callable[[], Any]] = None) -> OptionInfo:
        new_option = OptionInfo(text=text, help_text=help_text, shortcut=shortcut, callback=callback)
        self._options.append(new_option)
        if is_default:
            self._default_option = new_option
        return new_option

    def ui_select_option(self,
                         message: str = "Select one:",
                         title: str = "Select an option",
                         as_column_with_help: bool = False,
                         add_cancel_button: bool = False,
                         min_size_xy: Tuple[int, int] = (400, 300),
                         wrap_text_to_window_size: bool = True,
                         n_button_cols: int = 3,
                         call_callback_on_selection: bool = False,
                         font: Optional[Tuple[str, int]] = None,  # e.g. ('Helvetica', 24)
                         ) -> Optional[OptionInfo]:
        if len(self._options)==0:
            raise Exception("No options have been provided to select from.  Call add_option first.")

        result: Optional[OptionInfo] = None
        default = self._default_option

        def buttonfn(op: OptionInfo):
            nonlocal result
            if op == cancel_option:
                result = None
            else:
                result = op
            # choicewin.quit()
            choicewin.destroy()

        choicewin = tk.Toplevel()
        # Keep this window on top

        choicewin.minsize(*min_size_xy)
        choicewin.resizable(False, False)
        choicewin.title(title)

        # Position in center of main application window
        choicewin.geometry(f"+{int(choicewin.winfo_screenwidth() / 2 - min_size_xy[0] / 2)}+"
                           f"{int(choicewin.winfo_screenheight() / 2 - min_size_xy[1] / 2)}")

        parent_frame = tk.Frame(choicewin)
        parent_frame.pack(expand=True)

        options = list(self._options)

        cancel_option = OptionInfo(text="Cancel", help_text=None, shortcut='<Escape>', callback=None)
        if add_cancel_button:
            options += [cancel_option]

        # Align the text to the left, and wrap to the window width
        tk.Label(parent_frame, text=message, anchor=tk.W, wraplength=min_size_xy[0] if wrap_text_to_window_size else None, font=font)\
            .grid(row=0, column=0, sticky=tk.EW, columnspan=min(n_button_cols, len(options)))

        # Change above into loop which assigns to tk_buttons
        tk_buttons = []

        def add_button_for_option(op: OptionInfo, add_tooltip: bool) -> RespectableButton:
            is_default = op == self._default_option
            button = RespectableButton(parent_frame,
                                       text=op.text,
                                       command=partial(buttonfn, op),
                                       default=tk.ACTIVE if is_default else tk.NORMAL,
                                       # Make button highlighted if default
                                       # borderwidth=1,
                                       # border=1,
                                       highlightthickness=1,
                                       # padx=2,
                                       # pady=2,
                                       highlightbackground=ThemeColours.HIGHLIGHT_COLOR if is_default else None,
                                       # highlightcolor=ThemeColours.HIGHLIGHT_COLOR if op == self._default_option else None,
                                       tooltip=op.get_additional_info() or None if add_tooltip else None,
                                       font=font)
            if is_default:
                button.focus_set()
            tk_buttons.append(button)
            return button

        if as_column_with_help:
            for i, op in enumerate(options):
                add_button_for_option(op=op, add_tooltip=False).grid(row=i+1, column=0, sticky=tk.NSEW)
                tk.Label(parent_frame, text=op.get_additional_info(), justify=tk.LEFT, wraplength=min_size_xy[0]*3//4).grid(row=i+1, column=1, sticky=tk.NSEW)
        else:
            for i, op in enumerate(options):
                add_button_for_option(op=op, add_tooltip=True).grid(row=i // n_button_cols + 1, column=i % n_button_cols, sticky=tk.NSEW)

        # Change the selected button with arrow keys
        def change_selected_button(event: tk.Event):
            nonlocal default
            print(f"Key pressed: {event.keysym}")
            if event.keysym in ('Left', 'Up'):
                default = self._options[(self._options.index(default) - 1) % len(self._options)] if default is not None else self._options[-1]
            elif event.keysym in ('Right', 'Down'):
                default = self._options[(self._options.index(default) + 1) % len(self._options)] if default is not None else self._options[0]
            for i, choice in enumerate(self._options):
                if choice == default:
                    print(f"Changing focus to button {i}")
                    tk_buttons[i].focus_set()
                    tk_buttons[i].config(default=tk.ACTIVE)
                    tk_buttons[i].config(highlightbackground=ThemeColours.HIGHLIGHT_COLOR)
                else:
                    tk_buttons[i].config(default=tk.NORMAL)
                    # tk_buttons[i].config(highlightbackground=None)
                    # tk_buttons[i].config(highlightbackground=ThemeColours.BACKGROUND)
                    # No - need to get the actual default color from tkinter itself
                    tk_buttons[i].config(highlightbackground='systemWindowBackgroundColor')

        # tk_buttons = parent_frame.winfo_children()[1:]
        choicewin.bind('<Left>', change_selected_button)
        choicewin.bind('<Right>', change_selected_button)
        choicewin.bind('<Up>', change_selected_button)
        choicewin.bind('<Down>', change_selected_button)

        # If enter is pressed, select the default option:
        choicewin.bind('<Return>', lambda event: buttonfn(default) if default else None)

        # # If escape is pressed, cancel:
        choicewin.bind('<Escape>', lambda event: buttonfn(cancel_option))

        # choicewin.focus_set()

        # choicewin.grab_set()
        if _OPTION_SELECT_OVERRIDE_QUEUE is not None:  # Should only be true in tests
            try:
                option = _OPTION_SELECT_OVERRIDE_QUEUE.popleft()
            except IndexError:
                raise Exception(f"Tried using the option override queue in dialog with title:\n  '{title}'\nand options:\n  {[op.text for op in self._options]} but the queue was empty.")
            option_selected = next((op for op in self._options if op.text == option), None)
            assert option_selected is not None, f"Option {option} not found in options: {[op.text for op in self._options]}"
            buttonfn(option_selected)
        else:
            # Get it to stay on top!
            choicewin.attributes('-topmost', True)
            
            choicewin.wait_window()  # With this line commented in... CRASH (exit code 0, no message)
        # choicewin.mainloop()  # With this line... NO CRASH!
        # choicewin.grab_release()

        if call_callback_on_selection and result is not None and result.callback:
            result.callback()

        print("Got result:", result)

        return result


def tk_select_option(choicelist: Sequence[str],
                     message: str = "Select one:",
                     title: str = "Select an option",
                     default: Optional[str] = None,
                     add_cancel_button: bool = False,
                     min_size_xy: Tuple[int, int] = (400, 300),
                     wrap_text_to_window_size: bool = True,
                     n_button_cols: int = 3,
                     ) -> Optional[str]:

    option_builder = TkOptionListBuilder()

    # options = [option_builder.add_option(text=choice, is_default=choice==default) for choice in choicelist]
    for choice in choicelist:
        option_builder.add_option(text=choice, is_default=choice == default)

    selected = option_builder.ui_select_option(
        message=message, title=title, add_cancel_button=add_cancel_button, min_size_xy=min_size_xy,
        wrap_text_to_window_size=wrap_text_to_window_size, n_button_cols=n_button_cols
    )
    return selected.text if selected is not None else None

    # for choice in choicelist:
    #     opti
    #
    # result = None
    #
    # def buttonfn(choice: str):
    #     nonlocal result
    #     if add_cancel_button and choice == 'Cancel':
    #         result = None
    #     else:
    #         result = choice
    #     choicewin.quit()
    #     choicewin.destroy()
    #
    # choicewin = tk.Toplevel()
    # # Keep this window on top
    #
    # choicewin.minsize(*min_size_xy)
    # choicewin.resizable(False, False)
    # choicewin.title(title)
    #
    # # Position in center of main application window
    # choicewin.geometry(f"+{int(choicewin.winfo_screenwidth() / 2 - min_size_xy[0] / 2)}+"
    #                       f"{int(choicewin.winfo_screenheight() / 2 - min_size_xy[1] / 2)}")
    #
    # parent_frame = tk.Frame(choicewin)
    # # parent_frame.grid(row=0, column=0, sticky=tk.NSEW)
    # parent_frame.pack( expand=True)
    #
    # if add_cancel_button:
    #     choicelist = list(choicelist) + ['Cancel']
    #
    # # tk.Label(parent_frame, text=message).grid(row=0, column=0, sticky=tk.EW, columnspan=len(choicelist))
    # # Align the text to the left, and wrap to the window width
    # tk.Label(parent_frame, text=message, anchor=tk.W, wraplength=min_size_xy[0] if wrap_text_to_window_size else None)\
    #     .grid(row=0, column=0, sticky=tk.EW, columnspan=len(choicelist))
    #     # tk.Button(parent_frame, text=choice, command=partial(buttonfn, choice)).grid(row=i // n_button_cols + 1, column=i % n_button_cols)
    #
    #
    # # tk_buttons = [tk.Button(parent_frame, text=choice, command=partial(buttonfn, choice), default=tk.ACTIVE if choice == default else tk.NORMAL)\
    # #         .grid(row=i // n_button_cols + 1, column=i % n_button_cols, sticky=tk.NSEW) for i, choice in enumerate(choicelist)]
    #
    # # Change above into loop which assigns to tk_buttons
    # tk_buttons = []
    # for i, choice in enumerate(choicelist):
    #     tk_buttons.append(tk.Button(parent_frame, text=choice, command=partial(buttonfn, choice), default=tk.ACTIVE if choice == default else tk.NORMAL))
    #     tk_buttons[-1].grid(row=i // n_button_cols + 1, column=i % n_button_cols, sticky=tk.NSEW)
    #
    # # Change the selected button with arrow keys
    # def change_selected_button(event: tk.Event):
    #     nonlocal default
    #     print(f"Key pressed: {event.keysym}")
    #     if event.keysym == 'Left':
    #         default = choicelist[(choicelist.index(default) - 1) % len(choicelist)]
    #     elif event.keysym == 'Right':
    #         default = choicelist[(choicelist.index(default) + 1) % len(choicelist)]
    #     for i, choice in enumerate(choicelist):
    #         if choice == default:
    #             print(f"Changing focus to button {i}")
    #             tk_buttons[i].focus_set()
    #             tk_buttons[i].config(default=tk.ACTIVE)
    #         else:
    #             tk_buttons[i].config(default=tk.NORMAL)
    #
    # # tk_buttons = parent_frame.winfo_children()[1:]
    # choicewin.bind('<Left>', change_selected_button)
    # choicewin.bind('<Right>', change_selected_button)
    #
    # # If enter is pressed, select the default option:
    # if default is not None:
    #     choicewin.bind('<Return>', lambda event: buttonfn(default))
    #
    # # If escape is pressed, cancel:
    # choicewin.bind('<Escape>', lambda event: buttonfn(None))
    #
    # # choicewin.focus_set()
    #
    # choicewin.grab_set()
    # choicewin.mainloop()
    # # choicewin.wait_window()
    # choicewin.grab_release()
    # return result


def tk_indicate_focus_with_border(has_focus: bool, widget: tk.Frame, color: str = UIColours.EYE_BLUE, non_highlight_color: str = ThemeColours.BACKGROUND, border_thickness: int = 3):
    if has_focus:
        widget.config(highlightthickness=border_thickness, highlightcolor=color, highlightbackground=color)
    else:
        # widget.config(highlightthickness=0)
        widget.config(highlightcolor=non_highlight_color, highlightbackground=non_highlight_color)  # We keep the thickness to avoid all these reconfigurations


def get_focus_indicator_callbacks(widget: tk.Frame, color: str = UIColours.EYE_BLUE, border_thickness: int = 3) -> Mapping[str, Callable]:
    return {
        '<FocusIn>': lambda event: tk_indicate_focus_with_border(has_focus=True, widget=widget, color=color, border_thickness=border_thickness),
        '<FocusOut>': lambda event: tk_indicate_focus_with_border(has_focus=False, widget=widget, color=color, border_thickness=border_thickness),
    }


def tk_info_box(message: str, title: str = "Info"):
    tk_select_option(message=message, choicelist=["Ok"], title=title)


def tk_yesno_box(message: str, title: str = "") -> bool:
    return tk_select_option(message=message, choicelist=["Yes", "No"], title=title) == "Yes"


def tk_show_error(message: str, title: str = "Error"):
    return tk_select_option(message=message, choicelist=["Ok"], title=title)


ItemType = TypeVar('ItemType')


@dataclass
class BlockingTaskDialogFunction:
    n_steps: Optional[int] = None
    max_update_period: float = 0.2
    message: str = "Processing... please wait."

    def show_blocking_task_dialog(
            self,
            blocking_task_iterator: Iterable[ItemType],
    ) -> Optional[Sequence[ItemType]]:

        window = tk.Toplevel()
        # window.geometry('400x250')
        # Make that a minimum size for each dimention
        window.minsize(400, 250)
        tk.Label(window, text=self.message).pack(fill=tk.BOTH, expand=True)
        progress_label = tk.Label(window, font=('Helvetica', 24))
        progress_label.pack(fill=tk.BOTH, expand=True)

        def cancel():
            nonlocal items
            items = None

        tk.Button(window, text='Cancel', command=cancel).pack(pady=10)

        checker = PeriodicChecker(interval=self.max_update_period)

        window.update()
        items: Optional[List[ItemType]] = []
        for i, item in enumerate(blocking_task_iterator, start=1):
            if checker.is_time_for_update():
                progress_label.config(text=f'{i} / {self.n_steps or "?"}' + (f" ({i / self.n_steps:.0%})" if self.n_steps else ''))
                window.update()
            if items is None:
                break
            items.append(item)
        window.destroy()
        return items


class CollapseableWidget(tk.Widget):

    def __init__(self, master: tk.Frame, collapse_initially: bool = False, **kwargs):
        super().__init__(master, **kwargs)
        self._is_collapsed = collapse_initially
        self._pack_manager_args_kwargs: Optional[Tuple[str, Tuple, Mapping]] = None

    def is_collapsed(self) -> bool:
        return self._is_collapsed

    def pack(self, *args, **kwargs):
        self._pack_manager_args_kwargs = ('pack', args, kwargs)
        if self._is_collapsed:
            return
        super().pack(*args, **kwargs)

    def grid(self, *args, **kwargs):
        self._pack_manager_args_kwargs = ('grid', args, kwargs)
        if self._is_collapsed:
            return
        super().grid(*args, **kwargs)

    def set_collapsed(self, collapse_state):
        # print("Setting collapsed to ", collapse_state)
        self._is_collapsed = collapse_state
        if collapse_state and self._pack_manager_args_kwargs is not None:
            self.pack_forget() if self._pack_manager_args_kwargs[0] == 'pack' else self.grid_forget()
        else:
            if self._pack_manager_args_kwargs is None:
                # print("Warning: Cannot uncollapse before first packing")
                return
            pack_manager, args, kwargs = self._pack_manager_args_kwargs
            pack_func = self.pack if pack_manager == 'pack' else self.grid
            pack_func(*args, **kwargs)
            # print(f"Uncollapsed {self} with {pack_manager} with args {args} and kwargs {kwargs}")


class MessageListenerMixin:

    def __init__(self, message_listener: Optional[Callable[[str], None]] = None):
        self.message_listener = message_listener if message_listener is not None else print

    def set_message_listener(self, listener: Callable[[str], None]):
        self.message_listener = listener

    def _send_message(self, message: str):
        if self.message_listener is not None:
            self.message_listener(message)

    def _send_hint_message(self, message: str):
        self._send_message(f"ℹ️: {message}")


def assert_no_existing_root():
    assert tk._default_root is None, "A Tkinter root window already exists!"


_EXISTING_ROOT: Optional[tk.Tk] = None


@contextmanager
def hold_tkinter_root_context():
    """ A context manager that creates a Tk root and destroys it when the context is exited
    Careful now: If you schedule something under root to run with widget.after, it may crash if the root is destroyed before it runs.
    """
    # assert_no_existing_root()
    global _EXISTING_ROOT
    old_value = _EXISTING_ROOT
    root = tk.Tk() if _EXISTING_ROOT is None else _EXISTING_ROOT

    try:
        _EXISTING_ROOT = root
        yield root
    finally:
        try:
            if old_value is None:
                _EXISTING_ROOT = None
                root.destroy()
        except tk.TclError:  # This can happen if the root is destroyed before the context is exited
            pass

#
# def get_widget_overlay_frame(
#         widget: tk.Widget,
#     ) -> tk.Frame:


def is_dark_mode():
    os_name = platform.system()

    if os_name == "Darwin":  # macOS
        try:
            theme = subprocess.check_output(
                "defaults read -g AppleInterfaceStyle", shell=True
            ).decode().strip()
            return theme == "Dark"
        except subprocess.CalledProcessError:
            return False  # Default is light mode if the command fails

    elif os_name == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
            )
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return value == 0
        except:
            return False  # Default or error

    else:  # Linux and other OSes
        # This is more complex due to the variety of Linux environments.
        # Placeholder for now.
        return False


if __name__ == '__main__':
    # reply = messagebox.askyesnocancel(message="Wooooo")

    reply = tk_select_option(["one", "two", "three"])
    print("reply:", reply)

    # test the dialog
    # root=tk.Tk()
    # def run():
    #     values = ['Red','Green','Blue','Yellow']
    #     dlg = OptionDialog('TestDialog',"Select a color",values)
    #     print(dlg.result)
    # # tk.Button(root,text='Dialog',command=run).pack()
    # run()
    # root.mainloop()