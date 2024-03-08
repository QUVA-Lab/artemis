import tkinter as tk
from contextlib import contextmanager
from typing import Optional

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
