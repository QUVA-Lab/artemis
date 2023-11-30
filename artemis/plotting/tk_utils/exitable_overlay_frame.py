import tkinter as tk
from typing import Optional, Callable

from artemis.plotting.tk_utils.constants import ThemeColours
from artemis.plotting.tk_utils.ui_utils import RespectableLabel


class ExitableOverlayFrame(tk.Frame):
    """ A frame with a "main" frame and an "overlay" frame.  The overlay frame hides the main frame when it is visible."""

    def __init__(self,
                 parent_frame: tk.Frame,
                 overlay_label: str = "",
                 pre_state_change_callback: Optional[Callable[[bool], bool]] = None,
                 on_state_change_callback: Optional[Callable[[bool], None]] = None,
                 return_shortcut: Optional[str] = None,
                 return_button_config: Optional[dict] = None,
                 label_config: Optional[dict] = None,
                 ):

        tk.Frame.__init__(self, parent_frame, bg=ThemeColours.BACKGROUND)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._main_frame = tk.Frame(self, bg=ThemeColours.BACKGROUND)
        self._main_frame.grid(row=0, column=0, sticky=tk.NSEW)

        self._overlay_frame_parent = tk.Frame(self, bg=ThemeColours.BACKGROUND)
        # Do not grid this yet.  We will grid it when we want to show the overlay.

        self._overlap_panel = tk.Frame(self._overlay_frame_parent, bg=ThemeColours.BACKGROUND)
        self._overlap_panel.pack(side=tk.TOP, fill=tk.X, expand=False)

        self._overlay_exit_button = RespectableLabel(
            self._overlap_panel,
            text="Exit",
            command=lambda: self.set_overlay_visible(False),
            bg=ThemeColours.BACKGROUND,
            fg=ThemeColours.TEXT,
            shortcut=return_shortcut,
        )
        if return_button_config is not None:
            self._overlay_exit_button.configure(**return_button_config)
        self._overlay_exit_button.pack(side=tk.LEFT, fill=tk.X, expand=False)

        label = tk.Label(self._overlap_panel, text=overlay_label, bg=ThemeColours.BACKGROUND, fg=ThemeColours.TEXT)
        label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        if label_config is not None:
            label.configure(**label_config)

        self._overlay_frame: tk.Frame = tk.Frame(self._overlay_frame_parent, bg=ThemeColours.BACKGROUND)
        self._overlay_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._pre_state_change_callback = pre_state_change_callback
        self._on_state_change_callback = on_state_change_callback

    def get_main_frame(self) -> tk.Frame:
        return self._main_frame

    def get_overlay_frame(self) -> tk.Frame:
        return self._overlay_frame

    def set_overlay_visible(self, visible: bool):
        if self._pre_state_change_callback is not None:
            visible = self._pre_state_change_callback(visible)
        if visible:
            self._main_frame.grid_forget()
            self._overlay_frame_parent.grid(row=0, column=0, sticky=tk.NSEW)
        else:
            self._overlay_frame_parent.grid_forget()
            self._main_frame.grid(row=0, column=0, sticky=tk.NSEW)
        if self._on_state_change_callback is not None:
            self._on_state_change_callback(visible)
