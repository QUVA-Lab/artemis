import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable, Sequence, Union, List, Generic

from artemis.plotting.tk_utils.constants import ThemeColours
from artemis.plotting.tk_utils.ui_utils import RespectableLabel, MultiStateToggle, MultiStateEnumType


class TabbedFrame(tk.Frame, Generic[MultiStateEnumType]):
    """ A frame with a "main" frame and an "overlay" frame.  The overlay frame hides the main frame when it is visible."""

    def __init__(self,
                 parent_frame: tk.Frame,
                 tab_enum: type(MultiStateEnumType),
                 initial_state: Optional[MultiStateEnumType] = None,
                 # overlay_label: str = "",
                 pre_state_change_callback: Optional[Callable[[MultiStateEnumType], bool]] = None,
                 on_state_change_callback: Optional[Callable[[MultiStateEnumType], None]] = None,
                 add_tab_bar: bool = True,
                 hide_tab_bar_in_states: Sequence[MultiStateEnumType] = (),
                 # overlay_panel_title: str = "Overlay View",
                 tab_forward_shortcut: Optional[str] = None,
                 on_button_config=None,
                 off_button_config=None,
                 # tab_backward_shortcut: Optional[str] = None,
                 # return_button_config: Optional[dict] = None,
                 # label_config: Optional[dict] = None,
                 ):

        tk.Frame.__init__(self, parent_frame, bg=ThemeColours.BACKGROUND)
        if off_button_config is None:
            off_button_config = dict(fg='gray50', bg='gray25')
        if on_button_config is None:
            on_button_config = dict(fg='white', bg='gray50')
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self._hide_tab_bar_in_states = hide_tab_bar_in_states
        self._tab_enum = tab_enum
        self._on_button_config = on_button_config
        self._off_button_config = off_button_config
        self._tab_forward_shortcut = tab_forward_shortcut
        if initial_state is None:
            initial_state = list(tab_enum)[0]

        self._frames: List[tk.Frame] = []
        for state in tab_enum:
            frame = tk.Frame(self, bg=ThemeColours.BACKGROUND)
            self._frames.append(frame)
            if state == initial_state:
                frame.grid(row=1, column=0, sticky=tk.NSEW)

        self._tab_controls: List[MultiStateToggle] = []  # Controls that switch between the main and overlay frame.  We must keep a reference to them so we switch their states appropriately.
        self._main_tab_control = self.create_tab_switch_control(self)
        if add_tab_bar:
            self._main_tab_control.grid(row=0, column=0, sticky=tk.EW)
        self._pre_state_change_callback = pre_state_change_callback
        self._on_state_change_callback = on_state_change_callback

        self.set_active_tab(initial_state)

        # Callback
        if tab_forward_shortcut is not None:
            self.bind_all(tab_forward_shortcut, lambda e: self.increment_tab_index(1))

    def increment_tab_index(self, increment: int) -> None:
        states = list(self._tab_enum)
        current_index = states.index(self._main_tab_control.get_state())
        new_index = (current_index + increment) % len(states)
        self.set_active_tab(states[new_index])

    def get_frame(self, state: MultiStateEnumType) -> tk.Frame:
        return self._frames[list(self._tab_enum).index(state)]

    def create_tab_switch_control(self, parent: tk.Frame) -> MultiStateToggle:
        """ Create a control that shows the "main" tab and this tab side-by-side
        Clicking tabs (or pressing the shortcut) will switch between them.
        """
        tab_control = MultiStateToggle(
            parent,
            self._tab_enum,
            on_state_change_callback=lambda s: self.set_active_tab(s),
            call_callback_immediately=False,
            on_button_config=self._on_button_config,
            off_button_config=self._off_button_config,
            tooltip_maker=lambda s: f"Switch to '{s.value}' ({self._tab_forward_shortcut.strip('<>') if self._tab_forward_shortcut else ''})",
        )
        # tab_control.pack(side=tk.LEFT, fill=tk.X, expand=False)
        self._tab_controls.append(tab_control)
        return tab_control

    def set_active_tab(self, state: MultiStateEnumType, skip_callback: bool = False) -> None:

        # if skip_if_unchanged and state == self._main_tab_control.get_state():
        #     print(f"Skipping tab change to {state} because it's already in state {self._main_tab_control.get_state()}.")
        #     return  # Already in this state
        do_change = True
        if self._pre_state_change_callback is not None:
            do_change = self._pre_state_change_callback(state)
            if not do_change:
                return

        if state in self._hide_tab_bar_in_states:
            self._main_tab_control.grid_remove()
        else:
            self._main_tab_control.grid(row=0, column=0, sticky=tk.EW)

        if do_change:
            for tc in self._tab_controls:
                tc.set_state(state, skip_callback=True)
            index_to_keep = list(self._tab_enum).index(state)
            for i, f in enumerate(self._frames):  # Just to be safe forget all frames
                f.grid_forget()
            self._frames[index_to_keep].grid(row=1, column=0, sticky=tk.NSEW)
            if self._on_state_change_callback is not None and not skip_callback:  # Avoid recursion
                self._on_state_change_callback(state)


        # Redraw the window
        self.update()
        # self.winfo_toplevel().update()

    def get_active_tab(self) -> MultiStateEnumType:
        return self._main_tab_control.get_state()


# class RespectableNotebook(ttk.Notebook):
#     def __init__(self,
#                  parent_frame,
#                  panel_names: Sequence[str],
#                  initial_panel: Optional[str] = None,
#                  on_state_change_callback: Optional[Callable[[bool], None]] = None,
#                  next_tab_shortcut: Optional[str] = None,
#                  previous_tab_shortcut: Optional[str] = None,
#                  **kwargs):
#         super().__init__(parent_frame, **kwargs)
#
#         # Create the main frame and add it as the first tab
#
#         if initial_panel is None:
#             initial_panel = panel_names[0]
#         assert initial_panel in panel_names, f"Initial panel {initial_panel} not in {panel_names}"
#         for i, p in enumerate(panel_names):
#             frame = tk.Frame(self, bg=ThemeColours.BACKGROUND)
#             tab_id = self.add(frame, text=p)
#             if p == initial_panel:
#                 self.select(i)
#
#
#     def get_frame(self, name_or_index: Union[str, int]) -> tk.Frame:
#         if isinstance(name_or_index, str):
#             return self.nametowidget(self.select())
#         else:
#             return self.nametowidget(self.select())