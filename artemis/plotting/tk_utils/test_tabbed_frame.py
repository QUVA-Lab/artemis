from enum import Enum

from artemis.plotting.tk_utils.tabbed_frame import TabbedFrame
from artemis.plotting.tk_utils.tk_basic_utils import hold_tkinter_root_context
import tkinter as tk

def test_exitable_side_frame():

    with hold_tkinter_root_context() as root:

        root.geometry("800x600")

        class Tabs(Enum):
            MAIN = 'Main'
            OVERLAY = 'Overlay'

        frame: TabbedFrame = TabbedFrame(
            parent_frame=root,
            tab_enum=Tabs,
            tab_forward_shortcut='<Shift-Tab>',
            on_button_config=dict(fg='white', bg='gray50'),
            off_button_config=dict(fg='gray50', bg='gray25'),
        )
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        f=tk.Frame(frame.get_frame(Tabs.MAIN), bg='red')
        f.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Put a button in the center of parent frame
        button = tk.Button(f, text="Click Me to Show Overlay", command=lambda: frame.set_active_tab(Tabs.OVERLAY))
        button.pack(side=tk.TOP, fill=tk.NONE, expand=True)

        overlay_frame=tk.Frame(frame.get_frame(Tabs.OVERLAY), bg='blue')
        overlay_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        label = tk.Label(overlay_frame, text="Overlay")
        label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        root.mainloop()


if __name__ == '__main__':
    test_exitable_side_frame()

