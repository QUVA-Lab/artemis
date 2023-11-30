from artemis.plotting.tk_utils.exitable_overlay_frame import ExitableOverlayFrame
from video_scanner.ui.tk_utils import hold_tkinter_root_context
import tkinter as tk

def test_exitable_side_frame():

    with hold_tkinter_root_context() as root:

        root.geometry("800x600")

        frame: ExitableOverlayFrame = ExitableOverlayFrame(root)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        f=tk.Frame(frame.get_main_frame(), bg='red')
        f.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Put a button in the center of parent frame
        button = tk.Button(f, text="Click Me to Show Overlay", command=lambda: frame.set_overlay_visible(True))
        button.pack(side=tk.TOP, fill=tk.NONE, expand=True)

        overlay_frame=tk.Frame(frame.get_overlay_frame(), bg='blue')
        overlay_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        label = tk.Label(overlay_frame, text="Overlay")
        label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        root.mainloop()


if __name__ == '__main__':
    test_exitable_side_frame()

