from contextlib import contextmanager
import tkinter as tk
from typing import Optional

from PIL import Image, ImageTk
import os

from artemis.plotting.tk_utils.tk_basic_utils import hold_tkinter_root_context


@contextmanager
def hold_tk_show_splash_screen(image_path: str, root):

    # Ensure the image path is absolute
    image_path = os.path.abspath(image_path)

    # Initially, hide the main root window
    root.withdraw()

    # Create a Toplevel window for the splash screen
    splash = tk.Toplevel(root)
    splash.overrideredirect(True)  # Remove window decorations

    # Load the image
    image = Image.open(image_path)
    photo = ImageTk.PhotoImage(image)

    # Calculate position for splash screen (centered)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = photo.width()
    window_height = photo.height()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    splash.geometry(f'{window_width}x{window_height}+{x}+{y}')

    # Create a label to display the image and pack it in the splash window.
    label = tk.Label(splash, image=photo)
    label.pack()
    # Make sure splash window is topmost
    splash.lift()
    splash.attributes('-topmost', True)

    # Display the splash window and ensure it updates
    splash.update()

    try:
        yield
    finally:
        # Close the splash screen and clean up
        splash.destroy()
        # Once the splash is destroyed, reveal the main window
        root.deiconify()
