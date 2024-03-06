from contextlib import contextmanager
import tkinter as tk
from PIL import Image, ImageTk
import os


@contextmanager
def hold_tk_show_splash_screen(image_path: str):
    """Wrap this around your imports in your main function."""
    # Ensure the image path is absolute
    image_path = os.path.abspath(image_path)

    # Set up splash screen
    root = tk.Tk()
    root.overrideredirect(True)  # Remove window decorations

    # Get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Load the image
    image = Image.open(image_path)
    photo = ImageTk.PhotoImage(image)

    # Calculate position for splash screen (centered)
    window_width = photo.width()
    window_height = photo.height()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f'{window_width}x{window_height}+{x}+{y}')

    # Create a label to display the image
    label = tk.Label(root, image=photo)
    label.pack()

    # Make sure window is topmost
    root.lift()
    root.attributes('-topmost', True)

    # Display the window
    root.update()

    try:
        yield
    finally:
        # Close splash screen
        root.destroy()
