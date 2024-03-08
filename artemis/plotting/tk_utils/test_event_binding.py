import tkinter as tk

def recursively_bind(widget, event, handler):
    """
    Recursively bind an event to a widget and all its descendants.

    Args:
    - widget: The root widget to start binding from.
    - event: The event to bind, e.g., '<Escape>'.
    - handler: The function to call when the event occurs.
    """
    widget.bind(event, handler)
    for child in widget.winfo_children():
        recursively_bind(child, event, handler)

def on_any_escape(event):
    print("Escape pressed in:", event.widget)

# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("300x200")

    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    sub_frame = tk.Frame(frame, bg="lightblue")
    sub_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    entry = tk.Entry(sub_frame)
    entry.pack(padx=20, pady=20)

    button = tk.Button(sub_frame, text="Click Me")
    button.pack(pady=10)

    # Bind the Escape key to the frame and all its descendants
    recursively_bind(frame, '<Escape>', on_any_escape)

    root.mainloop()
