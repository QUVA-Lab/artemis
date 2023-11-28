import tkinter as tk
import webbrowser
from tkinter import messagebox
from typing import Optional, Tuple
import traceback
import requests
from dataclasses import dataclass


# from video_scanner.app_utils.utils import ErrorDetail

@dataclass
class ErrorDetail:
    error: Exception
    traceback: str
    additional_info: str = ""



def send_paste(error_message: str, pastebin_key: str) -> Tuple[bool, Optional[str]]:
    """ Does not work... """
    api_url = 'https://pastebin.com/api/api_post.php'
    paste_name = 'Error Traceback'
    data = {
        'api_dev_key': pastebin_key,
        'api_option': 'paste',
        'api_paste_code': error_message,
        'api_paste_name': paste_name,
        'api_paste_private': 1,  # 0=public, 1=unlisted, 2=private
        'api_paste_expire_date': '1M',

    }
    response = requests.post(api_url, data=data)
    if response.status_code == 200:  # success
        return True, response.text
    else:
        print('Failed to create paste. Status:', response.status_code)
        return False, response.text

def open_url(url):
    webbrowser.open(url)


def tk_show_error_dialog(exception: Exception, title: str = "Error", message: Optional[str] = None, traceback_str: Optional[str] = None,
                         online_help_url: Optional[str] = None, pastebin_key: Optional[str] = None,
                         reporting_email: Optional[str] = None, as_root: bool = False
                         ):

    default_width = 640
    # Define the behavior for "Report and Close"
    def report_and_close():
        if pastebin_key and traceback_str:
            response = send_paste(traceback_str, pastebin_key)
            messagebox.showinfo(title="Reported", message=f"Error reported.  Thank you!")
        root.destroy()

    # Define the behavior for "Close"
    def close():
        root.destroy()

    root = tk.Tk() if as_root else tk.Toplevel()
    root.title(title)
    # Center in parent window and make it a minimum with of 640
    root.geometry(f"+{int(root.winfo_screenwidth() / 2 - default_width//2)}+{int(root.winfo_screenheight() / 2 - 240)}")

    # Show the error message
    # messagebox.showerror(title, str(exception))

    top_row = tk.Frame(root)
    top_row.pack(side=tk.TOP, fill=tk.X, expand=True)

    # Put error icon on left and make it 50x50 pixels.  # TODO: Figure out how to make this reference survive pyinstaller
    # error_icon_relpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "error-icon.png"))
    # if os.path.exists(error_icon_relpath):
    #     error_icon = tk.PhotoImage(file=error_icon_relpath).subsample(4, 4)
    #     tk.Label(top_row, image=error_icon).grid(row=0, column=0, rowspan=2, sticky=tk.W)

    # Show title and error on right and make it stretch
    # tk.Label(top_row, text=title).grid(row=0, column=1, sticky=tk.EW, )
    # tk.Label(top_row, text=str(exception), wraplength=120).grid(row=1, column=1, sticky=tk.EW)
    # Make it wrap to the default width of the window
    message_text = message+'\n'+str(exception) if message else str(exception)
    tk.Label(top_row, text=message_text, wraplength=default_width).grid(row=0, column=1, sticky=tk.EW, columnspan=2)

    # Show the error message
    # tk.Label(root, text=title).pack()
    # tk.Label(root, text=str(exception)).pack()

    # Make a read only entry with the traceback in it
    if traceback_str:
        full_message_str = (message+'\n\n' if message else '')+traceback_str
        print(full_message_str)
        tk.Label(root, text="Error Info:").pack(side=tk.TOP, anchor=tk.W)
        traceback_entry = tk.Text(root, height=10, width=100)
        traceback_entry.insert(tk.END, full_message_str)
        traceback_entry.configure(state='disabled')
        traceback_entry.pack(expand=True, fill=tk.BOTH, side=tk.TOP)

    # Show the online help URL
    # Online help link
    if online_help_url:
        link = tk.Label(root, text="Online Help", fg="blue", cursor="hand2")
        link.pack()
        link.bind("<Button-1>", lambda e: open_url(online_help_url))

    if reporting_email:
        # messagebox.showinfo(title="Report Error", message="Would you like to report this error?")
        # tk.Label(root, text=f"Please copy info and email to {reporting_email}").pack()
        # Make it a clickable email link with a mailto:
        link = tk.Label(root, text=f"Please report this to {reporting_email}", fg="blue", cursor="hand2")
        link.pack()
        link.bind("<Button-1>", lambda e: open_url(f"mailto:{reporting_email}?subject=Error Report&body={traceback_str}"))

    button_row = tk.Frame(root)
    button_row.pack(side=tk.BOTTOM)

    # Add a button to copy the traceback
    if traceback_str:
        def copy_traceback():
            root.clipboard_clear()
            root.clipboard_append(traceback_str)
            email_note = f"\n\nPlease email this to {reporting_email}" if reporting_email else ""
            messagebox.showinfo(title="Copied", message=f"Error info copied to clipboard. "+email_note)

        copy_button = tk.Button(button_row, text="Copy Info", command=copy_traceback)
        copy_button.pack(side=tk.LEFT)


    # If there is a pastebin key
    if pastebin_key:
        # messagebox.showinfo(title="Report Error", message="Would you like to report this error?")
        report_button = tk.Button(button_row, text="Report and Close", command=report_and_close)
        report_button.pack(side=tk.LEFT)

    # Close button
    close_button = tk.Button(button_row, text="Close", command=close)
    close_button.pack(side=tk.LEFT)

    # On top
    root.attributes("-topmost", True)
    print("Starting error dialog")
    root.mainloop() if as_root else root.wait_window()
    print("Exiting error dialog")


def tk_show_eagle_eyes_error_dialog(error_details: ErrorDetail):
    trace_str = error_details.traceback or traceback.format_exc()
    print(trace_str)
    tk_show_error_dialog(
        exception=error_details.error,
        title='Error',
        traceback_str=trace_str,
        message=f"Error: {str(error_details.error)}"+(f"\n\nAdditional Info:\n{error_details.additional_info}" if error_details.additional_info else '')
    )



if __name__ == "__main__":
    try:
        from video_scanner.app_utils.constants import EAGLE_EYES_EMAIL_ADDRESS, EAGLE_EYES_SCAN_PAGE_URL, PASTEBIN_KEY

        raise Exception("Intentional error")
    except Exception as err:
        tk_show_error_dialog(
            exception=err,
            traceback_str=traceback.format_exc(),
            online_help_url=EAGLE_EYES_SCAN_PAGE_URL,
            pastebin_key=PASTEBIN_KEY
        )
