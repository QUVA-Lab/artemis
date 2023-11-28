import traceback
from dataclasses import fields, dataclass, field
from functools import partial
from tkinter import filedialog, messagebox
from typing import TypeVar, Optional
import tkinter as tk
import os

from artemis.plotting.tk_utils.constants import ThemeColours
from artemis.plotting.tk_utils.tooltip import create_tooltip

ParametersType = TypeVar('ParametersType')

def ui_choose_parameters(
        params_type: type,  # Some sort of dataclass (the class object)
        initial_params: Optional[ParametersType] = None,
        factory_reset_params: Optional[ParametersType] = None,
        title: str = "Select Parameters",
        prompt: str = "Hover mouse over for description of each parameter.  Tab to switch fields, Enter to accept, Escape to cancel."

) -> Optional[ParametersType]:
    """ Load, edit, save, and return the settings. """


    chosen_params = params_type() if initial_params is None else initial_params

    window = tk.Toplevel()
    # Set minimum width to 600px
    window.minsize(800, 1)
    # Make it fill parent
    # window.grid_columnconfigure(0, weight=1)
    window.grid_columnconfigure(1, weight=1)

    # window.geometry("800x500")
    window.title(title)

    label = tk.Label(window, text=prompt)
    label.grid(column=0, row=0, columnspan=2)

    var = {}
    for i, f in enumerate(fields(params_type), start=1):
        label = tk.Label(window, text=f.metadata['name'] if 'name' in f.metadata else f.name.replace("_", " ").capitalize() + ":")
        create_tooltip(label, f.metadata.get("help", "No help available."), background=ThemeColours.HIGHLIGHT_COLOR)
        label.grid(column=0, row=i)

        # Stre
        # Depending on the type of the field, we'll need to do something different.
        if f.type == str:  # Entry
            metadata_type = f.metadata.get("type", None)
            var[f.name] = tk.StringVar(value=getattr(chosen_params, f.name))
            if metadata_type is None:
                entry = tk.Entry(window, textvariable=var[f.name])
            elif metadata_type in ["file", "directory"]:
                # Store rel stores relative to the default directory, so that if the program is moved, the path is still valid.
                store_rel = f.metadata.get("store_relative", False)
                default_directory = f.metadata.get("default_directory", None)
                # if store_rel and default_directory is not None:
                #     var[f.name].set(os.path.join(default_directory, getattr(settings, f.name)))
                entry = tk.Entry(window, textvariable=var[f.name])

                def browse(v: tk.StringVar, for_directory: bool = False, default_directory: Optional[str] = None, store_rel: bool = False):
                    print(f"Default directory: {default_directory}")
                    path = filedialog.askdirectory(initialdir=default_directory) if for_directory else filedialog.askopenfilename(initialdir=default_directory)
                    if path:
                        if store_rel:
                            path = os.path.relpath(path, default_directory)
                        v.set(path)
                browse_button = tk.Button(window, text="Browse", command=partial(browse,
                                                                                 v=var[f.name],
                                                                                 for_directory=metadata_type == "directory",
                                                                                 default_directory = default_directory,
                                                                                 store_rel=store_rel
                                                                                 ))
                browse_button.grid(column=2, row=i, sticky="ew")
            else:
                raise NotImplementedError(f"Unknown metadata type {metadata_type}")

            if i==0:  # Make it so keyboard is directed to this right away
                entry.focus_set()
            entry.grid(column=1, row=i, sticky="ew", )

        elif f.type == bool:
            var[f.name] = tk.BooleanVar(value=getattr(chosen_params, f.name))
            check_box = tk.Checkbutton(window, variable=var[f.name])
            # check_box.grid(column=1, row=i)
            # Align left...
            check_box.grid(column=1, row=i, sticky="w")
        elif f.type == int:
            var[f.name] = tk.IntVar(value=getattr(chosen_params, f.name))
            entry = tk.Entry(window, textvariable=var[f.name])
            entry.grid(column=1, row=i, sticky="w")
        elif f.type == float:
            var[f.name] = tk.DoubleVar(value=getattr(chosen_params, f.name))
            entry = tk.Entry(window, textvariable=var[f.name])
            entry.grid(column=1, row=i, sticky="w")

        else:
            raise NotImplementedError(f"Type {f.type} not supported.")

    def read_settings_object() -> Optional[params_type]:
        """ Read the settings object from the UI. """
        try:
            settings_dict = {}
            for f in fields(params_type):
                settings_dict[f.name] = var[f.name].get()
            return params_type(**settings_dict)
        except Exception as e:
            messagebox.showerror("Error", f"Error reading settings from UI:\n\n {e} \n\n(see Log)")
            print(traceback.format_exc())
            return None

    # Ok, lets get buttons for "Cancel", "Update", and "Factory Reset", with Default being "save"
    def cancel():
        nonlocal chosen_params
        chosen_params = None
        window.destroy()

    def ok():
        nonlocal chosen_params
        new_settings = read_settings_object()
        if new_settings is not None:
            chosen_params = new_settings
            window.destroy()

    def factory_reset():
        nonlocal chosen_params
        window.destroy()
        # new_settings = ui_load_edit_save_get_settings(factory_reset=True)
        chosen_params = factory_reset_params

    button_row = tk.Frame(window)
    button_row.grid(column=0, row=100, columnspan=3)


    # cancel_button = tk.Button(window, text="Cancel", command=cancel)
    # cancel_button.grid(column=0, row=100)
    # update_button = tk.Button(window, text="Update", command=update)
    # update_button.grid(column=1, row=100)
    # factory_reset_button = tk.Button(window, text="Factory Reset", command=factory_reset)
    # factory_reset_button.grid(column=2, row=100)

    # Set the focus on the first parameter
    # var[fields(params_type)[0].name].focus_set()

    cancel_button = tk.Button(button_row, text="Cancel", command=cancel)
    cancel_button.grid(column=0, row=0)
    ok_button = tk.Button(button_row, text="Ok", command=ok, default=tk.ACTIVE)
    ok_button.grid(column=1, row=0)
    if factory_reset_params is not None:
        factory_reset_button = tk.Button(button_row, text="Factory Reset", command=factory_reset)
        factory_reset_button.grid(column=2, row=0)

    # Put on top
    window.attributes('-topmost', True)

    window.focus_force()
    ok_button.focus_set()
    window.bind('<Return>', lambda event: ok_button.invoke())
    window.bind('<Escape>', lambda event: cancel_button.invoke())

    window.wait_window()

    return chosen_params


FieldType = TypeVar("FieldType", bound=object)

nodefault = object()

def ui_choose_field(
        name: str,
        dtype: type(FieldType),
        default: Optional[FieldType] = nodefault,
        title: str = "Choose Value",
        prompt: str = "Choose a value ",
        tooltip: Optional[str] = None
) -> FieldType:

    @dataclass
    class TempClass:
        tempfield: dtype = field(default=default, metadata=dict(help=tooltip, name=name))

    result = ui_choose_parameters(
        params_type=TempClass,
        initial_params=TempClass(tempfield=default) if default is not nodefault else None,
        title=title,
        prompt=prompt
    )
    return result.tempfield if result is not None else None



if __name__ == "__main__":

    # @dataclass
    # class MyParams:
    #     some_float: float = 4
    #     some_int: int = field(default=3, metadata=dict(help="Select some integer"))
    #     some_file: str = field(default=os.path.expanduser("~/some_image.jpg"), metadata=dict(type='file'))
    #
    # result = ui_choose_parameters(params_type=MyParams)
    # print(result)

    result = ui_choose_field('N neighbours', int, default=40)
    print(result)