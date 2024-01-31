import os
import tkinter as tk
from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass, field, fields, replace
import pprint
from tkinter import messagebox, ttk
from typing import Optional, TypeVar, Sequence, get_origin, get_args, Any, Dict, Callable, Union, Generic, Tuple, Mapping

from more_itertools import first
from more_itertools.more import first

# Assuming the required modules from artemis.plotting.tk_utils are available
from artemis.plotting.tk_utils.constants import ThemeColours
from artemis.plotting.tk_utils.tk_utils import hold_tkinter_root_context
from artemis.plotting.tk_utils.tooltip import create_tooltip
from artemis.plotting.tk_utils.ui_utils import ButtonPanel
from video_scanner.ui.ui_utils import RespectableLabel

ParametersType = TypeVar('ParametersType')


def get_default_for_param_type(param_type: type) -> Any:
    if param_type == bool:
        return False
    elif isinstance(get_origin(param_type), type) and issubclass(get_origin(param_type), Sequence):
        return []
    elif isinstance(get_origin(param_type), type) and is_optional_type(param_type):
        return None
    elif hasattr(param_type, "__dataclass_fields__"):
        return param_type()
    elif param_type == str:
        return ""
    elif param_type == int:
        return 0
    elif param_type == float:
        return 0.0
    else:
        raise NotImplementedError(f"Type {param_type} not supported.")


class MockVariable(tk.Variable):

    def __init__(self, parent: tk.Widget, initial_value: Any = None):
        super().__init__(parent)
        self.value = initial_value
        self._write_callback: Optional[Callable[[Any], None]] = None

    def get(self):
        return self.value

    def set(self, value):
        self.value = value
        if self._write_callback is not None:
            self._write_callback(value)

    def trigger_write_callback(self):
        if self._write_callback is not None:
            self._write_callback(self.value)

    def trace_add(self, mode: str, callback: Callable[[Any, Any, Any], None]):
        if mode == "write":
            self._write_callback = callback
        else:
            raise NotImplementedError(f"Mode {mode} not supported.")


class IParameterSelectionFrame(tk.Frame, Generic[ParametersType], metaclass=ABCMeta):

    @abstractmethod
    def get_filled_parameters(self) -> Optional[ParametersType]:
        raise NotImplementedError()

    def get_variables(self) -> Mapping[Tuple[Union[int, str], ...], tk.Variable]:
        return {}


class EntryParameterSelectionFrame(IParameterSelectionFrame):

    def __init__(self, master: tk.Widget, builder: 'ParameterUIBuilder'):
        super().__init__(master, **builder.general_kwargs)
        self._builder = builder
        self.var = tk.StringVar(master=self, value=self._builder.initial_value) if self._builder.param_type == str \
            else tk.DoubleVar(master=self, value=self._builder.initial_value) if self._builder.param_type == float else \
            tk.IntVar(master=self, value=self._builder.initial_value)
        entry = tk.Entry(self, textvariable=self.var)
        entry.grid(column=0, row=0, sticky="ew")

    def get_filled_parameters(self) -> ParametersType:
        return self.var.get()

    def get_variables(self) -> Mapping[Tuple[Union[int, str], ...], tk.Variable]:
        return {(): self.var}



# def build_frame_with_added_widget(parent: tk.Widget, builder: 'ParameterUIBuilder', added_button_builder: Optional[Callable[[tk.Frame], tk.Widget]] = None) -> tk.Frame:
#     frame = tk.Frame(parent)
#     frame.grid(column=0, row=0, sticky="ew")
#     # frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
#     builder.build_parameter_frame(frame)
#     if added_button_builder is not None:
#         widget = added_button_builder(frame)
#         widget.grid(column=1, row=0, sticky="ew")
#     return frame


class AddedWidgetParameterSelectionFrame(IParameterSelectionFrame[ParametersType]):

    def __init__(self, master: tk.Widget, builder: 'ParameterUIBuilder', added_button_builder: Optional[Callable[[tk.Frame, 'ParameterUIBuilder'], tk.Widget]] = None):
        super().__init__(master, **builder.general_kwargs)
        self._builder = builder
        self._initial_value = builder.initial_value
        # Leftward text alignmnent
        # label = RespectableLabel(self, text=str(builder.initial_value), anchor=tk.W, justify=tk.LEFT)

        self._child = builder.remove_custom_constructor_matching_path().build_parameter_frame(self)

        self._child.grid(column=0, row=0, sticky="ew")
        if added_button_builder is not None:
            widget = added_button_builder(self, self._builder)
            widget.grid(column=1, row=0, sticky="ew")

    @classmethod
    def make_constructor(cls, added_button_builder: Optional[Callable[[tk.Frame, 'ParameterUIBuilder'], tk.Widget]] = None) -> Callable[[tk.Widget, 'ParameterUIBuilder'], 'IParameterSelectionFrame']:
        def constructor(master: tk.Widget, builder: 'ParameterUIBuilder') -> tk.Widget:
            return cls(master, builder, added_button_builder=added_button_builder)
        return constructor

    def get_filled_parameters(self) -> ParametersType:
        return self._child.get_filled_parameters()



class BooleanParameterSelectionFrame(IParameterSelectionFrame[bool]):

    def __init__(self, master: tk.Widget, builder: 'ParameterUIBuilder'):
        super().__init__(master, **builder.general_kwargs)
        self.var = tk.BooleanVar(master=self, value=self._builder.initial_value)
        check_box = tk.Checkbutton(self, variable=self.var, state=tk.NORMAL if builder.editable_fields else tk.DISABLED)
        check_box.grid(column=0, row=0, sticky="w")

    def get_filled_parameters(self) -> bool:
        return self.var.get()

    def get_variables(self) -> Mapping[Tuple[Union[int, str], ...], tk.Variable]:
        return {(): self.var}


def is_fixed_size_tuple(param_type: type) -> bool:
    """
    Checks if a type is a fixed-size tuple, e.g. Tuple[int, str] or Tuple[int, str, float]
    Examples of things that are not fixed-size tuples are Tuple[int, ...] and Tuple[int, str, ...]
    """
    # So if last arg is elipsis, it's not fixed size
    return isinstance(get_origin(param_type), type) and issubclass(get_origin(param_type), Tuple) and not get_args(param_type)[-1] is Ellipsis


class SequenceParameterSelectionFrame(IParameterSelectionFrame[Sequence[Any]]):

    def __init__(self, master: tk.Widget, builder: 'ParameterUIBuilder', allow_editing: bool = True):
        super().__init__(master, **builder.general_kwargs)
        self._builder = builder
        self.var = MockVariable(self, list(self._builder.initial_value))
        # self._current_values = list(self._builder.initial_value)
        # self._param_type = self._builder.param_type
        self._child_frames: Sequence[IParameterSelectionFrame] = []
        # self._editable_fields = self._builder.editable_fields
        # self._include_add_option = include_add_option

        self._rebuild_child_frames()

    def _rebuild_child_frames(self):
        self._child_frames = []

        is_editable = self._builder.is_path_matching_editable_fields()
        is_dynamic_type = not is_fixed_size_tuple(self._builder.param_type)
        is_dynamic = is_dynamic_type and is_editable
        for child in self.winfo_children():
            child.destroy()
        for i, item in enumerate(self._builder.initial_value):
            row = i * 2

            # Big minus sign:
            if is_dynamic:
                remove_button = RespectableLabel(self, text="➖", command=lambda i=i: self._remove_item(i))
                remove_button.grid(column=0, row=row, sticky="ew")
                label = RespectableLabel(self, text=str(i), anchor=tk.W)
                label.grid(column=1, row=row, sticky="ew")

            child_frame = self._builder.modify_for_subfield(subfield_index=i, subfield_type=get_args(self._builder.param_type)[0]).build_parameter_frame(self)

            # child_frame = build_parameter_frame(self, self._param_type, item, editable_fields=editable_subfields)
            child_frame.grid(column=2, row=row, sticky="ew")

            # Horizontal line after, but not after the last one
            separator = ttk.Separator(self, orient=tk.HORIZONTAL)
            separator.grid(column=0, row=row + 1, columnspan=3, sticky="ew")

            self._child_frames.append(child_frame)
        if is_dynamic and self._builder.allow_growing_collections:
            add_button = RespectableLabel(self, text="➕", command=self._add_item)
            add_button.grid(column=0, row=len(self._builder.initial_value) * 2, sticky="ew")
            label = RespectableLabel(self, text=str(len(self._builder.initial_value)) + "...", anchor=tk.W)
            label.grid(column=1, row=len(self._builder.initial_value) * 2, sticky="ew")

    def _add_item(self):
        # self._builder.initial_value.append(get_default_for_param_type(self._param_type))
        self._builder = replace(self._builder, initial_value=list(self._builder.initial_value) + [get_default_for_param_type(self._builder.param_type)])
        self._rebuild_child_frames()
        self.var.trigger_write_callback()

    def _remove_item(self, index: int):
        # del self._current_values[index]
        self._builder = replace(self._builder, initial_value=[*self._builder.initial_value[:index], *self._builder.initial_value[index + 1:]])
        self._rebuild_child_frames()
        self.var.trigger_write_callback()

    def get_filled_parameters(self) -> Sequence[Any]:
        return [child.get_filled_parameters() for child in self._child_frames]

    def get_variables(self) -> Mapping[Tuple[Union[int, str], ...], tk.Variable]:
        return {(): self.var, **{(i,) + k: v for i, child in enumerate(self._child_frames) for k, v in child.get_variables().items()}}


class OptionalParameterSelectionFrame(IParameterSelectionFrame[Optional[Any]]):

    def __init__(self, master: tk.Widget, builder: 'ParameterUIBuilder'):
        super().__init__(master, **builder.general_kwargs)
        self._builder = builder
        self._checkbox_var = tk.BooleanVar(value=self._builder.initial_value is not None)
        check_box = tk.Checkbutton(self, variable=self._checkbox_var)
        check_box.grid(column=0, row=0, sticky="w")
        # self._child_frame = build_parameter_frame(self, param_type, initial_value if initial_value is not None else get_default_for_param_type(param_type), editable_fields=editable_fields)
        self._child_frame = replace(self._builder,
                                    param_type=get_args(self._builder.param_type)[0],
                                    initial_value=self._builder.initial_value if self._builder.initial_value is not None else get_default_for_param_type(get_args(self._builder.param_type)[0]),
                                    ).build_parameter_frame(self)
        self._on_checkbox_change(self._checkbox_var.get())

    def _on_checkbox_change(self, new_value: bool):
        if new_value:
            self._child_frame.grid(column=1, row=0, sticky="ew")
        else:
            self._child_frame.grid_forget()

    def get_filled_parameters(self) -> Optional[Any]:
        return self._child_frame.get_filled_parameters() if self._checkbox_var.get() else None

    def get_variables(self) -> Mapping[Tuple[Union[int, str], ...], tk.Variable]:
        return self._child_frame.get_variables() if self._checkbox_var.get() else {}


class ButtonParameterSelectionFrame(IParameterSelectionFrame[ParametersType]):
    """ Parameter is summarized in a clickable label which, when clicked, can open up an edit menu
     with ui_choose_parameters
     """

    def __init__(self, master: tk.Widget, builder: 'ParameterUIBuilder'):
        super().__init__(master, **builder.general_kwargs)
        self._builder = builder
        self.var = MockVariable(self, initial_value=self._builder.initial_value)
        # self._param_type = param_type
        # self._current_value = initial_value
        self._label = RespectableLabel(self, anchor=tk.W, command=self._on_click, text="")
        self._label.grid(column=0, row=0, sticky="ew")
        # self._editable_fields = [e for e in editable_fields if e]
        # self.var.set(self._builder.initial_value)
        self._update_label()

    def _update_label(self):
        self._label.configure(text=str(self.var.get()))

    def _on_click(self):
        new_value = ui_choose_parameters(builder=self._builder)
        if new_value is not None:
            self.var.set(new_value)
            self._update_label()

    def get_filled_parameters(self) -> ParametersType:
        return self.var.get()

    def get_variables(self) -> Mapping[Tuple[Union[int, str]], tk.Variable]:
        return {(): self.var}


class DataclassParameterSelectionFrame(IParameterSelectionFrame[ParametersType]):

    def __init__(self, master: tk.Widget, builder: 'ParameterUIBuilder'):
        super().__init__(master, **builder.general_kwargs)
        self._builder = builder
        flds = fields(self._builder.param_type)
        # self.params_type = self._param_builder.param_type
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=3)
        self.rowconfigure(1, weight=1)
        # self.initial_params = initial_value

        self._child_frames = []

        for row_index, f in enumerate(flds):
            row = 2 * row_index
            label = RespectableLabel(self, text=f.metadata['name'] if 'name' in f.metadata else f.name.replace("_", " ").capitalize() + ":",
                                     tooltip=f.metadata.get("help", "No help available."))
            label.grid(column=0, row=row, sticky="ew")

            # editable_subfields = [e for e in editable_fields if len((x := e.split("."))) and x[0] == f.name] if isinstance(editable_fields, Sequence) else editable_fields
            # editable_subfields = filter_subfields(editable_fields, f.name)
            frame = self._builder.modify_for_subfield(subfield_index=f.name, subfield_type=f.type).build_parameter_frame(self)
                # parent=self,
                # param_type=f.type,
                # initial_value=getattr(initial_value, f.name) if initial_value is not None else None,
                # editable_fields=filter_subfields(editable_fields, f.name),

            frame.grid(column=1, row=row, sticky="ew")
            self._child_frames.append(frame)

            # Horizontal line after, but not after the last one
            if row_index < len(flds) - 1:
                separator = ttk.Separator(self, orient=tk.HORIZONTAL)
                separator.grid(column=0, row=row + 1, columnspan=3, sticky="ew")

    def get_filled_parameters(self) -> ParametersType:
        return replace(self._builder.initial_value, **{f.name: child.get_filled_parameters() for f, child in zip(fields(self._builder.param_type), self._child_frames)})

    def get_variables(self) -> Mapping[Tuple[Union[int, str]], tk.Variable]:
        return {(f.name,) + k: v for f, child in zip(fields(self._builder.param_type), self._child_frames) for k, v in child.get_variables().items()}


class UneditableParameterSelectionFrame(IParameterSelectionFrame[ParametersType]):

    def __init__(self, master: tk.Widget, builder: 'ParameterUIBuilder', added_button_builder: Optional[Callable[[tk.Frame], tk.Widget]] = None):
        super().__init__(master, **builder.general_kwargs)
        self._builder = builder
        self._initial_value = builder.initial_value
        # Leftward text alignmnent
        label = RespectableLabel(self, text=str(builder.initial_value), anchor=tk.W, justify=tk.LEFT)
        label.grid(column=0, row=0, sticky="ew")
        if added_button_builder is not None:
            widget = added_button_builder(self)
            widget.grid(column=1, row=0, sticky="ew")

    def get_filled_parameters(self) -> ParametersType:
        return self._initial_value


class WrapperParameterSelectionFrame(IParameterSelectionFrame[ParametersType]):

    def __init__(self, master: tk.Widget, frame: IParameterSelectionFrame):
        super().__init__(master)

        self._frame = frame
        # self._child_frame = self._builder.modify_for_subfield(subfield_index=0, subfield_type=self._builder.param_type).build_parameter_frame(self)
        # self._child_frame.grid(column=0, row=0, sticky="ew")

    def get_filled_parameters(self) -> ParametersType:
        return self._frame.get_filled_parameters()


def is_optional_type(param_type: type) -> bool:
    return get_origin(param_type) is Union and type(None) in get_args(param_type)



def does_field_match_pattern(
        field_path: str,   # e.g. "a.b.c"
        pattern: str,   # e.g. "a.*.c"
        include_subpatterns: bool = False  # If True, then field a.b will match the pattern a.b.c
) -> bool:
    """ Check if a field matches a pattern, e.g. does_field_match_pattern("a.b.c", "a.*.c") -> True

    If include_subpatterns is True,
        field_path 'a.b' will match pattern 'a.b.c'
        but note that
        field_path 'a.b.c' will NOT match pattern 'a.b'
    """
    field_path_parts = field_path.lstrip('.').split(".")
    pattern_parts = pattern.split(".")

    if (not include_subpatterns) and len(field_path_parts) < len(pattern_parts):
        return False

    for field_path_part, pattern_part in zip(field_path_parts, pattern_parts):
        if pattern_part != "*" and field_path_part != pattern_part:
            return False

    return True


def filter_subfields(original_editable_fields: Union[Sequence[str], bool], field_name: Union[int, str]) -> Union[Sequence[str], bool]:
    if isinstance(original_editable_fields, bool):
        return original_editable_fields
    else:
        matching_fields = [tuple(x) for e in original_editable_fields if e and (x := e.split(".")) and (x[0] == str(field_name) or x[0] == "*")]

        if matching_fields == [(str(field_name),)]:
            return True
        elif not matching_fields:
            return False
        else:
            return ['.'.join(x[1:]) for x in matching_fields if len(x) >= 1]


@dataclass
class ParameterUIBuilder:
    """ A class that builds a UI for a parameter. """
    # parent: Optional[tk.Widget]  # The parent widget
    param_type: type  # The type of the parameter.  If None, initial_value must be provided.
    initial_value: Any  # The initial value to display.  If None, param_type must be provided.
    path: str = ''  # The path to the parameter, e.g. "a.b.c" means the "c" field of the "b" field of the "a" field.
    editable_fields: Union[bool, Sequence[str]] = True  # Either
    end_field_patterns: Sequence[str] = ()  # Paths to fields that should not be expanded upon
    allow_growing_collections: bool = False  # If True, then if the parameter is a collection, you can add new elements to it.
    custom_constructors: Mapping[str, Callable[[tk.Widget, 'ParameterUIBuilder'], IParameterSelectionFrame]] = field(default_factory=dict)
    general_kwargs: Dict[str, Any] = field(default_factory=dict)
    on_change_callback: Optional[Callable[[Tuple[Union[str, int]], tk.Variable], None]] = None
    custom_widget_constructors: Mapping[str, Callable[[tk.Widget, 'ParameterUIBuilder'], tk.Widget]] = field(default_factory=dict)
    # A callback in the form f(path, variable) that will be called whenever a field changes.
    # The path is a tuple of strings and ints that describes the path to the variable, e.g. ("a", 1, "b") means the
    # variable is the "b" field of the 1st element of the "a" field.  The variable is the tk.Variable that was changed.

    def remove_custom_constructor_matching_path(self) -> 'ParameterUIBuilder':
        return replace(self, custom_constructors={pattern: constructor for pattern, constructor in self.custom_constructors.items() if not does_field_match_pattern(self.path, pattern)})

    def is_path_matching_editable_fields(self, include_subpatterns=False) -> bool:
        return self.editable_fields is True or not isinstance(self.editable_fields, bool) and any(does_field_match_pattern(self.path, pattern, include_subpatterns=include_subpatterns) for pattern in self.editable_fields)

    def get_extra_widget_or_none(self, path: str) -> Optional[tk.Widget]:
        matching_pattern = first((pattern for pattern in self.extra_widget_builders if does_field_match_pattern(path, pattern)), None)
        if matching_pattern is not None:
            builder = self.extra_widget_builders[matching_pattern]
            return builder(path)
        else:
            return None

    def is_end_field(self) -> bool:
        return any(does_field_match_pattern(self.path, pattern) for pattern in self.end_field_patterns)

    def modify_for_subfield(self, subfield_index: Union[int, str], subfield_type: type) -> 'ParameterUIBuilder':
        return replace(
            self,
            # parent=parent,
            path=self.path + "." + str(subfield_index),
            initial_value=getattr(self.initial_value, subfield_index) if isinstance(subfield_index, str) else self.initial_value[subfield_index],
            param_type=subfield_type,
        )

    def modify_for_new_menu_window(self) -> 'ParameterUIBuilder':
        """ Modify the builder for a new window that is opened after clicking on an editable end-field"""
        return replace(self, end_field_patterns=[e for e in self.end_field_patterns if not does_field_match_pattern(self.path, e)])

    def build_parameter_frame(self, parent: tk.Widget) -> IParameterSelectionFrame:
        """
        Build a TKinter frame that lets you view and edit parameters.

        :param parent: The parent widget
        :param param_type: The type of the parameter.  If None, initial_value must be provided.
        :param initial_value: The initial value to display.  If None, param_type must be provided.
        :param editable_fields: Either
            - True: All fields (or THE field if it is a single value) are editable
            - False: All fields (or THE field if it is a single value) are NOT editable
            - A sequence of strings: Only the fields with those names are editable.
              The strings can be nested, e.g. "a.b.c" will make the "c" field of the "b" field of the "a" field editable.
              "*" in the nested strings means "all fields", e.g. "a.*.c" will make the "c" field of all the "a" fields editable.
            - EMPTY: No fields are editable (same as False)
        :param on_change_callback: A callback in the form f(path, variable) that will be called whenever a field changes.
            The path is a tuple of strings and ints that describes the path to the variable, e.g. ("a", 1, "b") means the
            variable is the "b" field of the 1st element of the "a" field.  The variable is the tk.Variable that was changed.
        :param kwargs:
        :return:
        """
        # assert self.parent is not None, "Parent must be provided."

        if self.param_type is None:
            assert self.initial_value is not None, "If params_type is None, initial_value must be provided."
            param_type=type(self.initial_value)
        else:
            param_type = self.param_type

        if (constructor:=first((f for pattern, f in self.custom_constructors.items() if does_field_match_pattern(self.path, pattern)), None)) is not None:
            return constructor(parent, self)
        elif param_type in [str, int, float, bool]:  # It's just a single value we don't have to think about whether to break in
            if not self.is_path_matching_editable_fields():  # If we're not editing anything, just show the value
                frame = UneditableParameterSelectionFrame(parent, builder=self)
            elif param_type == bool:  # Start with shallow objects
                frame = BooleanParameterSelectionFrame(parent, builder=self)
            elif param_type == str or param_type == int or param_type == float:
                frame = EntryParameterSelectionFrame(parent, builder=self)
            else:
                raise NotImplementedError(f"Type {param_type} not supported.")
        else:  # We need to break in
            # If we're not editing anything, just show the value
            if self.is_end_field():  # We do not recurse further down, but leave it as a label
                # If the field or subfields are editable, make it a button
                if self.is_path_matching_editable_fields(include_subpatterns=True):
                    frame = ButtonParameterSelectionFrame(parent, builder=self.modify_for_new_menu_window())
                else:
                    frame = UneditableParameterSelectionFrame(parent, builder=self)
            elif isinstance(get_origin(param_type), type) and issubclass(get_origin(param_type), Sequence):  # Now nested objects
                frame = SequenceParameterSelectionFrame(parent, builder=self)
            elif isinstance(get_origin(param_type), type) and is_optional_type(param_type):
                frame = OptionalParameterSelectionFrame(parent, builder=self)
            elif hasattr(param_type, "__dataclass_fields__"):
                frame = DataclassParameterSelectionFrame(parent, builder=self)
            elif is_optional_type(param_type):
                frame = OptionalParameterSelectionFrame(parent, builder=self)
            else:
                raise NotImplementedError(f"Type {param_type} not supported.")

        # extra_widget = self.get_extra_widget_or_none(self.path)
        # if extra_widget is not None:
        #     extra_widget.grid(column=2, row=0, sticky="ew")  # Note - assumes parent is a grid

        # Thin outline, align to top-left
        # frame.configure(borderwidth=1, relief=tk.SOLID, highlightthickness=1, highlightbackground=ThemeColours.HIGHLIGHT_COLOR)

        # Handle extra widgets - requires inserting a parent frame to contain the extra widget
        # extra_widget_pattern_or_none = first((pattern for pattern in self.extra_widget_builders if does_field_match_pattern(self.path, pattern)), None)
        # if extra_widget_pattern_or_none is not None:
        #     # print(f"Got pattern {extra_widget_pattern_or_none} for path {self.path}")
        #     grandparent = parent
        #     parent = tk.Frame(grandparent)
        #     parent.grid(column=0, row=0, sticky="ew")
        #     # parent.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        #     widget = self.extra_widget_builders[extra_widget_pattern_or_none](grandparent, self.path)
        #     widget.grid(column=1, row=0, sticky="ew")
        #     # widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        #     return WrapperParameterSelectionFrame(parent, frame=frame)
        # else:
        #
        #     return frame
        return frame


# def build_parameter_frame(
#         parent: tk.Widget,
#         param_type: Optional[type],
#         initial_value: Any,
#         editable_fields: Union[bool, Sequence[str]] = True,
#         extra_widget_builders = {},
#         # on_change_callback: Optional[Callable[[Tuple[Union[str, int]], tk.Variable], None]] = None,
#         **kwargs
#     ) -> IParameterSelectionFrame:
#     """
#     Build a TKinter frame that lets you view and edit parameters.
#
#     :param parent: The parent widget
#     :param param_type: The type of the parameter.  If None, initial_value must be provided.
#     :param initial_value: The initial value to display.  If None, param_type must be provided.
#     :param editable_fields: Either
#         - True: All fields (or THE field if it is a single value) are editable
#         - False: All fields (or THE field if it is a single value) are NOT editable
#         - A sequence of strings: Only the fields with those names are editable.
#           The strings can be nested, e.g. "a.b.c" will make the "c" field of the "b" field of the "a" field editable.
#           "*" in the nested strings means "all fields", e.g. "a.*.c" will make the "c" field of all the "a" fields editable.
#         - EMPTY: No fields are editable (same as False)
#     :param on_change_callback: A callback in the form f(path, variable) that will be called whenever a field changes.
#         The path is a tuple of strings and ints that describes the path to the variable, e.g. ("a", 1, "b") means the
#         variable is the "b" field of the 1st element of the "a" field.  The variable is the tk.Variable that was changed.
#     :param kwargs:
#     :return:
#     """
#     if param_type is None:
#         assert initial_value is not None, "If params_type is None, initial_value must be provided."
#         param_type = type(initial_value)
#
#     is_leaf_param = param_type in [str, int, float, bool]
#
#     if is_leaf_param:  # It's just a single value we don't have to think about whether to break in
#         if not editable_fields:  # If we're not editing anything, just show the value
#             frame = UneditableParameterSelectionFrame(parent, param_type, initial_value, **kwargs)
#         elif param_type == bool:  # Start with shallow objects
#             frame = BooleanParameterSelectionFrame(parent, initial_value, **kwargs)
#         elif param_type == str or param_type == int or param_type == float:
#             frame = EntryParameterSelectionFrame(parent, param_type, initial_value, **kwargs)
#         else:
#             raise NotImplementedError(f"Type {param_type} not supported.")
#     else:  # We need to break in
#         # If we're not editing anything, just show the value
#         if editable_fields is False:
#             frame = UneditableParameterSelectionFrame(parent, param_type, initial_value, **kwargs)
#         elif editable_fields is True or '' in editable_fields:  # If we're editing everything, or the top-level object, then we can edit it
#             return ButtonParameterSelectionFrame(parent, param_type, initial_value, editable_fields=editable_fields, **kwargs)
#         elif isinstance(get_origin(param_type), type) and issubclass(get_origin(param_type), Sequence):  # Now nested objects
#             frame = SequenceParameterSelectionFrame(parent, get_args(param_type)[0], initial_value, editable_fields=editable_fields, **kwargs)
#         elif isinstance(get_origin(param_type), type) and is_optional_type(param_type):
#             frame = OptionalParameterSelectionFrame(parent, get_args(param_type)[0], initial_value, editable_fields=editable_fields, **kwargs)
#         elif hasattr(param_type, "__dataclass_fields__"):
#             frame = DataclassParameterSelectionFrame(parent, param_type, initial_value, editable_fields=editable_fields, **kwargs)
#         elif is_optional_type(param_type):
#             frame = OptionalParameterSelectionFrame(parent, get_args(param_type)[0], initial_value, editable_fields=editable_fields, **kwargs)
#         else:
#             raise NotImplementedError(f"Type {param_type} not supported.")
#
#     # Thin outline, align to top-left
#     # frame.configure(borderwidth=1, relief=tk.SOLID, highlightthickness=1, highlightbackground=ThemeColours.HIGHLIGHT_COLOR)
#
#     return frame


class ParameterSelectionFrame(tk.Frame):

    def __init__(self,
                 master: tk.Widget,
                 on_change_callback: Optional[Callable[[Tuple[Union[int, str]], tk.Variable], None]] = None,
                 **kwargs):
        super().__init__(master, **kwargs)

        self._param_frame: Optional[IParameterSelectionFrame] = None
        # self._on_change_callback = on_change_callback

    def reset_frame(self):
        if self._param_frame is not None:
            self._param_frame.pack_forget()
            self._param_frame.destroy()
        self._param_frame = None

    def set_parameters(self, builder: ParameterUIBuilder):

        if self._param_frame is not None:
            self._param_frame.destroy()
        self._param_frame = builder.build_parameter_frame(self)
        self._param_frame.pack(fill=tk.BOTH, expand=True)
        if builder.on_change_callback is not None:
            # TODO: Handle cases where number of variables changes
            # Add a trace to all variables
            for path, var in self._param_frame.get_variables().items():
                var.trace_add("write", lambda *args, path=path, var=var: builder.on_change_callback(path, var))

        # Now, set the focus to the first editable field
        first_editable_field: Optional[tk.Widget] = first((child for child in self._param_frame.winfo_children() if child.winfo_class() == "Entry"), default=None)
        if first_editable_field is not None:
            first_editable_field.focus_set()

    def get_filled_parameters(self) -> Optional[ParametersType]:
        return self._param_frame.get_filled_parameters() if self._param_frame is not None else None

    def get_variables(self) -> Mapping[Tuple[Union[int, str]], tk.Variable]:
        return self._param_frame.get_variables() if self._param_frame is not None else {}


# class ParameterSelectionFrame(tk.Frame):
#
#     def __init__(self, parent,
#                  title: str = "Select Parameters",
#                  prompt: str = "",
#                  include_buttons: bool = False,
#                  on_change_callback: Optional[Callable[[ParametersType], None]] = None,
#                  **kwargs):
#         super().__init__(parent, **kwargs)
#         self.params_type = None
#         self.var: Dict[str, tk.Variable] = {}  #
#         self.initial_params = None
#         self.factory_reset_params = None
#         self._editable_fields: Optional[Sequence[str]] = None
#         self._on_change_callback = on_change_callback
#
#         self._param_frame = tk.Frame(self)
#         self._param_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
#         # Configure grid to take up all available space
#         self._param_frame.columnconfigure(0, weight=0)
#         self._param_frame.columnconfigure(1, weight=3)
#         self._param_frame.rowconfigure(1, weight=1)
#         tk.Label(self._param_frame, text=prompt).grid(column=0, row=0, columnspan=3)
#
#         if include_buttons:
#             button_panel = ButtonPanel(self)
#             button_panel.pack(side=tk.BOTTOM, fill=tk.X)
#             button_panel.add_button("Cancel", self._on_cancel, shortcut="<Escape>")
#             button_panel.add_button("OK", self._on_ok, shortcut="<Return>")
#             button_panel.add_button("Reset", self._on_reset, shortcut="<Control-r>")
#
#
#
#
#
#
#         # if include_exit_button:
#         #     bottom_frame = tk.Frame(self)
#         #     bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
#         #
#         #     tk.Button(bottom_frame, text="Cancel", command=self.master.destroy).pack(side=tk.RIGHT)
#         #     tk.Button(bottom_frame, text="OK", command=self.master.destroy).pack(side=tk.RIGHT)
#         #
#     def _on_cancel(self):
#         self.set_parameters(params_type=self.params_type, initial_params=self.initial_params, factory_reset_params=self.factory_reset_params)
#         self.master.destroy()
#
#     def _on_ok(self):
#         self.master.destroy()
#
#     def _on_reset(self):
#         self.set_parameters(params_type=self.params_type, initial_params=self.factory_reset_params, factory_reset_params=self.factory_reset_params)
#
#     def set_parameters(self,
#                        params_type: Optional[type] = None,
#                        initial_params: Optional[ParametersType] = None,
#                        factory_reset_params: Optional[ParametersType] = None,
#                        editable_fields: Optional[Sequence[str]] = None,  # Note means "all", ok?
#                        ):
#
#         if params_type is None:
#             assert initial_params is not None, "If params_type is None, initial_params must be provided."
#             params_type = type(initial_params)
#
#         self.params_type = params_type
#         self.initial_params = initial_params
#         self.factory_reset_params = factory_reset_params
#         self._editable_fields = editable_fields
#
#
#
#
#         self._recreate_fields()
#
#     def set_field_value(self, field_name: str, value: Any, field_collection_index: Optional[int] = None):
#         if field_collection_index is None:
#             self.var[field_name].set(value)
#         else:
#             self.var[field_name].get()[field_collection_index] = value
#
#     def _recreate_fields(self, ):
#
#         # First, clear any existing fields
#         for child in self._param_frame.winfo_children():
#             child.destroy()
#         if self.params_type is None:
#             return
#         for i, f in enumerate(fields(self.params_type), start=1):
#             # if editable_field_names is None or f.name in editable_field_names:
#             self._create_field_ui(f, i, editable=f.name in self._editable_fields if self._editable_fields is not None else True)
#             # else:
#
#     def _create_field_ui(self, f, row: int, editable: bool = True):
#         label = tk.Label(self._param_frame, text=f.metadata['name'] if 'name' in f.metadata else f.name.replace("_", " ").capitalize() + ":")
#         create_tooltip(label, f.metadata.get("help", "No help available."), background=ThemeColours.HIGHLIGHT_COLOR)
#         label.grid(column=0, row=row)
#
#         initial_value = getattr(self.initial_params, f.name) if self.initial_params else f.default
#         factory_value = getattr(self.factory_reset_params, f.name) if self.factory_reset_params else None
#         value_to_set = factory_value if factory_value is not None else initial_value
#
#         if isinstance(get_origin(f.type), type) and issubclass(get_origin(f.type), Sequence):
#
#             # Just add a bunch of string fields in a single frame
#             frame = tk.Frame(self._param_frame)
#             frame.grid(column=1, row=row, sticky="ew")
#             self.var[f.name] = MockVariable(value=list(value_to_set))
#             self._bind_post_write_callback(self.var[f.name])
#             element_frame = tk.Frame(self._param_frame)
#             element_frame.grid(column=1, row=row, sticky="ew")
#             for i, item in enumerate(value_to_set):
#                 # var = tk.StringVar(value=item)
#                 # entry = tk.Entry(frame, textvariable=var, state='readonly')
#                 # entry.grid(column=1, row=i, sticky="ew")
#                 # self.var[f.name] = var
#                 # just make a label
#                 if editable:
#                     label = RespectableLabel(element_frame, text=str(item), anchor=tk.W)
#                     label.set_command(lambda n=f.name, i=i, label=label: self._edit_subfield(label, n, i))
#                 else:
#                     label = RespectableLabel(element_frame, text=str(item), anchor=tk.W)
#
#                 label.grid(column=0, row=i, sticky="ew")
#                 if editable:  # Add a garbage-bin button to the right to delete the annotation
#                     delete_button = RespectableLabel(element_frame, text="🗑", command=lambda n=f.name, i=i, label=label, : self._delete_subfield(label, n, i))
#                     delete_button.grid(column=1, row=i, sticky="ew")
#
#
#         else:
#
#             if editable:
#                 if f.type == str:
#                     self._create_string_field(f, row, value_to_set)
#                 elif f.type == bool:
#                     self._create_boolean_field(f, row, value_to_set)
#                 elif f.type in [int, float]:
#                     self._create_numeric_field(f, row, value_to_set)
#                 else:
#                     raise NotImplementedError(f"Type {f.type} not supported.")
#             else:
#                 label = RespectableLabel(self._param_frame, text=str(value_to_set), anchor=tk.W)
#                 label.grid(column=1, row=row, sticky="ew")
#
#     def _edit_subfield(self, label: tk.Label, field_name: str, field_collection_index: Optional[int] = None):
#         """ Edit a subfield of a field that is a sequence. """
#         current_field_object = self.var[field_name].get() if field_collection_index is None else self.var[field_name].get()[field_collection_index]
#         new_object = ui_choose_parameters(params_type=type(current_field_object), initial_params=current_field_object)
#         self.set_field_value(field_name, new_object, field_collection_index)
#         label.configure(text=str(new_object))
#
#     def _delete_subfield(self, label: tk.Label, field_name: str, field_collection_index: int):
#         """ Delete a subfield of a field that is a sequence. """
#         del self.var[field_name].get()[field_collection_index]
#         label.master.destroy()
#         self._on_change_callback(self.get_filled_parameters())
#
#     def _bind_post_write_callback(self, var: tk.Variable):
#         if self._on_change_callback is not None:
#             var.trace_add("write", lambda *args, var=var: self._on_change_callback(self.get_filled_parameters()))
#
#     def _create_string_field(self, f, row, initial_value):
#         var = tk.StringVar(value=initial_value)
#         entry = tk.Entry(self._param_frame, textvariable=var)
#         entry.grid(column=1, row=row, sticky="ew")
#         self.var[f.name] = var
#         # Link to change callback
#         self._bind_post_write_callback(var)
#
#     def _create_boolean_field(self, f, row, initial_value):
#         var = tk.BooleanVar(value=initial_value)
#         check_box = tk.Checkbutton(self._param_frame, variable=var)
#         check_box.grid(column=1, row=row, sticky="w")
#         self.var[f.name] = var
#         self._bind_post_write_callback(var)
#
#     def _create_numeric_field(self, f, row, initial_value):
#         if f.type == int:
#             var = tk.IntVar(value=initial_value)
#         else:  # f.type == float
#             var = tk.DoubleVar(value=initial_value)
#         entry = tk.Entry(self._param_frame, textvariable=var)
#         entry.grid(column=1, row=row, sticky="ew")
#         self.var[f.name] = var
#         self._bind_post_write_callback(var)
#
#     def get_filled_parameters(self) -> Optional[ParametersType]:
#         try:
#             settings_dict = {f.name: self.var[f.name].get() if isinstance(self.var[f.name], tk.Variable) else self.var[f.name]
#                              for f in fields(self.params_type) if f.name in self._editable_fields}
#             return replace(self.initial_params, **settings_dict)
#         except Exception as e:
#             messagebox.showerror("Error", f"Error reading settings from UI:\n\n{e}\n\n(see Log)")
#             return None


def ui_choose_parameters(
        builder: ParameterUIBuilder,
        # params_type: Optional[type] = None,
        # initial_params: Optional[ParametersType] = None,
        # factory_reset_params: Optional[ParametersType] = None,
        timeout: Optional[float] = None,
        title: str = "Select Parameters",
        # editable_fields: Union[bool, Sequence[str]] = True,
        prompt: str = "Hover mouse over for description of each parameter.  Tab to switch fields, Enter to accept, Escape to cancel."
) -> Optional[ParametersType]:
    # with hold_tkinter_root_context() as root:
    #
    # params_type: Optional[type] = None,
    # initial_params: Optional[ParametersType] = None,
    # editable_fields: Union[bool, Sequence[str]] = True,
    # extra_widget_builders: Mapping[str, Callable[[tk.Widget], None]] = {},
    # ):

    # builder = ParameterUIBuilder(
    #
    #
    #     param_type = params_type,
    #     initial_value = initial_params,
    #     editable_fields = editable_fields,
    #     on_change_callback = self._on_change_callback,
    #     # custom_constructors=extra_widget_builders,
    #     )

    window = tk.Toplevel()
    window.title(title)
    ps_frame = builder.build_parameter_frame(window)
    # ps_frame = build_parameter_frame(window, params_type, initial_params, editable_fields=editable_fields)
    ps_frame.pack()
    bottom_panel = ButtonPanel(window)

    final_params = None

    def on_cancel():
        nonlocal final_params
        final_params = None
        window.destroy()

    def on_ok():
        nonlocal final_params
        final_params = ps_frame.get_filled_parameters()
        window.destroy()

    def on_reset():
        nonlocal ps_frame
        ps_frame.pack_forget()
        # ps_frame = build_parameter_frame(window, params_type, factory_reset_params, editable_fields=editable_fields)
        ps_frame = builder.build_parameter_frame()
        ps_frame.pack()

    bottom_panel.pack(side=tk.BOTTOM, fill=tk.X)
    bottom_panel.add_button("Cancel", on_cancel, shortcut="<Escape>")
    bottom_panel.add_button("OK", on_ok, shortcut="<Return>")
    bottom_panel.add_button("Reset", on_reset, shortcut="<Control-r>")

    if timeout is not None:
        window.after(int(timeout * 1000), on_ok)
    # root.mainloop()
    window.wait_window()
    return final_params


if __name__ == "__main__":
    @dataclass
    class MyParams:
        some_float: float = 4
        some_int: int = field(default=3, metadata=dict(help="Select some integer"))
        some_file: str = field(default=os.path.expanduser("~/some_image.jpg"), metadata=dict(type='file'))


    result = ui_choose_parameters(ParameterUIBuilder(None, MyParams, MyParams()))
    print(result)

    # result = ui_choose_field('N neighbours', int, default=40)
    # print(result)
