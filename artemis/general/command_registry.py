from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Dict, Mapping


@dataclass
class NamedCommand:
    name: str
    command: Callable[[], None]
    description: str = ""
    unique_command_id: Optional[str] = None  # Optionally, a unique ID for the command
    keywords: Sequence[str] = ()  # Keywords that can be used to search for this command


_COMMAND_REGISTRY: Optional[Dict[str, NamedCommand]] = None


@contextmanager
def hold_command_registry(registry_dict: Optional[Dict[str, NamedCommand]] = None):
    global _COMMAND_REGISTRY
    if registry_dict is None:
        registry_dict = {}
    old_reg = _COMMAND_REGISTRY
    try:
        _COMMAND_REGISTRY = registry_dict
        yield _COMMAND_REGISTRY
    finally:
        _COMMAND_REGISTRY = old_reg


def add_command_to_registry(
        command: NamedCommand,
        identifier: Optional[str] = None,
        skip_silently_if_no_registry: bool = True,
) -> bool:
    identifier = identifier or command.unique_command_id
    if _COMMAND_REGISTRY is None:
        if skip_silently_if_no_registry:
            return False
        else:
            raise Exception(f"Tried to register command {identifier} but no registry was open")
    if identifier is None:
        return False
    _COMMAND_REGISTRY[identifier] = command
    return True


def get_registry_or_none() -> Mapping[str, NamedCommand]:
    return _COMMAND_REGISTRY
