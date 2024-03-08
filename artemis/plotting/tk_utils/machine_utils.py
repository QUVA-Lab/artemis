import os

def is_windows_machine() -> bool:
    return os.name == 'nt'
