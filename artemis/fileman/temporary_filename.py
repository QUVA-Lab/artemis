import os
from contextlib import contextmanager


@contextmanager
def use_temporary_filename(file_path):
    """
    Delete any existing file with this name, and delete this file after the block completes.
    :param file_path: A full file path
    :yield: A the file path again.
    """
    directory, _ = os.path.split(file_path)

    try:
        os.makedirs(directory)
    except OSError:
        pass

    if os.path.exists(file_path):
        os.remove(file_path)

    yield file_path

    os.remove(file_path)
