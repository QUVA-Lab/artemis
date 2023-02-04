import os


def get_filename_without_extension(path):
    return os.path.splitext(os.path.basename(path))[0]
