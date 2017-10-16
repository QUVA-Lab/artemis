import os


def crawl_directory(directory):
    """

    :param directory: A dict of
    :param full_paths:
    :return:
    """
    this_dir = {}
    contents = os.listdir(directory)
    for c in contents:
        full_path = os.path.join(directory, c)
        if os.path.isdir(full_path):
            this_dir[c] = crawl_directory(full_path)
        else:
            this_dir[c] = full_path
    return this_dir
