import pickle
import time
import os
from artemis.fileman.experiment_record import get_current_experiment_id
from artemis.fileman.local_dir import get_local_path


def smart_save(obj, relative_path, remove_file_after = False):
    """
    Save an object locally.  How you save it depends on its extension.
    Extensions currently supported:
        pkl: Pickle file.
        That is all.
    :param obj: Object to save
    :param relative_path: Path to save it, relative to "Data" directory.  The following placeholders can be used:
        %T - ISO time
        %R - Current Experiment Record Identifier (includes experiment time and experiment name)
    :param remove_file_after: If you're just running a test, it's good to verify that you can save, but you don't
        actually want to leave a file behind.  If that's the case, set this argument to True.
    """
    if '%T' in relative_path:
        iso_time = time.now().isoformat().replace(':', '.').replace('-', '.')
        relative_path = relative_path.replace('%T', iso_time)
    if '%R' in relative_path:
        relative_path = relative_path.replace('%R', get_current_experiment_id())
    _, ext = os.path.splitext(relative_path)
    local_path = get_local_path(relative_path, make_local_dir=True)

    print 'Saved object <%s at %s> to file: "%s"' % (obj.__class__.__name__, hex(id(object)), local_path)
    if ext=='.pkl':
        with open(local_path, 'w') as f:
            pickle.dump(obj, f)
    else:
        raise Exception("No method exists yet to save '.%s' files.  Add it!" % (ext, ))

    if remove_file_after:
        os.remove(local_path)

    return local_path


def smart_load(relative_path):
    """
    Load a file, with the method based on the extension.  See smart_save doc for the list of extensions.
    :param relative_path: Path, relative to your data directory.  The extension determines the type of object you will load.
    :return: An object, whose type depends on the extension.  Generally a numpy array for data or an object for pickles.
    """
    # TODO... Support for local files, urls, etc...
    _, ext = os.path.splitext(relative_path)
    local_path = get_local_path(relative_path)
    if ext=='.pkl':
        with open(local_path) as f:
            obj = pickle.load(f)
    else:
        raise Exception("No method exists yet to load '.%s' files.  Add it!" % (ext, ))
    return obj


