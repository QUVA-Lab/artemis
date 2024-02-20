import os
import shutil
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Sequence, Mapping, Iterator


def get_filename_without_extension(path):
    return os.path.splitext(os.path.basename(path))[0]


def modified_timestamp_to_filename(timestamp: float, time_format: str = "%Y-%m-%d_%H-%M-%S") -> str:
    return datetime.utcfromtimestamp(timestamp).strftime(time_format)


def get_dest_filepath(src_path: str, src_root_dir: str, dest_root_dir: str, time_format: str = "%Y-%m-%d_%H-%M-%S", is_daylight = time.daylight) -> str:
    src_root_dir = src_root_dir.rstrip(os.sep) + os.sep
    assert src_path.startswith(src_root_dir), f"File {src_path} was not in root dir {src_root_dir}"
    src_rel_folder, src_filename = os.path.split(src_path[len(src_root_dir):])
    src_name, ext = os.path.splitext(src_filename)
    src_order_number = src_name.split('_', 1)[1]  # 'DJI_0215' -> '0215'
    timestamp = os.path.getmtime(src_path)
    if is_daylight:
        timestamp += 3600.
    timestamp -= 3600. if time.daylight else 0
    new_filename = f'{modified_timestamp_to_filename(timestamp, time_format=time_format)}_{src_order_number}{ext.lower()}'
    return os.path.join(dest_root_dir, src_rel_folder, new_filename)


def iter_filepaths_in_directory_recursive(directory, allowed_extensions: Optional[Sequence[str]], relative = False) -> Iterator[str]:
    """ Yields file paths in a directory
    :param directory: The directory to search
    :param allowed_extensions: If not None, only files with these extensions will be returned
    :param relative: If True, the paths will be relative to the directory
    :return: A generator of file paths
    """
    allowed_extensions = tuple(e.lower() for e in allowed_extensions)
    for dp, dn, filenames in os.walk(directory):
        for f in filenames:
            if allowed_extensions is None or any(f.lower().endswith(e) for e in allowed_extensions):
                abs_path = os.path.join(dp, f)
                if relative:
                    yield os.path.relpath(abs_path, directory)
                else:
                    yield abs_path

    # yield from (f if relative else os.path.join(dp, f) for dp, dn, filenames in os.walk(directory)
    #             for f in filenames if allowed_extensions is None or any(f.lower().endswith(e) for e in allowed_extensions))


def copy_creating_dir_if_needed(src_path: str, dest_path: str):
    parent, _ = os.path.split(dest_path)
    if not os.path.exists(parent):
        os.makedirs(parent)
    shutil.copyfile(src_path, dest_path)


@contextmanager
def open_and_create_parent(path, mode='r'):
    parent, _ = os.path.split(path)
    os.makedirs(parent, exist_ok=True)
    with open(path, mode) as f:
        yield f


def get_recursive_directory_contents_string(directory: str, indent_level=0, indent='  ', max_entries: Optional[int] = None) -> str:
    lines = []
    this_indent = indent * indent_level
    for i, f in enumerate(os.listdir(directory)):
        if max_entries is not None and i >= max_entries:
            lines.append(this_indent + '...')
            break
        lines.append(this_indent + f)
        fpath = os.path.join(directory, f)
        if os.path.isdir(fpath):
            lines.append(get_recursive_directory_contents_string(fpath, indent_level=indent_level + 1, max_entries=max_entries))
    return '\n'.join(lines)


def sync_src_files_to_dest_files(
        src_path_to_new_path: Mapping[str, str],
        overwrite: bool = False,  # Overwrite existing files on machine
        check_byte_sizes=True,  # Check that, for existing files, file-size matches source.  If not, overwrite.
        verbose: bool = True,  # Prints a lot.
        prompt_user_for_confirmation: bool = True  # Prompt user to confirm sync (default true for historical reasons.  Should really default false)
):
    # Filter to only copy when destination file does not exist. TODO: Maybe check file size match here too
    src_path_to_size = {src_path: os.path.getsize(src_path) for src_path in src_path_to_new_path}
    src_to_dest_to_copy = {src: dest for src, dest in src_path_to_new_path.items() if
                           overwrite or not os.path.exists(dest) or (check_byte_sizes and src_path_to_size[src] != os.path.getsize(dest))}

    # Get file size data and prompt user to confirm copy
    size_to_be_copied = sum(src_path_to_size[src] for src in src_to_dest_to_copy)
    if len(src_path_to_new_path)==0:
        print("No files to sync.  Closing")
        return
    elif len(src_to_dest_to_copy)==0:
        print(f"All {len(src_path_to_new_path)} are already synced.  No copying needed.  Pass overwrite=True to force overwrite")
        return
    if verbose:
        print('Files to be copied: ')
        print('  ' + '\n  '.join(f'{i}: {src} -> {dest} ' for i, (src, dest) in enumerate(src_to_dest_to_copy.items())))

    if prompt_user_for_confirmation:
        response = input(f"{len(src_to_dest_to_copy)}/{len(src_path_to_new_path)} files ({size_to_be_copied:,} bytes) will be copied.\n  Type 'copy' to copy >>")
        go_for_it = response.strip(' ') == 'copy'
    else:
        go_for_it = True

    # Do the actual copying.
    if go_for_it:
        print('Copying...')
        data_copied = 0
        for i, src_path in enumerate(sorted(src_to_dest_to_copy), start=1):
            dest_path = src_to_dest_to_copy[src_path]
            print(
                f'Copied {i}/{len(src_to_dest_to_copy)} files ({data_copied / size_to_be_copied:.1%} of data).  Next: {src_path} -> {dest_path} ({src_path_to_size[src_path]:,} B)')
            src_path = os.path.expanduser(src_path)
            dest_path = os.path.expanduser(dest_path)
            copy_creating_dir_if_needed(src_path, dest_path)
            data_copied += src_path_to_size[src_path]
        print('Done copying')
        # if verbose:
        #     print(f'Destination now contains:')
        #     print(get_recursive_directory_contents_string(destination_folder, max_entries=3, indent_level=1))
    else:
        print("You didn't type 'copy'")

