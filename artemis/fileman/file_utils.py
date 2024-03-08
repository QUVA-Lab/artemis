import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence, Mapping, Iterator

from more_itertools import first

from artemis.general.utils_utils import byte_size_to_string


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
    shutil.copy2(src_path, dest_path)


def copy_file_with_mtime(src, dest, overwrite: bool = True, create_dir_if_needed: bool = True):
    """
    Copies a file from src to dest, preserving the file's modification time,
    and expands the "~" to the user's home directory. It can optionally overwrite
    the destination file (otherwise it will raise a FileExistsError if the
    destination file already exists).
    """
    # Expand the "~" to the user's home directory for both source and destination
    src = os.path.expanduser(src)
    dest = os.path.expanduser(dest)

    # Check if destination file exists
    if os.path.exists(dest) and not overwrite:
        raise FileExistsError(f"The file {dest} already exists and overwrite is set to False.")

    # Ensure the destination directory exists
    dest_dir = os.path.dirname(dest)
    os.makedirs(dest_dir, exist_ok=True)

    if create_dir_if_needed:
        parent, _ = os.path.split(dest)
        if not os.path.exists(parent):
            os.makedirs(parent)

    # Copy the file content
    shutil.copyfile(src, dest)

    # Copy the file metadata
    shutil.copystat(src, dest)





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


def get_files_to_sync(src_folder: str, dest_folder: str, allowed_extensions: Optional[Sequence[str]] = None, skip_existing: bool = True) -> Mapping[str, str]:
    src_files = iter_filepaths_in_directory_recursive(src_folder, allowed_extensions, relative=True)
    files_to_sync = {os.path.join(src_folder, src): os.path.join(dest_folder, src) for src in src_files}
    if skip_existing:
        files_to_sync = {src: dest for src, dest in files_to_sync.items() if not os.path.exists(dest)}
    return files_to_sync


@dataclass
class SyncJobStatus:
    files_completed: int
    total_files: int
    bytes_completed: int
    total_bytes: int
    time_elapsed: float
    time_remaining: float
    next_file: str

    def get_sync_progress_string(self) -> str:
        return f"Synced {self.files_completed}/{self.total_files} files ({byte_size_to_string(self.bytes_completed)} / {byte_size_to_string(self.total_bytes)}).  \nAbout: {self.time_remaining:.1f}s remaining.  Next: {os.path.basename(self.next_file)}"


def iter_sync_files(src_path_to_new_path: Mapping[str, str], overwrite: bool = False, check_byte_sizes=True, verbose: bool = True
                    ) -> Iterator[SyncJobStatus]:
    tstart = time.monotonic()
    per_file_bytes = {src: os.path.getsize(src) for src in src_path_to_new_path}
    total_bytes = sum(per_file_bytes.values())
    total_n_files = len(src_path_to_new_path)
    bytes_completed = 0

    yield SyncJobStatus(files_completed=0, total_files=total_n_files, bytes_completed=0, total_bytes=sum(os.path.getsize(src) for src in src_path_to_new_path), time_elapsed=0, time_remaining=0, next_file=first(src_path_to_new_path.keys(), default=''))
    pairs = list(src_path_to_new_path.items())
    for i, (src_path, dest_path) in enumerate(pairs):
        if overwrite or not os.path.exists(dest_path) or (check_byte_sizes and os.path.getsize(src_path) != os.path.getsize(dest_path)):
            if verbose:
                print(f'Copying {src_path} -> {dest_path}')
            copy_creating_dir_if_needed(src_path, dest_path)
        else:
            if verbose:
                print(f'Skipping {src_path} -> {dest_path}')
        bytes_completed += per_file_bytes[src_path]
        yield SyncJobStatus(
            files_completed=i+1,
            total_files=total_n_files,
            bytes_completed=bytes_completed,
            total_bytes=total_bytes,
            time_elapsed=time.monotonic()-tstart,
            time_remaining=(time.monotonic()-tstart)/(i+1)*(total_n_files-i-1),
            next_file=pairs[i+1][0] if i+1<total_n_files else ''
        )
    if verbose:
        print(f"Done syncing {total_n_files} files ({total_bytes:,} bytes) in {time.monotonic()-tstart:.1f} seconds")


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

