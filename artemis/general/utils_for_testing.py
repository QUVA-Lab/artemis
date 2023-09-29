import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional, Callable
import os

import cv2
import numpy as np

from artemis.fileman.local_dir import get_artemis_data_path
from artemis.general.custom_types import MaskImageArray, Array
from video_scanner.app_utils.utils import AssetVideos
from video_scanner.detection_utils.basic_utils import is_media_path
from video_scanner.general_utils.threaded_cap import VideoLoopingCap


# from video_scanner.app_utils.utils import AssetVideos


# ASSET_FOLDER_PATH = "~/projects/eagle_eyes_app/app/src/main/assets"
# VIDEO_SCANNER_RESOURCE_MODELS_PATH = "~/projects/eagle_eyes_video_scanner/resources/models"



def mask_to_imstring(mask: MaskImageArray) -> str:

    return '\n'.join(''.join('X' if m else '•' for m in row) for row in mask)


def stringlist_to_mask(*stringlist: Sequence[str]) -> MaskImageArray:
    return np.array([list(row) for row in stringlist])=='X'


def delete_existing(path: str) -> bool:
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    return True


def prepare_path_for_write(path: str, overwright_callback: Callable[[str], bool] = lambda s: True) -> str:
    """
    Prepare a path for writing.  Overwright callback will be called if the file exists already and returns True if it is ok to overwright
    """
    final_path = os.path.expanduser(path)
    if os.path.exists(final_path):
        if not overwright_callback(path):
            raise FileExistsError(f"File {path} already exists")
    parent_dir, _ = os.path.split(final_path)
    os.makedirs(parent_dir, exist_ok=True)
    return final_path


def load_mock_video_cap(size_xy: Optional[Tuple[int, int]] = None) -> cv2.VideoCapture:
    return VideoLoopingCap(AssetVideos.FOREST_BLUES.path, size_xy=size_xy)


@contextmanager
def hold_tempdir(path_if_successful: Optional[str] = None):

    tempdir = tempfile.mkdtemp()
    try:
        yield tempdir
        if path_if_successful and os.path.exists(tempdir):
            final_path = prepare_path_for_write(path_if_successful, overwright_callback=delete_existing)
            shutil.move(tempdir, final_path)
            print(f"Wrote temp dir to {final_path}")
    finally:
        if os.path.exists(tempdir):
            try:
                shutil.rmtree(tempdir)
            except PermissionError:
                print(f"Could not delete temp dir {tempdir} - its ok, this can happen on windows")

@contextmanager
def hold_tempfile(ext = '', path_if_successful: Optional[str] = None):
    tempfilename = tempfile.mktemp() + ext
    try:
        yield tempfilename
        if path_if_successful:
            if os.path.exists(tempfilename):
                final_path = prepare_path_for_write(path_if_successful)
                shutil.move(tempfilename, final_path)
                print(f"Wrote temp file to {final_path}")
            else:
                print(f"Temp file did not exist, so could not save it to {tempfilename}")
    finally:
        if os.path.exists(tempfilename):
            os.remove(tempfilename)


def get_all_media_files_in_folder(folder: str) -> Sequence[str]:
    return [os.path.join(folder, p) for p in os.listdir(folder) if is_media_path(p)]


@dataclass
class HeatmapBuilder:

    heatmap: Array['H,W', float]

    @classmethod
    def from_wh(cls, width, height):
        return HeatmapBuilder(np.zeros((height, width)))

    @property
    def width(self) -> int:
        return self.heatmap.shape[1]

    @property
    def height(self) -> int:
        return self.heatmap.shape[0]

    def draw_gaussian(self, mean_xy: Tuple[float, float], std_xy: Tuple[float, float], corr: float = 0., scale: float = 1.):
        assert -1 < corr < 1
        mean_xy = np.asarray(mean_xy)
        vxx, vyy = np.asarray(std_xy)**2
        vxy = corr * np.sqrt(vxx) * np.sqrt(vyy)
        covmat = np.array([[vxx, vxy], [vxy, vyy]])
        xs, ys = np.meshgrid(np.arange(self.width), np.arange(self.height))
        grid_xy = np.concatenate([xs[:, :, None], ys[:, :, None]], axis=2).reshape(-1, 2)
        deltamean = grid_xy - mean_xy
        heat = scale * np.exp(-np.einsum('ni,ij,nj->n', deltamean, 0.5*np.linalg.inv(covmat), deltamean)).reshape(self.heatmap.shape)
        self.heatmap += heat
        return self


def get_or_download_sample_video() -> str:

    url_path = 'https://github.com/petered/data/raw/master/images/dji_2022-11-16_16-47-48_0613.mp4'
    local_path = get_artemis_data_path('test_data/sample_video.mp4', make_local_dir=True)
    if not os.path.exists(local_path):
        print(f"Downloading sample video to {local_path}")
        import requests
        r = requests.get(url_path, allow_redirects=True)
        with open(local_path, 'wb') as f:
            f.write(r.content)
    return local_path
