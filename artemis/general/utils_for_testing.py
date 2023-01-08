import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence, Tuple
import os
import numpy as np

from artemis.general.custom_types import MaskImageArray, Array


def mask_to_imstring(mask: MaskImageArray) -> str:

    return '\n'.join(''.join('X' if m else '•' for m in row) for row in mask)


def stringlist_to_mask(*stringlist: Sequence[str]) -> MaskImageArray:
    return np.array([list(row) for row in stringlist])=='X'


@contextmanager
def hold_tempdir():

    tempdir = tempfile.mkdtemp()
    try:
        yield tempdir
    finally:
        if os.path.exists(tempdir):
            shutil.rmtree(tempdir)


@contextmanager
def hold_tempfile(ext = ''):
    tempfilename = tempfile.mktemp() + ext
    try:
        yield tempfilename
    finally:
        if os.path.exists(tempfilename):
            os.remove(tempfilename)


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


