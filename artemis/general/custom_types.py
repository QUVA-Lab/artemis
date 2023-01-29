from typing import Tuple

import numpy as np

from typing import TypeVar, Generic, Tuple, Union, Optional
import numpy as np

Shape = TypeVar("Shape")
DType = TypeVar("DType")


class Array(Generic[Shape, DType], np.ndarray):
    """
    Use this to type-annotate numpy arrays, e.g.

        def transform_image(image: Array['H,W,3', np.uint8], ...):
            ...

    """
    pass


GeneralImageArray = np.ndarray  # Can be (H, W, C) or (H, W), uint8 or float or int or whatever
BGRImageArray = np.ndarray
RGBImageArray = np.ndarray
IndexImageArray = np.ndarray  # A (H, W) array of integer indices
FlatBGRImageArray = np.ndarray
GreyScaleImageArray = np.ndarray
BGRFloatImageArray = np.ndarray  # A BGR image containing floating point data (expected to be in range 0-255, but possibly outside)
HeatMapArray = np.ndarray  # A (H, W) array of floats indicating a heatmap
IndexVector = np.ndarray  # A vector of indices
FloatVector = np.ndarray  # A vector of floats
BoolVector = np.ndarray  # A vector of floats
LTRBBoxArray = np.ndarray  # A (N, 4) array of (left, rop, right, bottom) integer box coordinates.
AnyImageArray = Union[BGRImageArray, GreyScaleImageArray, BGRFloatImageArray, HeatMapArray]
PointIJArray = np.ndarray
RelPointIJArray = np.ndarray  # (i, j) coordinate, normalized to (0, 1)


BGRImageDeltaArray = np.ndarray  # A (H, W) float array of values in [-255, 255] representing a delta between images
MaskImageArray = np.ndarray  # A (H, W) array of floats indicating a heatmap
LabelImageArray = np.ndarray  # A (H, W) array of integer labels

XYSizeTuple = Tuple[int, int]
BGRColorTuple = Tuple[int, int, int]
XYPointTuple = Tuple[float, float]
IJPixelTuple = Tuple[int, int]
TimeIntervalTuple = Tuple[Optional[float], Optional[float]]
