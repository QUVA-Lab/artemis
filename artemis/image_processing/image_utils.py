import dataclasses
import itertools
import os
from abc import abstractmethod, ABCMeta
from dataclasses import dataclass, replace
from datetime import datetime
from math import floor, ceil
from typing import Iterable, Tuple, Union, Optional, Sequence, Callable, TypeVar, Iterator

import cv2
import exif
import numpy as np
from attr import attrs, attrib

from artemis.general.custom_types import XYSizeTuple, BGRColorTuple, HeatMapArray, BGRImageDeltaArray, MaskImageArray, LabelImageArray, BGRFloatImageArray, GreyScaleImageArray, \
    BGRImageArray, TimeIntervalTuple, Array, GeneralImageArray
from artemis.general.geometry import reframe_from_a_to_b, reframe_from_b_to_a


class BGRColors:
    BLACK = 0, 0, 0
    WHITE = 255, 255, 255
    BLUE = 255, 0, 0
    GREEN = 0, 255, 0
    YELLOW = 0, 255, 255
    MAGENTA = 255, 0, 255
    CYAN = 255, 255, 0
    SLATE_BLUE = 255, 100, 100
    EYE_BLUE_DARK = 113, 80, 36
    EYE_BLUE = 175, 142, 68
    EYE_BLUE_GRAY = 144, 73, 42
    GRAY = 127, 127, 127
    RED = 0, 0, 255
    DARK_RED = 0, 0, 128
    DARK_GREEN = 0, 100, 0
    DARK_GRAY = 50, 50, 50
    LIGHT_GRAY = 200, 200, 200
    ORANGE = 0, 150, 255
    SKY_BLUE = 255, 180, 100
    VERY_DARK_BLUE = 20, 0, 0


DEFAULT_GAP_COLOR = BGRColors.VERY_DARK_BLUE


def iter_colour_fade(start_color: BGRColorTuple, end_color: BGRColorTuple, steps: Union[int, Sequence[float]]) -> Iterator[BGRColorTuple]:
    b1, g1, r1 = start_color
    b2, g2, r2 = end_color
    if isinstance(steps, int):
        steps = np.linspace(0, 1, steps)
    return ((int(b1 * (1 - f) + b2 * f), int(g1 * (1 - f) + g2 * f), int(r1 * (1 - f) + r2 * f)) for f in steps)


def normalize_to_bgr_image(image_data: Union[MaskImageArray, HeatMapArray, BGRImageArray]) -> BGRImageArray:
    if isinstance(image_data, str):
        image = TextDisplayer().render(image_data)
    elif image_data.dtype == np.uint8:
        if image_data.ndim == 2:
            image_data = np.repeat(image_data[:, :, None], repeats=3, axis=2)
        image = image_data
    elif image_data.dtype == bool:
        image = mask_to_color_image(image_data)
    elif image_data.dtype in (int, float, np.float32):
        image = heatmap_to_color_image(image_data)
    else:
        raise Exception(f"Can't handle image of dtype: {image_data.dtype}")
    return image


ShowFunction = Callable[[BGRImageArray, str], Optional[str]]
_ALTERNATE_SHOW_FUNC: Optional[ShowFunction] = None


# Alternate function for showing an image.  Returns a key-string if any key pressed.


def resize_to_fit(image: BGRImageArray, xy_size: Union[int, Tuple[int, int]], expand: bool = False, interpolation=cv2.INTER_AREA
                  ) -> BGRImageArray:
    if isinstance(xy_size, int):
        xy_size = (xy_size, xy_size)
    sy, sx = image.shape[:2]
    smx, smy = xy_size
    x_rat, y_rat = sx / smx, sy / smy
    if x_rat >= y_rat and (expand or y_rat > 1):
        return cv2.resize(image, dsize=(smx, int(sy / x_rat)), interpolation=interpolation)
    elif y_rat >= x_rat and (expand or x_rat > 1):
        return cv2.resize(image, dsize=(int(sx / y_rat), smy), interpolation=interpolation)
    else:
        return image


def put_image_in_box(image: BGRImageArray, xy_size: Union[int, Tuple[int, int]],
                     gap_colour: BGRColorTuple = DEFAULT_GAP_COLOR, expand=True,
                     interpolation=cv2.INTER_AREA
                     ) -> BGRImageArray:
    resized_image = resize_to_fit(image, xy_size=xy_size, expand=expand, interpolation=interpolation)
    box = create_gap_image(xy_size, gap_colour=gap_colour)
    y_start = (xy_size[1] - resized_image.shape[0]) // 2
    x_start = (xy_size[0] - resized_image.shape[1]) // 2
    box[y_start: y_start + resized_image.shape[0], x_start: x_start + resized_image.shape[1]] = resized_image
    return box


def compose_time_intervals(t1: TimeIntervalTuple, t2: TimeIntervalTuple) -> TimeIntervalTuple:
    t1s, t1e = t1
    t2s, t2e = t2
    t3s = t2s if t1s is None else t1s if t2s is None else t1s + t2s
    t3e = t2e if t1e is None else t1e if t2e is None else t3s + min(t1e - t1s, t2e - t2s)
    return t3s, t3e


def fit_image_to_max_size(image: BGRImageArray, max_size: Tuple[int, int]):
    """ Make sure image fits within (width, height) max_size while preserving aspect ratio """
    if image.shape[0] > max_size[1] or image.shape[1] > max_size[0]:
        scale_factor = min(max_size[1] / image.shape[0], max_size[0] / image.shape[1])
        return cv2.resize(src=image, dsize=None, fx=scale_factor, fy=scale_factor)
    else:
        return image


""" Deprecated.  Use VideoSegment.iter_images """


def iter_images_from_livestream(livestream_url) -> Iterator[BGRImageArray]:
    cap = cv2.VideoCapture(livestream_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame


def iter_images_from_video(path: str, max_size: Optional[Tuple[int, int]] = None,
                           frame_interval: Tuple[Optional[int], Optional[int]] = (None, None),
                           time_interval: TimeIntervalTuple = (None, None),
                           frames_of_interest: Optional[Sequence[int]] = None,
                           use_scan_selection: bool = False,  # Select frames of interest by scanning video
                           keep_ratio: float = 1.,  # Use this to match videos with different framerates
                           rotation: int = 0,
                           verbose: bool = False,
                           ) -> Iterable[BGRImageArray]:

    # On windows machines the normal video reading does not work for images
    if any(path.lower().endswith(e) for e in ('.jpg', '.jpeg')):
        for i, image_path in enumerate(path.split(';')):
            if frames_of_interest is not None and i not in frames_of_interest:
                continue
            image = cv2.imread(os.path.expanduser(image_path))
            if max_size is not None:
                image = fit_image_to_max_size(image, max_size)
            assert image is not None, f"Could not read any image from {image_path}"
            yield image
        return




    assert not use_scan_selection, "This does not work.  See bug: https://github.com/opencv/opencv/issues/9053"
    path = os.path.expanduser(path)
    cap = cv2.VideoCapture(path)
    start_frame, stop_frame = frame_interval
    start_time, end_time = time_interval
    if max_size is not None:  # Set cap size.  Sometimes this does not work so we also have the code below.
        sx, sy = max_size if rotation in (0, 2) else max_size[::-1]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, sx)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, sy)

    if start_time is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000.)

    if start_frame is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_ix = start_frame
    else:
        frame_ix = 0

    fps = cap.get(cv2.CAP_PROP_FPS)

    unique_frame_ix = -1

    iter_frames_of_interest = iter(frames_of_interest) if frames_of_interest is not None else None

    initial_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

    while cap.isOpened():

        if iter_frames_of_interest is not None and use_scan_selection:
            try:
                next_frame = initial_frame + next(iter_frames_of_interest) + (1 if initial_frame == 0 else 0)  # Don't know why it just works
            except StopIteration:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)

        if stop_frame is not None and frame_ix >= stop_frame:
            break
        elif end_time is not None and frame_ix / fps > end_time - (start_time or 0.):
            break

        unique_frame_ix += 1

        isgood, image = cap.read()

        if not isgood:
            print(f'Reach end of video at {path}')
            break
        if max_size is not None:
            image = fit_image_to_max_size(image, max_size)
        if keep_ratio != 1:
            if verbose:
                print(f'Real surplus: {frame_ix - keep_ratio * unique_frame_ix}')
            frame_surplus = round(frame_ix - keep_ratio * unique_frame_ix)
            if frame_surplus < 0:  # Frame debt - yield this one twice
                if verbose:
                    print('Yielding extra frame due to frame debt')
                yield image
                frame_ix += 1
            elif frame_surplus > 0:  # Frame surplus - skip it
                if verbose:
                    print('Skipping frame due to frame surplus')
                continue

        if iter_frames_of_interest is None or use_scan_selection or (not use_scan_selection and frame_ix in frames_of_interest):
            if rotation != 0:
                yield cv2.rotate(image, rotateCode={1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation])
            else:
                yield image
        frame_ix += 1


def iter_passthrough_write_video(image_stream: Iterable[BGRImageArray], path: str, fps: float = 30.) -> Iterable[BGRImageArray]:
    path = os.path.expanduser(path)
    dirs, _ = os.path.split(path)
    try:
        os.makedirs(dirs)
    except OSError:
        pass
    cap = None
    for img in image_stream:
        if cap is None:
            # cap = cv2.VideoWriter(path, fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps=fps, frameSize=(img.shape[1], img.shape[0]))
            cap = cv2.VideoWriter(path, fourcc=cv2.VideoWriter_fourcc('H', '2', '6', '4'), fps=fps, frameSize=(img.shape[1], img.shape[0]))
            # Make it write high-quality video
            # cap = cv2.VideoWriter(path, fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps=fps, frameSize=(img.shape[1], img.shape[0]), isColor=True)
        cap.write(img)
        yield img
    cap.release()
    print(f'Saved video to {path}')


def fade_image(image: BGRImageArray, fade_level: float) -> BGRImageArray:
    return (image.astype(np.float) * fade_level).astype(np.uint8)


def mask_to_color_image(mask: MaskImageArray) -> BGRImageArray:
    image = np.zeros(mask.shape[:2] + (3,), dtype=np.uint8)
    image[mask] = 255
    return image


def compute_heatmap_bounds(heatmap: HeatMapArray, assume_zero_min: bool = False, assume_zero_center: bool = False) -> Tuple[float, float]:
    assert not (assume_zero_min and assume_zero_center), "You can't assube both the min and the center are zero"
    if assume_zero_center:
        extreme = np.max(np.abs(heatmap))
        min_heat, max_heat = -extreme, extreme
    else:
        min_heat = 0. if assume_zero_min else np.min(heatmap)
        max_heat = np.max(heatmap)
    return min_heat, max_heat


def heatmap_to_greyscale_image(heatmap: HeatMapArray, assume_zero_min: bool = False, assume_zero_center: bool = False
                               ) -> GreyScaleImageArray:
    min_heat, max_heat = compute_heatmap_bounds(heatmap, assume_zero_min=assume_zero_min, assume_zero_center=assume_zero_center)
    img = np.zeros(heatmap.shape[:2], dtype=np.uint8)
    if min_heat != max_heat:
        img[:] = ((heatmap - min_heat) * (255 / (max_heat - min_heat))).astype(np.uint8)
    return img


def heatmap_to_color_image(heatmap: HeatMapArray, assume_zero_min: bool = True, assume_zero_center: bool = False, show_range=False,
                           upsample_factor: int = 1, additional_text: Optional[str] = None, text_scale=1.
                           ) -> BGRImageArray:
    min_heat, max_heat = compute_heatmap_bounds(heatmap, assume_zero_min=assume_zero_min, assume_zero_center=assume_zero_center)
    if heatmap.ndim == 2:
        heatmap = heatmap[:, :, None]
    img = np.zeros(heatmap.shape[:2] + (3,), dtype=np.uint8)
    if min_heat != max_heat:
        img[:] = ((heatmap - min_heat) * (255 / (max_heat - min_heat))).astype(np.uint8)
    if upsample_factor != 1:
        img = cv2.resize(img, dsize=None, fx=upsample_factor, fy=upsample_factor, interpolation=cv2.INTER_NEAREST)
    if show_range:
        text = f'Scale: {min_heat:.2g} -> {max_heat:.2g}'
        if additional_text is not None:
            text = ', '.join([text, additional_text])
        range_info = TextDisplayer(max_size=(img.shape[1], 15), match_max_size=True, scale=text_scale).render(text)
        img = np.vstack([img, range_info])
    return img


def float_color_image_to_color_image(float_color_image: BGRFloatImageArray) -> BGRImageArray:
    return np.clip(float_color_image, 0, 255).astype(np.uint8)


def image_and_heatmap_to_color_image(image: BGRImageArray, heatmap: HeatMapArray) -> BGRImageArray:
    max_heat = np.max(heatmap)
    return (image * (heatmap[:, :, None] / max_heat if max_heat != 0 else 0.)).astype(np.uint8)


def delta_image_to_color_image(delta_image: BGRImageDeltaArray, show_range: bool = False) -> BGRImageArray:
    if delta_image.ndim == 2:
        delta_image = np.repeat(delta_image[:, :, None], axis=2, repeats=3)
    max_dev = np.max(np.abs(delta_image))
    img = ((delta_image * (127 / max_dev)) + 127).astype(np.uint8)
    if show_range:
        range_info = TextDisplayer(max_size=(delta_image.shape[1], 15), match_max_size=True).render(f'Scale: {-max_dev:.2g} -> {max_dev:.2g}')
        img = np.vstack([range_info, img])
    return img


class IWorldToPixFunc(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, world_coords: Tuple[float, float]) -> Tuple[int, int]:
        raise NotImplementedError()


class IdentityWorldToPixFunc(IWorldToPixFunc):

    def __call__(self, world_coords: Tuple[float, float]) -> Tuple[int, int]:
        wx, wy = world_coords

        return round(wx), round(wy)


@dataclass
class NormalizedWorldToPixFunc(IWorldToPixFunc):
    img_size: XYSizeTuple
    y_from_bottom: bool = False

    def __call__(self, world_coords: Tuple[float, float]) -> Tuple[int, int]:
        wx, wy = world_coords
        sx, sy = self.img_size
        return round(wx * sx), round(wy * sy)


def get_segmentation_colours(include_null: bool = True) -> Sequence[BGRColorTuple]:
    colours = (BGRColors.BLACK,) if include_null else ()
    colours += (BGRColors.SLATE_BLUE, BGRColors.GREEN, BGRColors.RED, BGRColors.CYAN, BGRColors.YELLOW, BGRColors.MAGENTA, \
                BGRColors.GRAY, BGRColors.ORANGE, BGRColors.BLUE, BGRColors.DARK_GREEN)
    return colours


def label_image_to_bgr(label_image: LabelImageArray, colours: Optional[Iterable[BGRColorTuple]] = get_segmentation_colours()
                       ) -> BGRImageArray:
    color_cycle = itertools.cycle(colours)
    max_label = int(np.max(label_image))
    colors = np.array([c for i, c in zip(range(max_label + 1), color_cycle)], dtype=np.uint8)
    return colors[label_image]


T = TypeVar('T', bound='BaseBox')


@dataclass
class BaseBox:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    label: str = ''
    score: float = 1.

    @classmethod
    def from_ltrb(cls, l, t, r, b, label: str = '', score: float = 1.) -> 'BaseBox':
        return cls(x_min=l, x_max=r, y_min=t, y_max=b, label=label, score=score)

    @classmethod
    def from_ijhw(cls, i, j, h, w, label: str = '', score: float = 1.) -> 'BaseBox':
        return cls(x_min=j - w / 2, x_max=j + w / 2, y_min=i - h / 2, y_max=i + h / 2, label=label, score=score)

    @classmethod
    def from_xywh(cls, x, y, w, h, label: str = '') -> 'BaseBox':
        return cls(x_min=x - w / 2, x_max=x + w / 2, y_min=y - h / 2, y_max=y + h / 2, label=label)

    @classmethod
    def from_center_crop(cls, box_size_xy: Tuple[float, float], parent_size_xy: Tuple[float, float], center_xy: Optional[Tuple[float, float]] = None):
        bw, bh = box_size_xy
        pw, ph = parent_size_xy
        x_min, y_min = (pw - bw) / 2, (ph - bh) / 2
        return cls(x_min=x_min, x_max=x_min + bw, y_min=y_min, y_max=y_min + bh)

    def scale_about_center(self, scale: float) -> 'BaseBox':
        x, y, w, h = self.get_xywh()
        return self.from_xywh(x, y, w * scale, h * scale, label=self.label)

    def get_xy_size(self) -> Tuple[float, float]:
        return self.x_max - self.x_min, self.y_max - self.y_min

    def get_center(self) -> Tuple[float, float]:
        return (self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2

    def get_xywh(self, ) -> Tuple[float, float, float, float]:
        return self.get_center() + self.get_xy_size()

    def get_area(self) -> float:
        h, w = self.get_xy_size()
        return h * w

    def get_diagonal_length(self) -> float:
        sx, sy = self.get_xy_size()
        return (sx ** 2 + sy ** 2) ** .5

    def get_intersection_box(self, other) -> Optional['BoundingBox']:
        x_min, x_max, y_min, y_max = max(self.x_min, other.x_min), min(self.x_max, other.x_max), max(self.y_min, other.y_min), min(self.y_max, other.y_max)
        if x_max > x_min and y_max > y_min:  # Box is valid
            return BoundingBox(x_min, x_max, y_min, y_max)
        else:
            return None

    def is_valid(self) -> bool:
        return self.x_max > self.x_min and self.y_max > self.y_min

    def get_intersection_area(self, other) -> float:
        ibox = self.get_intersection_box(other)
        return 0. if ibox is None else ibox.get_area()

    def get_union_box(self, other) -> 'BoundingBox':
        x_min, x_max, y_min, y_max = min(self.x_min, other.x_min), max(self.x_max, other.x_max), min(self.y_min, other.y_min), max(self.y_max, other.y_max)
        return BoundingBox(x_min, x_max, y_min, y_max)

    @classmethod
    def total_area(cls, boxes: Sequence['RelativeBoundingBox']):
        return sum(bb.get_area() for bb in boxes) - sum(bb1.get_intersection_area(bb2) for bb1 in boxes for bb2 in boxes if bb1 is not bb2)

    def is_containing_point(self, xy: Tuple[float, float]) -> bool:
        x, y = xy
        return self.x_min <= x < self.x_max and self.y_min <= y < self.y_max


def slice_image_with_pad(image: "Array['H,W,C', dtype]", xxyy_box: Tuple[int, int, int, int], gap_color: "Array['C', dtype]"):
    """
    Slice the image with the xxyy_box (x_start, x_stop, y_start, y_stop)
    And fill in the empty areas with the gap color
    Such that the result will have shape
    """
    l, r, t, b = xxyy_box
    l_pad = max(0, -l)
    r_pad = max(0, r - image.shape[1])
    t_pad = max(0, -t)
    b_pad = max(0, b - image.shape[0])
    if l_pad == t_pad == r_pad == b_pad == 0:
        return image[t:b, l:r].copy()
    else:
        img = np.full((b - t, r - l,) + image.shape[2:], dtype=image.dtype, fill_value=gap_color)
        img[t_pad:img.shape[0] - b_pad, l_pad:img.shape[1] - r_pad] = image[max(0, t):max(0, b), max(0, l):max(0, r)]
        return img


@dataclass
class BoundingBox(BaseBox):

    def get_xwraps(self, x_size: int) -> Tuple['BoundingBox', 'BoundingBox']:
        """ get the left-and-wright wraps of this box (the 'right' will fall out of the image if the box does not wrap) """

        offset = (int(self.x_min) // x_size) * x_size
        right_wrap = dataclasses.replace(self, x_min=self.x_min - offset, x_max=self.x_max - offset)
        left_wrap = dataclasses.replace(self, x_min=self.x_min - offset - x_size, x_max=self.x_max - offset - x_size)
        return left_wrap, right_wrap

    def get_relative_distance(self, other: 'BoundingBox') -> float:
        """ Get relative distance between boxes"""
        w1, h1 = self.get_xy_size()
        w2, h2 = self.get_xy_size()
        wm, hm = (w1 + w2) / 2, (h1 + h2) / 2
        x1, y1 = self.get_center()
        x2, y2 = other.get_center()
        return np.sqrt(((x2 - x1) / wm) ** 2 + ((y2 - y1) / hm) ** 2)

    def is_contained_in_image(self, image_size_xy: Tuple[int, int]):
        sx, sy = image_size_xy
        return self.x_min >= 0 and self.x_max < sx and self.y_max >= 0 and self.y_min < sy

    # @classmethod
    # def from_lbwh(cls, l, b, w, h, label: str = '') -> 'BoundingBox':
    #     return BoundingBox(x_min=l, x_max=l+w, y_min=b, y_max=b+h, label=label)

    def to_ij(self) -> Tuple[int, int]:
        """ Get the (row, col) of the center of the box """
        return round((self.y_min + self.y_max) / 2.), round((self.x_min + self.x_max) / 2.)

    def to_crop_ij(self) -> Tuple[int, int]:
        """ Get the (row, col) of the center of the box in the frame of the cropped image"""
        i, j = self.to_ij()
        return i - int(self.y_min), j - int(self.x_min)

    def compute_iou(self, other: 'BoundingBox') -> float:
        """ Get Intersection-over-Union overlap area between boxes - will be between zero and 1 """
        intesection_box = self.get_intersection_box(other)
        if intesection_box is not None:
            return intesection_box.get_area() / (self.get_area() + other.get_area() - intesection_box.get_area())
        else:
            return 0.

    def get_shifted(self, xy_shift: Tuple[float, float], frame_size_limit: Tuple[Optional[int], Optional[int]] = (None, None)
                    ) -> 'BoundingBox':
        dx, dy = xy_shift
        lx, ly = frame_size_limit
        if lx is not None:
            dx = min(max(dx, -self.x_min), lx - self.x_max)
        if ly is not None:
            dy = min(max(dy, -self.y_min), ly - self.y_max)
        return BoundingBox(x_min=self.x_min + dx, x_max=self.x_max + dx, y_min=self.y_min + dy, y_max=self.y_max + dy, label=self.label)

    def get_lrbt_int(self) -> Tuple[int, int, int, int]:
        return max(0, floor(self.x_min)), max(0, ceil(self.x_max)), max(0, floor(self.y_min)), max(0, ceil(self.y_max))

    def get_xywh_int(self):
        l, r, b, t = self.get_lrbt_int()
        return (l + r) // 2, (b + t) // 2, r - l, t - b

    def get_image_slicer(self) -> Tuple[slice, slice]:
        y_slice = slice(max(0, floor(self.y_min)), max(0, ceil(self.y_max + 1e-9)))
        x_slice = slice(max(0, floor(self.x_min)), max(0, ceil(self.x_max + 1e-9)))
        return y_slice, x_slice

    def slice_image(self, image: BGRImageArray, copy: bool = False, wrap_x=False) -> BGRImageArray:

        y_slice, x_slice = self.get_image_slicer()
        if wrap_x:
            x_slice = np.arange(floor(self.x_min), ceil(self.x_max + 1e-9)) % image.shape[1]
            copy = False  # Bewcase this slicing will already cause a copy

        image_crop = image[y_slice, x_slice]
        if copy:
            image_crop = image_crop.copy()
        # assert image_crop.shape[1] == round(self.x_max - self.x_min) + 1
        return image_crop

    def crop_image(self, image: BGRImageArray, gap_color: BGRColorTuple = DEFAULT_GAP_COLOR):
        return slice_image_with_pad(image, xxyy_box=[int(self.x_min), int(self.x_max), int(self.y_min), int(self.y_max)], gap_color=gap_color)

    def squareify(self) -> 'BoundingBox':
        sx, sy = self.get_center()
        size = max(self.get_xy_size())
        return BoundingBox.from_xywh(sx, sy, size, size, label=self.label)

    def scale_by(self, factor: float) -> 'BoundingBox':
        sx, sy = self.get_center()
        w, h = self.get_xy_size()
        return BoundingBox.from_xywh(sx, sy, w * factor, h * factor, label=self.label)

    def pad(self, pad: float) -> 'BoundingBox':
        sx, sy = self.get_center()
        w, h = self.get_xy_size()
        return BoundingBox.from_xywh(sx, sy, w + pad, h + pad, label=self.label)

    def to_relative(self, img_size_xy: Tuple[int, int], clip_if_needed=False) -> "RelativeBoundingBox":
        return RelativeBoundingBox.from_absolute_bbox(self, img_size_xy=img_size_xy, clip_if_needed=clip_if_needed)


@dataclass
class RelativeBoundingBox(BaseBox):
    """ A bounding box defined relative to the size of the image. """

    # x_min: float
    # x_max: float
    # y_min: float
    # y_max: float
    # label: str = ''
    # score: float = 1.  # 1=positive, 0=neutral, -1=negative.

    def __post_init__(self):
        assert 0 <= (b := self.x_min) <= 1, f"Bad value: {b}"
        assert 0 <= (b := self.x_max) <= 1, f"Bad value: {b}"
        assert 0 <= (b := self.y_min) <= 1, f"Bad value: {b}"
        assert 0 <= (b := self.y_max) <= 1, f"Bad value: {b}"

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float, label: str = '', cut_to_size=True) -> 'RelativeBoundingBox':
        x_min, x_max, y_min, y_max = x - w / 2, x + w / 2, y - h / 2, y + h / 2
        if cut_to_size:
            x_min, x_max, y_min, y_max = [np.clip(l, 0, 1) for l in (x_min, x_max, y_min, y_max)]
        return RelativeBoundingBox(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, label=label)

    @classmethod
    def from_absolute_bbox(cls, bbox: BoundingBox, img_size_xy: Tuple[int, int], clip_if_needed: bool = False) -> 'RelativeBoundingBox':
        w, h = img_size_xy
        if clip_if_needed:
            return RelativeBoundingBox(x_min=max(0., bbox.x_min / w), x_max=min(1., bbox.x_max / w),
                                       y_min=max(0., bbox.y_min / h), y_max=min(1., bbox.y_max / h),
                                       score=bbox.score, label=bbox.label)
        else:
            return RelativeBoundingBox(x_min=bbox.x_min / w, x_max=bbox.x_max / w,
                                       y_min=bbox.y_min / h, y_max=bbox.y_max / h,
                                       score=bbox.score, label=bbox.label)

    def to_xxyy(self):
        return self.x_min, self.x_max, self.y_min, self.y_max

    def to_ij(self):
        return (self.y_min + self.y_max) / 2., (self.x_min + self.x_max) / 2.

    def to_bounding_box(self, image_size_xy: XYSizeTuple) -> BoundingBox:
        w, h = image_size_xy
        return BoundingBox(x_min=self.x_min * w, x_max=self.x_max * w, y_max=self.y_max * h, y_min=self.y_min * h, score=self.score, label=self.label)

    def to_bounding_box_given_image(self, image: GeneralImageArray) -> BoundingBox:
        return self.to_bounding_box(image_size_xy=(image.shape[1], image.shape[0]))

    def slice_image(self, image: BGRImageArray, copy: bool = False, wrap_x=False) -> BGRImageArray:

        y_slice, x_slice = self.to_bounding_box_given_image(image).get_image_slicer()
        if wrap_x:
            x_slice = np.arange(floor(self.x_min), ceil(self.x_max + 1e-9)) % image.shape[1]
            copy = False  # Bewcase this slicing will already cause a copy

        image_crop = image[y_slice, x_slice]
        if copy:
            image_crop = image_crop.copy()
        # assert image_crop.shape[1] == round(self.x_max - self.x_min) + 1
        return image_crop

    def to_ij_abs(self, img_size_xy) -> Tuple[int, int]:
        w, h = img_size_xy
        irel, jrel = self.to_ij()
        iabs, jabs = irel * h, jrel * w
        return round(iabs), round(jabs)


def vstack_images(images: Sequence[BGRImageArray], gap_colour: BGRColorTuple = DEFAULT_GAP_COLOR, border: int = 1
                  ) -> BGRImageArray:
    ysize = sum(im.shape[0] for im in images) + (len(images) + 1) * border
    xsize = max(im.shape[1] for im in images) + 2 * border
    base_img = create_gap_image(size=(xsize, ysize), gap_colour=gap_colour)
    y_start = border
    for im in images:
        x_start = (base_img.shape[1] - im.shape[1]) // 2 + border
        y_end = y_start + im.shape[0]
        base_img[y_start: y_end, x_start: x_start + im.shape[1]] = im
        y_start = y_end + border
    return base_img


def create_gap_image(  # Generate a colour image filled with one colour
        size: Tuple[int, int],  # Image (width, height))
        gap_colour: BGRColorTuple = None  # BGR color to fill gap, or None to use default
) -> BGRImageArray:
    if gap_colour is None:
        gap_colour = DEFAULT_GAP_COLOR

    width, height = size
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img += np.array(gap_colour, dtype=np.uint8)
    return img


def create_placeholder_image(  # Generate a colour image filled with one colour
        size: Tuple[int, int],  # Image (width, height))
        gap_colour: BGRColorTuple = None,  # BGR color to fill gap, or None to use default
        x_color: BGRColorTuple = BGRColors.LIGHT_GRAY,
) -> BGRImageArray:
    gap_image = create_gap_image(gap_colour=gap_colour, size=size)
    n_steps = max(gap_image.shape[:2])
    row_ixs = np.linspace(0, gap_image.shape[0] - 1, n_steps).astype(int)
    col_ixs = np.linspace(0, gap_image.shape[1] - 1, n_steps).astype(int)
    gap_image[row_ixs, col_ixs] = x_color
    gap_image[row_ixs, -col_ixs] = x_color
    return gap_image


def create_random_image(size_xy: Tuple[int, int], in_color: bool = True, seed=None) -> BGRImageArray:
    return np.random.RandomState(seed).randint(0, 256, size=(size_xy[1], size_xy[0]) + ((3,) if in_color else ()), dtype=np.uint8)


@attrs
class TextDisplayer:
    """ Converts text to image """
    text_color = attrib(default=BGRColors.WHITE)
    thickness = attrib(default=1)
    font = attrib(default=cv2.FONT_HERSHEY_PLAIN)
    scale = attrib(default=1)
    background_color = attrib(factory=lambda: DEFAULT_GAP_COLOR)
    size = attrib(type=Optional[Tuple[int, int]], default=None)  # (width, height) in characters
    vspace = attrib(type=float, default=0.4)
    expand_box = attrib(type=bool, default=True)
    max_size = attrib(type=Tuple[Optional[int], Optional[int]], default=(None, None))
    match_max_size = attrib(type=bool, default=False)
    _last_size = attrib(type=Optional[Tuple[int, int]], default=None)

    def render(self, data: str) -> BGRImageArray:

        lines = data.split('\n')
        longest_line = max(lines, key=len)
        (text_width, text_height), baseline = cv2.getTextSize(longest_line, self.font, self.scale, self.thickness)

        width, height = text_width + 10, int(len(lines) * text_height * (1 + self.vspace))
        wmax, hmax = self.max_size
        if self.match_max_size:
            assert wmax is not None and hmax is not None, f"If you match max size, you need to specify.  Got {self.max_size}"
            width, height = wmax, hmax
        else:
            width, height = (min(width, wmax) if wmax is not None else width), (min(height, hmax) if hmax is not None else height)

        if self.expand_box:
            oldwidth, oldheight = self._last_size if self._last_size is not None else (0, 0)
            self._last_size = max(oldwidth, width), max(oldheight, height)
            width, height = self._last_size
        img = create_gap_image((width, height) if self.size is None else self.size, gap_colour=self.background_color)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (0, int(baseline * 2 + i * (1 + self.vspace) * text_height)), fontFace=self.font,
                        fontScale=self.scale, color=self.text_color,
                        thickness=self.thickness, bottomLeftOrigin=False)
        return img


def conditional_running_min(values, condition, axis, default):
    max_value = np.iinfo(values.dtype).max
    result_array = np.full(values.shape, fill_value=max_value)
    agg = np.full(values.shape[axis], fill_value=max_value)
    for i in range(values.shape[axis]):
        index_slice = (slice(None),) * axis + (i,)
        these_values = values[index_slice]
        these_conditions = condition[index_slice]
        agg = np.where(these_conditions, np.minimum(agg, these_values), max_value)
        result_array[index_slice] = agg
    result_array[result_array == max_value] = default
    return result_array


def mask_to_boxes(mask: Array['H,W', bool]) -> Array['N,4', int]:
    """ Convert a boolean (Height x Width) mask into a (N x 4) array of NON-OVERLAPPING bounding boxes
    surrounding "islands of truth" in the mask.  Boxes indicate the (Left, Top, Right, Bottom) bounds
    of each island, with Right and Bottom being NON-INCLUSIVE (ie they point to the indices AFTER the island).

    This algorithm (Downright Boxing) does not necessarily put separate connected components into
    separate boxes.

    You can "cut out" the island-masks with
        boxes = mask_to_boxes(mask)
        island_masks = [mask[t:b, l:r] for l, t, r, b in boxes]
    """
    max_ix = max(s + 1 for s in mask.shape)  # Use this to represent background
    # These arrays will be used to carry the "box start" indices down and to the right.
    x_ixs = np.full(mask.shape, fill_value=max_ix)
    y_ixs = np.full(mask.shape, fill_value=max_ix)

    # Propagate the earliest x-index in each segment to the bottom-right corner of the segment
    for i in range(mask.shape[0]):
        x_fill_ix = max_ix
        for j in range(mask.shape[1]):
            above_cell_ix = x_ixs[i - 1, j] if i > 0 else max_ix
            still_active = mask[i, j] or ((x_fill_ix != max_ix) and (above_cell_ix != max_ix))
            x_fill_ix = min(x_fill_ix, j, above_cell_ix) if still_active else max_ix
            x_ixs[i, j] = x_fill_ix

    # Propagate the earliest y-index in each segment to the bottom-right corner of the segment
    for j in range(mask.shape[1]):
        y_fill_ix = max_ix
        for i in range(mask.shape[0]):
            left_cell_ix = y_ixs[i, j - 1] if j > 0 else max_ix
            still_active = mask[i, j] or ((y_fill_ix != max_ix) and (left_cell_ix != max_ix))
            y_fill_ix = min(y_fill_ix, i, left_cell_ix) if still_active else max_ix
            y_ixs[i, j] = y_fill_ix

    # Find the bottom-right corners of each segment
    new_xstops = np.diff((x_ixs != max_ix).astype(np.int32), axis=1, append=False) == -1
    new_ystops = np.diff((y_ixs != max_ix).astype(np.int32), axis=0, append=False) == -1
    corner_mask = new_xstops & new_ystops
    y_stops, x_stops = np.array(np.nonzero(corner_mask))

    # Extract the boxes, getting the top-right corners from the index arrays
    x_starts = x_ixs[y_stops, x_stops]
    y_starts = y_ixs[y_stops, x_stops]
    ltrb_boxes = np.hstack([x_starts[:, None], y_starts[:, None], x_stops[:, None] + 1, y_stops[:, None] + 1])
    return ltrb_boxes


def display_to_pixel_dim(display_coord: float, pixel_center_coord: float, window_dim: int, zoom: float, pixel_limit: Optional[int] = None) -> float:
    pixel_coord = pixel_center_coord + (display_coord - (window_dim / 2)) / zoom
    if pixel_limit is not None:
        pixel_coord = np.maximum(0, np.minimum(pixel_limit, pixel_coord))
    return pixel_coord


def get_min_zoom(img_wh: Tuple[int, int], window_wh: Tuple[int, int]) -> float:
    return min(window_wh[i] / img_wh[i] for i in (0, 1))


def clip_to_slack_bounds(x: float, bound: Tuple[float, float]) -> float:
    x_lower, x_upper = bound
    if x_lower <= x_upper:
        return np.clip(x, x_lower, x_upper)
    else:
        return (x_lower + x_upper) / 2


@dataclass
class ImageViewInfo:
    # image: BGRImageArray
    zoom_level: float  # Zoom level
    center_pixel_xy: Tuple[int, int]  # (x, y) coordinates of center-pixel
    window_disply_wh: Tuple[int, int]  # (width, height) of display window
    image_wh: Tuple[int, int]
    scroll_bar_width: int = 10

    @classmethod
    def from_initial_view(cls, window_disply_wh: Tuple[int, int], image_wh: Tuple[int, int], scroll_bar_width: int = 10) -> 'ImageViewInfo':
        return ImageViewInfo(
            zoom_level=get_min_zoom(img_wh=image_wh, window_wh=np.asarray(window_disply_wh) - scroll_bar_width),
            center_pixel_xy=tuple(s // 2 for s in image_wh),
            window_disply_wh=window_disply_wh,
            image_wh=image_wh
        )

    def adjust_frame_and_image_size(self, new_frame_wh: Tuple[int, int], new_image_wh: Tuple[int, int]) -> 'ImageViewInfo':
        return replace(self, window_disply_wh=new_frame_wh, image_wh=new_image_wh)

    def _get_display_wh(self) -> Tuple[int, int]:
        return self.window_disply_wh[0] - self.scroll_bar_width, self.window_disply_wh[1] - self.scroll_bar_width

    def _get_display_midpoint_xy(self) -> Tuple[float, float]:
        w, h = self._get_display_wh()
        return w / 2, h / 2

    def _get_min_zoom(self) -> float:
        return get_min_zoom(img_wh=self.image_wh, window_wh=self._get_display_wh())

    def zoom_out(self) -> 'ImageViewInfo':
        new_zoom = self._get_min_zoom()
        return replace(self, zoom_level=new_zoom).adjust_pan_to_boundary()

    def zoom_by(self, relative_zoom: float, invariant_display_xy: Optional[Tuple[float, float]] = None, limit: bool = True) -> 'ImageViewInfo':

        new_zoom = max(self._get_min_zoom(), self.zoom_level * relative_zoom) if limit else self.zoom_level * relative_zoom
        if invariant_display_xy is None:
            invariant_display_xy = self._get_display_midpoint_xy()
        else:
            invariant_display_xy = np.maximum(0, np.minimum(self._get_display_wh(), invariant_display_xy))
        invariant_pixel_xy = self.display_xy_to_pixel_xy(display_xy=invariant_display_xy)

        coeff = (1 - 1 / relative_zoom)
        new_center_pixel_xy = tuple(np.array(self.center_pixel_xy) * (1 - coeff) + np.array(invariant_pixel_xy) * coeff)
        result = replace(self, zoom_level=new_zoom, center_pixel_xy=new_center_pixel_xy)
        if limit:
            result = result.adjust_pan_to_boundary()
        return result

    def zoom_to_pixel(self, pixel_xy: Tuple[int, int], zoom_level: float) -> 'ImageViewInfo':
        return replace(self, center_pixel_xy=pixel_xy, zoom_level=zoom_level)

    def adjust_pan_to_boundary(self) -> 'ImageViewInfo':
        display_edge_xy = np.asarray(self._get_display_midpoint_xy())
        pixel_edge_xy = display_edge_xy / self.zoom_level
        adjusted_pixel_center_xy = tuple(clip_to_slack_bounds(v, bound=(e, self.image_wh[i] - e)) for i, (v, e) in enumerate(zip(self.center_pixel_xy, pixel_edge_xy)))
        return replace(self, center_pixel_xy=adjusted_pixel_center_xy)

    def pan_by_pixel_shift(self, pixel_shift_xy: Tuple[float, float], limit: bool = True) -> 'ImageViewInfo':
        new_center_pixel_xy = tuple(np.array(self.center_pixel_xy) + pixel_shift_xy)
        result = replace(self, center_pixel_xy=new_center_pixel_xy)
        if limit:
            result = result.adjust_pan_to_boundary()
        return result

    def pan_by_display_relshift(self, display_rel_xy: Tuple[float, float], limit: bool = True) -> 'ImageViewInfo':
        pixel_shift_xy = np.asarray(display_rel_xy) * self._get_display_wh() / self.zoom_level
        return self.pan_by_pixel_shift(pixel_shift_xy=pixel_shift_xy, limit=limit)

    def pan_by_display_shift(self, display_shift_xy: Tuple[float, float], limit: bool = True) -> 'ImageViewInfo':
        pixel_shift_xy = np.asarray(display_shift_xy) * self.zoom_level
        return self.pan_by_pixel_shift(pixel_shift_xy=pixel_shift_xy, limit=limit)

    def display_xy_to_pixel_xy(self, display_xy: Array["N,2", float], limit: bool = True) -> Array["N,2", float]:
        """ Map pixel-location in display image to pixel image.  Optionally, limit result to bounds of image """
        pixel_xy = reframe_from_a_to_b(
            xy_in_a=display_xy,
            reference_xy_in_b=self.center_pixel_xy,
            reference_xy_in_a=self._get_display_midpoint_xy(),
            scale_in_a_of_b=1 / self.zoom_level,
        )
        if limit:
            pixel_xy = np.maximum(0, np.minimum(self.image_wh, pixel_xy))
        return pixel_xy

    def pixel_xy_to_display_xy(self, pixel_xy: Tuple[float, float], limit: bool = True) -> Tuple[float, float]:
        """ Map pixel-location in image to displayt image """
        display_xy = reframe_from_b_to_a(
            xy_in_b=pixel_xy,
            reference_xy_in_b=self.center_pixel_xy,
            reference_xy_in_a=self._get_display_midpoint_xy(),
            scale_in_a_of_b=1 / self.zoom_level,
        )
        if limit:
            display_xy = np.maximum(0, np.minimum(np.asarray(self._get_display_wh()), display_xy))
        return display_xy

    def create_display_image(self,
                             image: BGRImageArray,
                             gap_color=DEFAULT_GAP_COLOR,
                             scroll_bg_color=BGRColors.DARK_GRAY,
                             scroll_fg_color=BGRColors.LIGHT_GRAY,
                             nearest_neighbor_zoom_threshold: float = 5,
                             ) -> BGRImageArray:

        result_array = np.full(self.window_disply_wh[::-1] + image.shape[2:], dtype=image.dtype, fill_value=gap_color)
        result_array[-self.scroll_bar_width:, :-self.scroll_bar_width] = scroll_bg_color
        result_array[:-self.scroll_bar_width, -self.scroll_bar_width:] = scroll_bg_color

        src_topleft_xy = self.display_xy_to_pixel_xy(display_xy=(0, 0), limit=True).astype(int)
        src_bottomright_xy = self.display_xy_to_pixel_xy(display_xy=self._get_display_wh(), limit=True).astype(int)

        dest_topleft_xy = self.pixel_xy_to_display_xy(pixel_xy=src_topleft_xy, limit=True).astype(int)
        dest_bottomright_xy = self.pixel_xy_to_display_xy(pixel_xy=src_bottomright_xy, limit=True).astype(int)
        (src_x1, src_y1), (src_x2, src_y2) = src_topleft_xy, src_bottomright_xy
        (dest_x1, dest_y1), (dest_x2, dest_y2) = dest_topleft_xy, dest_bottomright_xy

        # Add the image
        src_image = image[src_y1:src_y2, src_x1:src_x2]
        try:
            src_image_scaled = cv2.resize(src_image, (dest_x2 - dest_x1, dest_y2 - dest_y1), interpolation=cv2.INTER_NEAREST if self.zoom_level > nearest_neighbor_zoom_threshold else cv2.INTER_LINEAR)
        except Exception as err:
            print(f"Resize failed on images of shape {src_image} with dest shape {(dest_x2 - dest_x1, dest_y2 - dest_y1)} and interpolation {cv2.INTER_NEAREST if self.zoom_level > nearest_neighbor_zoom_threshold else cv2.INTER_LINEAR}")
            raise err
        result_array[dest_y1:dest_y2, dest_x1:dest_x2] = src_image_scaled

        # Add the scroll bars
        scroll_fraxs_x: Tuple[float, float] = (src_x1 / image.shape[1], src_x2 / image.shape[1])
        scroll_fraxs_y: Tuple[float, float] = (src_y1 / image.shape[0], src_y2 / image.shape[0])
        space_x, space_y = self._get_display_wh()
        scroll_bar_x_slice = slice(max(0, round(scroll_fraxs_x[0] * space_x)), min(space_x, round(scroll_fraxs_x[1] * space_x)))
        scroll_bar_y_slice = slice(max(0, round(scroll_fraxs_y[0] * space_y)), min(space_y, round(scroll_fraxs_y[1] * space_y)))
        result_array[scroll_bar_y_slice, -self.scroll_bar_width:] = scroll_fg_color
        result_array[-self.scroll_bar_width:, scroll_bar_x_slice] = scroll_fg_color

        return result_array


def load_artemis_image(which: str = 'statue') -> BGRImageArray:
    path = {
        'statue': os.path.join(os.path.split(os.path.abspath(__file__))[0], 'artemis.jpeg'),
        'drawing': os.path.join(os.path.split(os.path.abspath(__file__))[0], 'artemis-drawing.jpeg'),
    }[which]
    # path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'artemis.jpeg')
    return cv2.imread(path)

# @dataclass
# class ImageBoxViewer:
#     scroll_bar_width: int = 10
#     background_colour: BGRColorTuple = DEFAULT_GAP_COLOR
#     _canvas_cache: Optional[np.ndarray] = None
#
#     def view_box(self,
#                  source_image: BGRImageArray,
#                  center_pixel_xy: Tuple[int, int],  # (x,y) position of center in coordinates of source_image pixels
#                  window_disply_wh: Tuple[int, int],  # (width, height) of display window
#                  zoom_level: float = 0.,  #
#                  ) -> ImageViewInfo:
#         if self._canvas_cache is None or self._canvas_cache.shape[2] != (window_disply_wh[1], window_disply_wh[0]):
#             self._canvas_cache = np.empty((window_disply_wh[1], window_disply_wh[0])+source_image.shape[2:], dtype=source_image.dtype)
#
#         # TODO: Fill in


#
# def mask_to_boxes(mask: MaskImageArray) -> Sequence[BoundingBox]:
#
#     mask = np.pad(mask, pad_width=[(0, 1), (0, 1)])
#
#
#     # x_stops = ~padmask[:-1, 1:] & padmask[:-1, :-1]
#     # y_stops = ~padmask[1:, :-1] & padmask[:-1, :-1]
#
#     x_stop_shifted = np.zeros_like(mask)
#     for i in range(mask.shape[0]):
#         x_fill_active = False
#         for j in range(mask.shape[1]):
#             above_cell_is_true = i>0 and x_stop_shifted[i-1, j]
#             x_fill_active = mask[i, j] or (x_fill_active and above_cell_is_true)
#             x_stop_shifted[i, j] = x_fill_active
#
#     y_stop_shifted = np.zeros_like(mask)
#     for j in range(mask.shape[1]):
#         y_fill_active = False
#         for i in range(mask.shape[0]):
#             left_cell_is_true = j>0 and x_stop_shifted[i, j-1]
#             y_fill_active = mask[i, j] or (y_fill_active and left_cell_is_true)
#             y_stop_shifted[i, j] = y_fill_active
#
#
#
#     new_xstops = x_stop_shifted[:-1, :-1] & ~x_stop_shifted[:-1, 1:]
#     new_ystops = y_stop_shifted[:-1, :-1] & ~y_stop_shifted[1:, :-1]
#     corner_mask = new_xstops & new_ystops
#     return corner_mask
#     # print('Here')
#
#     # for i, j in itertools.product(range(mask.shape[0]), range(mask.shape[1])):
#
#
#
#     #
#     # for i, row in enumerate(x_stops):
#     #
#
#
#
#
#     bigshape = tuple(s for s in mask.shape)
#     xstops = np.zeros(bigshape, dtype=bool)
#     ystops = np.zeros(bigshape, dtype=bool)
#     xystops = np.zeros(bigshape, dtype=bool)
#
#     print('here')
#     for i in range(1, mask.shape[0]):
#         for j in range(1, mask.shape[1]):
#             xstop_above = xstops[i-1, j]
#             this_is_an_xstop = (mask[i, j-1] and not mask[i, j]) and not mask[i-1, j]  # x-stops propagate down
#             xstops[i, j] = xstop_above or this_is_an_xstop
#
#             ystop_left = j > 0 and ystops[i, j-1]
#             this_is_a_ystop = (i > 0) and (mask[i-1, j] and not mask[i, j]) and ((j==0) or not mask[i, j-1])
#             ystops[i, j] = ystop_left or this_is_a_ystop
#
#             if xstops[i, j] and ystops[i, j]:
#                 xystops[i, j] = True
#                 xstops[i, j] = ystops[i, j] = False
#
#     return xystops
def read_image_time_or_none(image_path: str) -> Optional[float]:
    """ Get the epoch time of the image as a float """
    with open(image_path, 'rb') as image_file:
        try:
            exif_data = exif.Image(image_file)
        except Exception as err:
            return None
        if exif_data.has_exif:
            # parse string like '2022:11:20 14:03:13' into datetime
            datetime_obj = datetime.strptime(exif_data.datetime_original, '%Y:%m:%d %H:%M:%S')
            return datetime_obj.timestamp()
        else:
            return None
