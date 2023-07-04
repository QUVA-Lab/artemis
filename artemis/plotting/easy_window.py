from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from math import ceil
from typing import Union, Hashable, Dict, Tuple, List, Set, Optional, Sequence, Iterable, ContextManager, Callable

import cv2
import numpy as np
import rpack
from attr import attrib, attrs
from rpack import PackingImpossibleError

from artemis.general.custom_types import BGRColorTuple, BGRImageArray, Array
from artemis.plotting.cv_keys import Keys, cvkey_to_key
from artemis.image_processing.image_utils import BGRColors, DEFAULT_GAP_COLOR, create_gap_image, normalize_to_bgr_image, \
    TextDisplayer, heatmap_to_greyscale_image
from artemis.plotting.cv2_plotting import hold_alternate_show_func

DEFAULT_WINDOW_NAME = 'Window'


class InputTimeoutError(Exception):
    """ Raised if you have not received user input within the timeout """
    pass


def cv_window_input(  # Get text input from a cv2 window.
        prompt: str,  # A text prompt
        window_size: Optional[Tuple[int, int]] = None,  # Optionally, window size (otherwise it will just expand to fit)
        timeout=30,  # Timeout for user input (raise InputTimeoutError if no response in this time)
        return_none_if_timeout=True,  # Just return None if timeout
        text_color=BGRColors.WHITE,  # Text color
        background_color=BGRColors.DARK_GRAY,  # Background color
        window_name='User Input (Enter to complete, Exc to Cancel)'  # Name of CV2 windot
) -> Optional[str]:  # The Response, or None if you press ESC

    displayer = TextDisplayer(text_color=text_color, background_color=background_color, size=window_size)
    next_cap = False
    character_keys = {Keys.SPACE: '  ', Keys.PERIOD: '.>', Keys.COMMA: ',<', Keys.SEMICOLON: ';:', Keys.SLASH: '/?',
                      Keys.DASH: '-=', Keys.EQUALS: '=+'}
    response = ''
    while True:
        img = displayer.render('{}\n >> {}'.format(prompt, response))
        cv2.imshow(window_name, img)
        key = cvkey_to_key(cv2.waitKey(int(timeout * 1000)))
        if key is None:
            if return_none_if_timeout:
                return None
            else:
                raise InputTimeoutError("User provided no input for {:.2f}s".format(timeout))
        elif key == Keys.RETURN:
            cv2.destroyWindow(window_name)
            return response
        elif key == Keys.ESC:
            cv2.destroyWindow(window_name)
            return None
        elif len(key) == 1:
            response += key.upper() if next_cap else key.lower()
            next_cap = False
        elif key in character_keys:
            base_key, shift_key = character_keys[key]
            response += shift_key if next_cap else base_key
            next_cap = False
        elif key == Keys.PERIOD:
            response += '.'
        elif key == Keys.PERIOD:
            response += '.'
        elif key == Keys.BACKSPACE:
            response = response[:-1]
        elif key in (Keys.LSHIFT, Keys.RSHIFT):
            next_cap = True
        else:
            print("Don't know how to handle key '{}'.  Skipping.".format(key))


# def put_text_at(img, text, pos=(0, -1), scale=1, color=(0, 0, 0), shadow_color: Optional[Tuple[int, int, int]] = None, background_color: Optional[Tuple[int, int, int]] = None, thickness=1, font=cv2.FONT_HERSHEY_PLAIN,
#                 dry_run=False):
#     """
#     Add text to an image
#     :param img:  add to this image
#     :param text: add this text
#     :param pos:  (x, y) location of text to add, pixel values, point indicates bottom-left corner of text.
#     :param scale: size of text to add
#     :param color:  (r,g,b) uint8
#     :param thickness:  for adding text
#     :param font:  font constant from cv2
#     :param dry_run:  don't add to text, just calculate size required to add text
#     :return:  dict with 'x': [x_min, x_max], 'y': [y_min, y_max], 'baseline': location of text baseline relative to y_max
#     """
#     (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
#     y_pos = pos[1] + baseline if pos[1] >= 0 else img.shape[0] + pos[1] + baseline
#     x_pos = pos[0] if pos[0] >= 0 else img.shape[1] + pos[0]
#     box = {'y': [y_pos - h, y_pos],
#            'x': [x_pos, x_pos + w],
#            'baseline': baseline}
#
#     if background_color is not None:
#         pad = 4
#         img[max(0, pos[1]-h-pad): pos[1]+pad, max(0, pos[0]-pad): pos[0]+w+pad] = background_color
#     if not dry_run:
#         if shadow_color is not None:
#             cv2.putText(img, text, (x_pos, y_pos-h//2), font, scale, shadow_color, thickness + 2, bottomLeftOrigin=False)
#         cv2.putText(img, text, (x_pos, y_pos-h//2), font, scale, color, thickness, bottomLeftOrigin=False)
#     return box


def resize_to_fit_in_box(img: BGRImageArray, size: Union[int, Tuple[int, int]], expand=True, shrink=True,
                         interpolation=cv2.INTER_LINEAR):
    """
    Resize an image to fit in a box
    :param img: Imageeeee
    :param size: Box (width, height) in pixels
    :param expand: Expand to fit
    :param shrink: Shrink to fit
    :param interpolation: cv2 Interpolation enum, e.g. cv2.INTERP_NEAREST
    :return: The resized images
    """
    assert img.size > 0, f"Got an image of shape {img.shape} - which contains a zero dimension"

    if isinstance(size, (int, float)):
        size = (size, size)
    ratio = min(float(size[0]) / img.shape[1] if size[0] is not None else float('inf'),
                float(size[1]) / img.shape[0] if size[1] is not None else float('inf'))
    if (shrink and ratio < 1) or (expand and ratio > 1):
        img = cv2.resize(img, fx=ratio, fy=ratio, dsize=None, interpolation=interpolation)
    return img


def put_text_in_corner(img: BGRImageArray, text: str, color: BGRColorTuple, shadow_color: Optional[BGRColorTuple] = None,
                       background_color: Optional[BGRColorTuple] = None, corner ='tl', scale=1, thickness=1, font=cv2.FONT_HERSHEY_PLAIN, ):
    """ Put text in the corner of the image"""
    # TODO: Convert to use put_text_at
    assert corner in ('tl', 'tr', 'bl', 'br')
    cv, ch = corner
    (twidth, theight), baseline = cv2.getTextSize(text, font, scale, thickness)
    position = ({'l': 0, 'r': img.shape[1]-twidth}[ch], {'t': theight+baseline, 'b': img.shape[0]}[cv])
    if background_color is not None:
        pad = 4
        img[max(0, position[1]-theight-pad): position[1]+pad, max(0, position[0]-pad): position[0]+twidth+pad] = background_color
    if shadow_color is not None:
        cv2.putText(img=img, text=text, org=position, fontFace=font, fontScale=scale, color=shadow_color, thickness=thickness + 2, bottomLeftOrigin=False)
    cv2.putText(img=img, text=text, org=position, fontFace=font, fontScale=scale, color=color, thickness=thickness, bottomLeftOrigin=False)



def put_text_at(
        img: BGRImageArray,
        text: str,
        color: BGRColorTuple,
        position_xy: Tuple[int, int] = None,
        anchor_xy: Tuple[float, float] = (0., 0.),  # Position of anchor relative to width/height of text area
        shadow_color: Optional[BGRColorTuple] = None,
        background_color: Optional[BGRColorTuple] = None,
        scale=1,
        thickness=1,
        font=cv2.FONT_HERSHEY_PLAIN,
        shift_down_by_baseline: bool = False,
    ):
    for i, line in enumerate(text.split('\n')):
        (twidth, theight), baseline = cv2.getTextSize(line, font, scale, thickness)
        px, py = position_xy
        if px < 0:
            px = img.shape[1]+px
        if py < 0:
            py = img.shape[0]+py
        ax, ay = anchor_xy
        px = round(px - ax*twidth)
        py = round(py - ay*theight) + (2*baseline if shift_down_by_baseline else 0) + i*(theight+baseline)

        if background_color is not None:
            pad = 4
            img[max(0, py-pad): py+theight+pad, max(0, px-pad): px+twidth+pad] = background_color
        if shadow_color is not None:
            cv2.putText(img=img, text=line, org=(px, py), fontFace=font, fontScale=scale, color=shadow_color, thickness=thickness + 3, bottomLeftOrigin=False)
        cv2.putText(img=img, text=line, org=(px, py), fontFace=font, fontScale=scale, color=color, thickness=thickness, bottomLeftOrigin=False)


def draw_matrix(
        matrix: Array['H,W', float],
        approx_size_wh: Tuple[float, float] = (400, 400),
        text_color = BGRColors.GREEN,
        row_headers: Optional[Sequence[str]] = None,
        col_headers: Optional[Sequence[str]] = None,
        header_pad: int = 100
    ) -> BGRImageArray:

    w, h = approx_size_wh
    if matrix.size==0:
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        put_text_at(blank, "No data", position_xy=(w//2, h//2), anchor_xy=(0.5, 0.5), scale=1, thickness=1, color=text_color, shadow_color=(0, 0, 0))
        return blank


    heatmap = cv2.resize(matrix, (w, h), interpolation=cv2.INTER_NEAREST)

    rowpad = 0 if row_headers is None else header_pad
    colpad = 0 if col_headers is None else header_pad

    heatmap = np.pad(heatmap, ((rowpad, 0), (colpad, 0)))

    img = heatmap_to_greyscale_image(heatmap, assume_zero_center=True, assume_zero_min=False)
    for i, j in np.ndindex(matrix.shape):
        put_text_at(img, f"{matrix[i, j]:.2f}", position_xy=(rowpad+(j+.5)*w/matrix.shape[1], colpad+(i+.5)*h/matrix.shape[0]), anchor_xy=(0.5, 0.5), scale=1.5, thickness=1, color=text_color, shadow_color=(0, 0, 0))
        # cv2.circle(img, (int(i*w/matrix.shape[1]), int(j*h/matrix.shape[0])), 3, (255, 255, 255), -1)
    if row_headers is not None:
        for i, header in enumerate(row_headers):
            put_text_at(img, str(header), position_xy=(rowpad//2, colpad+(i+.5)*h/matrix.shape[0]), anchor_xy=(0.5, 0.5), scale=1.5, thickness=1, color=text_color, shadow_color=(0, 0, 0))
    if col_headers is not None:
        for j, header in enumerate(col_headers):
            put_text_at(img, str(header), position_xy=(rowpad+(j+.5)*w/matrix.shape[1], colpad//2), anchor_xy=(0.5, 0.5), scale=1.5, thickness=1, color=text_color, shadow_color=(0, 0, 0))
    return img


def draw_image_to_region_inplace(  # Assign an image to a region in a parent image
        parent_image,  # type: array(WP,HP,...)[uint8]  # The parent image into which to draw inplace
        img,  # type: array(W,H,...)[uint8]  # The image to draw in the given region
        xxyy_region=None,
        # type: Optional[Tuple[int, int, int, int]]  # The (left, right, bottom, top) edges (right/top-non-inclusive) to drawbottom,
        expand=True,  # True to expand image to fill region
        gap_colour=None,  # type: Optional[Tuple[int, int, int]]  # BGR fill colour
        fill_gap=True  # True to fill in the gaps at the edges with fill colour
):
    if gap_colour is None:
        gap_colour = DEFAULT_GAP_COLOR

    x1, x2, y1, y2 = xxyy_region if xxyy_region is not None else (0, parent_image.shape[1], 0, parent_image.shape[0])
    width, height = x2 - x1, y2 - y1
    resized = resize_to_fit_in_box(img, size=(width, height), expand=expand, shrink=True)
    xs, ys = (x1 + (width - resized.shape[1]) // 2), (y1 + (height - resized.shape[0]) // 2)
    if fill_gap:
        parent_image[y1:y2, x1:x2] = gap_colour
    if resized.ndim == 2:
        resized = resized[:, :, None]
    parent_image[ys:ys + resized.shape[0], xs:xs + resized.shape[1]] = resized


def draw_multiple_images_inplace(  # Draw multiple images into a parent image
        parent_image,  # type: 'array(HP,WP,3)[uint8]'  # The image in which to draw
        image_dict,  # type: Dict[str, 'array(H,W,3)[uint8]']  # The images to insert
        xxyy_dict,
        # type: Dict[str, Tuple[int, int, int, int]]  # The bounding boxes (referenced by the same keys as image_dict)
        float_norm='max',  # type: str  # How to map floats to color range.  See to_uint8_color_image
        expand=True,  # True to expand image to fit bounding box
        gap_colour=None,  # type: Optional[Tuple[int, int, int]]  # The Default BGR color to fill the gaps
):
    for name, img in image_dict.items():
        assert name in xxyy_dict, "There was no bounding box for image named '{}'".format(name)
        draw_image_to_region_inplace(parent_image=parent_image, img=img, xxyy_region=xxyy_dict[name], expand=expand,
                                     gap_colour=gap_colour, fill_gap=False)


@attrs
class WindowLayout(object):
    """ Object defining the location of named boxes within a window. """
    panel_xxyy_boxes = attrib(type=Dict[str, Tuple[int, int, int, int]])
    size = attrib(type=Tuple[int, int], default=None)

    def __attrs_post_init__(self):
        if self.size is None:
            self.size = (max(x for _, x, _, _ in self.panel_xxyy_boxes.values()),
                         max(y for _, _, _, y in self.panel_xxyy_boxes.values()))

    def render(self,  # Given a dictionary of images, render it into a single image
               image_dict,  # type: Dict[str, 'array(H,W,3)[uint8']
               gap_color=None,  # type: Optional[Tuple[int, int, int]]  # The Default BGR color to fill the gaps
               float_norm='max'  # If images are passed as floats, how to normalize them (see to_uint8_color_image)
               ):  # type: (...) -> 'array(H,W,3)[uint8]'  # The rendered image

        parent_frame = create_gap_image(self.size, gap_colour=gap_color)
        draw_multiple_images_inplace(parent_image=parent_frame, image_dict=image_dict, xxyy_dict=self.panel_xxyy_boxes,
                                     float_norm=float_norm, expand=False)
        return parent_frame


class IBoxPacker(metaclass=ABCMeta):

    @abstractmethod
    def pack_boxes(self,  # Pack images into a resulting images
                   box_size_dict: Dict[str, Tuple[int, int]],
                   ) -> WindowLayout:
        """ Pack boxes into the layout """


def rpack_from_aspect_ratio(sizes: Sequence[Tuple[int, int]], aspect_ratio = 1., max_iter=8) -> Sequence[Tuple[int, int]]:
    """
    Find box corners of the smallest packing solution that fits in the given aspect ratio.
    This is approximate, and tries up to n_iter iterations.
    """
    upper_bound_width = max(sum(w for w, _ in sizes), round(sum(h*aspect_ratio for _, h in sizes)))
    lower_bound_width = max(max(w for w, _ in sizes), round(max(h*aspect_ratio for _, h in sizes)))
    result = None
    width = upper_bound_width
    for i in range(max_iter):
        height = int(width / aspect_ratio)
        try:
            print(f"Trying packing with {width}x{height}")
            result = rpack.pack(sizes, max_width=width, max_height=height)
        except PackingImpossibleError:
            lower_bound_width = width
        else:
            upper_bound_width = width
        width = int(lower_bound_width + upper_bound_width)//2
    assert result is not None, f"Should have been able to pack with width {upper_bound_width}"
    return result


class OptimalBoxPacker(IBoxPacker):

    aspect_ratio = 4/3

    def pack_boxes(self,  # Pack images into a resulting images
                   box_size_dict: Dict[str, Tuple[int, int]],
                   ) -> WindowLayout:
        try:
            import rpack
        except ImportError:
            raise ImportError(f"If you want to use {self.__class__.__name__}, you need to 'pip install rectangle-packer'. ")

        # corners = rpack_from_aspect_ratio(sizes=)
        corners = rpack_from_aspect_ratio(sizes=list(box_size_dict.values()), aspect_ratio=self.aspect_ratio)
        # corners = rpack.pack(sizes=list(box_size_dict.values()))
        return WindowLayout({name: (x, x+w, y, y+h) for (name, (w, h)), (x, y) in zip(box_size_dict.items(), corners)})


class RowColumnPacker:
    """ Use to pack boxes by defining nested rows and columns (use Row/Col subclasses for convenience)

        packer = Row(Col(Row('panel_1', 'panel_2'),
                         'panel_3'),
                     'panel_4')
        layout = packer.pack_boxes({'panel_1': (300, 400), 'panel_2': (200, 200), 'panel_3': (600, 300), 'panel_4': (700, 700)})

    """

    OTHER = None  # Used to indicate that

    def __init__(self, *args: Union['RowColumnPacker', str, None], orientation='h', wrap: Optional[int] = None):
        """
        :param args: The panels in this object.  This can either be a nested Row/Col/RowColumnPacker object, or a string panel name, or RowColumnPacker.OTHER to indicate
            that this panel takes any unclaimed windows.
        :param orientation: How to stack them: 'h' for horizontal or 'v' for vertical
        """
        assert orientation in ('h', 'v')
        assert all(isinstance(obj, Hashable) for obj in args), f"Unhashable elements in args: {args}"
        self.orientation = orientation
        self.items = args
        self.wrap = wrap

    def pack_boxes(self,  # Pack images into a resulting images
                   box_size_dict: Dict[str, Tuple[int, int]],
                   ) -> WindowLayout:

        if self.wrap is not None:
            major_orientation, minor_orientation = 'hv'.replace(self.orientation, ''), self.orientation
            items = [RowColumnPacker(*self.items[i * self.wrap: (i + 1) * self.wrap], orientation=self.orientation) for i in range(ceil(len(box_size_dict) / self.wrap))]
            this_orientation = major_orientation
        else:
            items = self.items
            this_orientation = self.orientation

        new_contents_dict = {}

        # We reorder so that the window with the "OTHER" box is packed last (after all images with panels have been assigned)
        reordered_items = sorted(items, key=lambda item_: isinstance(item_,
                                                                     RowColumnPacker) and RowColumnPacker.OTHER in item_.get_all_items())

        # Get the layout for each sub-widow
        window_layouts: List[WindowLayout] = []
        remaining_boxes = box_size_dict
        for item in reordered_items:
            if isinstance(item, RowColumnPacker):
                window_layouts.append(item.pack_boxes(remaining_boxes))
            elif item in box_size_dict:
                window_layouts.append(WindowLayout(size=box_size_dict[item], panel_xxyy_boxes={
                    item: (0, box_size_dict[item][0], 0, box_size_dict[item][1])}))
            elif item is RowColumnPacker.OTHER:
                child_frame = RowColumnPacker(*remaining_boxes.keys(), orientation=this_orientation)
                window_layouts.append(child_frame.pack_boxes(remaining_boxes))
            else:  # This panel is not included in the data, which is ok, we just skip.
                continue
            remaining_boxes = {name: box for name, box in remaining_boxes.items() if
                               name not in window_layouts[-1].panel_xxyy_boxes}

        # Combine them into a single window layout
        if this_orientation == 'h':
            total_height = max([0] + [layout.size[1] for layout in window_layouts])
            total_width = 0
            for layout in window_layouts:
                width, height = layout.size
                v_offset = (total_height - height) // 2
                for name, (xmin, xmax, ymin, ymax) in layout.panel_xxyy_boxes.items():
                    new_contents_dict[name] = (xmin + total_width, xmax + total_width, ymin + v_offset, ymax + v_offset)
                total_width += width
        else:
            total_width = max([0] + [layout.size[0] for layout in window_layouts])
            total_height = 0
            for layout in window_layouts:
                width, height = layout.size
                h_offset = (total_width - width) // 2
                for name, (xmin, xmax, ymin, ymax) in layout.panel_xxyy_boxes.items():
                    new_contents_dict[name] = [xmin + h_offset, xmax + h_offset, ymin + total_height,
                                               ymax + total_height]
                total_height += height

        return WindowLayout(panel_xxyy_boxes=new_contents_dict, size=(total_width, total_height))

    def get_all_items(self) -> Set[str]:
        return {name for item in self.items for name in
                (item.get_all_items() if isinstance(item, RowColumnPacker) else [
                    item])}  # pylint:disable=superfluous-parens

    def concat(self, *objects):
        return RowColumnPacker(*([self.items] + list(objects)), orientation=self.orientation)

    def __len__(self):
        return len(self.items)


class Row(RowColumnPacker):
    """ A row of panels """

    def __init__(self, *args):
        RowColumnPacker.__init__(self, *args, orientation='h')


class Col(RowColumnPacker):
    """ A column of panels """

    def __init__(self, *args):
        RowColumnPacker.__init__(self, *args, orientation='v')


class ImagePacker:

    def __init__(self, *images, orientation: str, gap_color: BGRColorTuple = DEFAULT_GAP_COLOR, wrap: Optional[int] = None, **named_images):
        self.named_images = OrderedDict([(str(i), im) for i, im in enumerate(images)] + [(name, im) for name, im in named_images.items()])
        self.window = EasyWindow(RowColumnPacker(*self.named_images.keys(), orientation=orientation, wrap=wrap),
                                 gap_color=gap_color, skip_title_for={str(i) for i in range(len(images))}
                                 )

    def render(self) -> BGRImageArray:
        for name, entity in self.named_images.items():
            image = entity.render() if isinstance(entity, ImagePacker) else entity
            self.window.update(image=image, name=name)
        return self.window.render()


class ImageRow(ImagePacker):

    def __init__(self, *args, gap_color: BGRColorTuple = DEFAULT_GAP_COLOR, wrap: Optional[int] = None, **named_image):
        ImagePacker.__init__(self, *args, orientation='h', gap_color=gap_color, wrap=wrap, **named_image)


class ImageCol(ImagePacker):

    def __init__(self, *args, gap_color: BGRColorTuple = DEFAULT_GAP_COLOR, wrap: Optional[int] = None, **named_image):
        ImagePacker.__init__(self, *args, orientation='v', gap_color=gap_color, wrap=wrap, **named_image)


def get_raster_box_packing(sizes: Sequence[Tuple[int, int]], box: Tuple[int, int], truncate_if_full: bool = False) -> Optional[Sequence[Tuple[int, int, int, int]]]:
    """ Find the locations into which to pack windows of the given sizes, returning a (l, t, r, b) box for each, or None if no space. """
    assignment_slices = []
    bx, by = box
    sx, sy = 0, 0
    current_row_height = 0
    while len(assignment_slices) < len(sizes):
        ix, iy = sizes[len(assignment_slices)]
        ex, ey = sx + ix, sy + iy  # First, try the next available slot
        current_row_height = max(current_row_height, iy)
        if ex <= bx:  # It fits horizontally...
            if ey <= by:  # It fits vertically...
                assignment_slices.append((sx, sy, ex, ey))
                sx = ex
                current_row_height = max(current_row_height, iy)
            else:
                return assignment_slices if truncate_if_full else None
                # No more space
        else:  # Move to new row
            sx, sy = 0, sy + current_row_height
            current_row_height = 0
            if sy > by:
                return assignment_slices if truncate_if_full else None
    return assignment_slices


def pack_images_into_box(images: Iterable[BGRImageArray], box: Tuple[int, int], gap_colour=DEFAULT_GAP_COLOR, overflow_handling: str = 'resize') -> BGRImageArray:
    images = list(images)
    while True:
        sizes = [(img.shape[1], img.shape[0]) for img in images]
        boxes = get_raster_box_packing(sizes, box, truncate_if_full=overflow_handling == 'truncate')
        if boxes is not None:
            break
        else:
            images = [cv2.resize(im, dsize=None, fx=0.5, fy=0.5) for im in images]
    base_image = create_gap_image(box, gap_colour=gap_colour)
    for (l, b, r, t), im in zip(boxes, images):
        base_image[b:t, l:r] = im
    return base_image


@attrs
class EasyWindow(object):
    """ Contains multiple updating subplots """
    box_packer = attrib(type=IBoxPacker)
    identifier = attrib(default=DEFAULT_WINDOW_NAME)
    panel_scales = attrib(factory=dict, type=Dict[Optional[str], float])  # e.g. {'panel_name': 2.}
    skip_title_for = attrib(factory=set, type=Set[Optional[str]])  # e.g. {'panel_name'}
    images = attrib(factory=OrderedDict)
    gap_color = attrib(default=DEFAULT_GAP_COLOR)
    title_background = attrib(default=np.array([50, 50, 50], dtype=np.uint8))
    _last_size_and_layout = attrib(type=Optional[Tuple[Dict[str, Tuple[int, int]], WindowLayout]], default=None)

    ALL_SUBPLOTS = None  # Flag that can be used in panel_scales and skip_title_for to indicate that all subplots should have this property

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def update(self,  # Update the window and maybe show it.
               image: BGRImageArray,
               # The data to plot (if display not provided, we try to infer appropriate plot type from data)
               name: str,  # The name of the "plot panel" to plot it in
               scale: Optional[float] = None,  # Optionally, how much to scale the image
               skip_none: bool = False,  # If data is None, just skip the update
               add_title: Optional[bool] = None,  # Add a totle matching the name,
               title: Optional[str] = None,  # Optional title to put at top instead of name
               ):
        if image is None:
            if name in self.images:
                del self.images[name]
            return
        image = normalize_to_bgr_image(image)


        # Allow panel settings to be set
        add_title = add_title if add_title is not None else (
                name not in self.skip_title_for and EasyWindow.ALL_SUBPLOTS not in self.skip_title_for)
        scale = scale if scale is not None else self.panel_scales[
            name] if name in self.panel_scales else self.panel_scales.get(EasyWindow.ALL_SUBPLOTS, None)
        if scale is not None:
            image = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        if add_title:
            title_sec = create_gap_image((image.shape[1], 30), gap_colour=self.title_background)
            put_text_at(img=title_sec, position_xy=(0, 15), text=title or name, color=(255, 255, 255))
            image = np.vstack([title_sec, image])
        if image is None and skip_none:
            return
        self.images[name] = image

    def render(self) -> BGRImageArray:
        """ Render the image into an array """
        sizes = OrderedDict((name, (im.shape[1], im.shape[0])) for name, im in self.images.items())
        if self._last_size_and_layout is not None and self._last_size_and_layout[0] == sizes:
            _, window_layout = self._last_size_and_layout
        else:
            window_layout = self.box_packer.pack_boxes(sizes)  # type: WindowLayout
            self._last_size_and_layout = (sizes, window_layout)
        return window_layout.render(image_dict=self.images, gap_color=self.gap_color)

    def close(self):  # Close this plot window
        try:
            cv2.destroyWindow(self.identifier)
        except cv2.error:
            pass  # It's ok, it's already gone
            print("/\\ Ignore above error, it's fine")  # Because cv2 still prints it


@dataclass
class JustShowCapturer:
    window: EasyWindow = field(default_factory=lambda: EasyWindow(box_packer=OptimalBoxPacker()))
    callback: Optional[Callable[[BGRImageArray], None]] = None
    _any_images_added: bool = False

    def on_just_show(self, image: BGRImageArray, name: str):
        self.window.update(image, name)
        self._any_images_added = True
        if self.callback is not None:
            self.callback(self.window.render())

    def render_captured_images(self) -> Optional[BGRImageArray]:
        if self._any_images_added:
            return self.window.render()
        else:
            return None

    @contextmanager
    def hold_capture(self) -> ContextManager[Callable[[], Optional[BGRImageArray]]]:
        with hold_alternate_show_func(self.on_just_show):
            yield self.render_captured_images


@contextmanager
def hold_just_show_capture(wrap_rows: int = 2) -> ContextManager[Callable[[], BGRImageArray]]:
    with JustShowCapturer().hold_capture() as captured_image_getter:
        yield captured_image_getter
    # yield from JustShowCapturer.from_row_wrap(wrap_rows).hold_capture()
