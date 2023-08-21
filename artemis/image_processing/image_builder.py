from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from typing import Tuple, Union, Optional, Sequence, Mapping, Iterable

import cv2
import numpy as np

from artemis.general.custom_types import BGRImageArray, XYPointTuple, IJPixelTuple, HeatMapArray, XYSizeTuple, BGRColorTuple, Array, GreyScaleImageArray, BGRFloatImageArray
from artemis.plotting.easy_window import ImageRow, ImageCol, put_text_at, put_text_in_corner
from artemis.image_processing.image_utils import heatmap_to_color_image, BoundingBox, BGRColors, DEFAULT_GAP_COLOR, RelativeBoundingBox, TextDisplayer


@dataclass
class ImageBuilder:
    image: BGRImageArray
    resolution: float = 1.  # In distance_unit/pix
    origin: Tuple[float, float] = (0., 0.)
    y_from_bottom: bool = False

    def __post_init__(self):
        self.image = np.ascontiguousarray(self.image)  # Some opencv functions expect this

    def get_xlims(self) -> Tuple[float, float]:
        return self.origin[0], self.origin[0] + self.image.shape[1] / self.resolution

    def get_ylims(self) -> Tuple[float, float]:
        return self.origin[1], self.origin[1] + self.image.shape[0] / self.resolution

    def _xy_to_ji(self, center_xy: XYPointTuple) -> IJPixelTuple:
        cx, cy = center_xy
        ox, oy = self.origin
        cj, ci = round((cx - ox) / self.resolution), round((cy - oy) / self.resolution)
        if self.y_from_bottom:
            ci = self.image.shape[0] - ci - 1
        return cj, ci

    @classmethod
    def from_image(cls, image: Union[str, BGRImageArray], normalize: bool = False, y_from_bottom: bool = False, copy: bool = True
                   ) -> 'ImageBuilder':

        if isinstance(image, str):
            image = cv2.imread(os.path.expanduser(image))

        size = image.shape[1], image.shape[0]
        resolution = 1. / max(size) if normalize else 1.
        return ImageBuilder(image=image.copy() if copy else image, resolution=resolution, y_from_bottom=y_from_bottom)

    @classmethod
    def from_heatmap(cls, heatmap: HeatMapArray, assume_zero_min: bool = False, show_range: bool = False, additional_text: Optional[str] = None):

        return cls.from_image(heatmap_to_color_image(heatmap, assume_zero_min=assume_zero_min, show_range=show_range, additional_text=additional_text))

    @classmethod
    def from_text(cls, text: str, text_displayer: Optional[TextDisplayer] = None) -> 'ImageBuilder':
        if text_displayer is None:
            text_displayer = TextDisplayer()
        return ImageBuilder(text_displayer.render(text))

    def stack_with(self, image_or_builder: Union['ImageBuilder', BGRImageArray]) -> 'ImageBuilder':
        image = image_or_builder.image if isinstance(image_or_builder, ImageBuilder) else image_or_builder
        return ImageBuilder(ImageRow(self.image, image).render(), resolution=self.resolution)

    @classmethod
    def from_blank(cls, size: XYSizeTuple, color: BGRColorTuple, normalize: bool = False, y_from_bottom: bool = False
                   ) -> 'ImageBuilder':
        sx, sy = size
        image = np.full(shape=(sy, sx, 3), dtype=np.uint8, fill_value=color)
        return ImageBuilder.from_image(image=image, normalize=normalize, y_from_bottom=y_from_bottom)

    def rescale(self, factor: float, interp=cv2.INTER_NEAREST):
        self.image = cv2.resize(self.image, dsize=None, fx=factor, fy=factor, interpolation=interp)
        return self

    def copy(self) -> 'ImageBuilder':
        return ImageBuilder(image=self.image.copy(), origin=self.origin, resolution=self.resolution)

    def get_crop(self, box: BoundingBox, copy=False, wrap_x=False) -> 'ImageBuilder':
        box = box.scale_by(self.resolution)
        return ImageBuilder(
            image=box.slice_image(self.image, copy=copy, wrap_x=wrap_x),
            origin=(box.x_min, box.y_min),
            resolution=self.resolution
        )

    def get_center_crop(self, size_xy: Tuple[float, float]):
        size_pix_xy = (int(u/self.resolution) for u in size_xy)


    def get_downscaled(self, factor: float) -> 'ImageBuilder':
        return ImageBuilder(
            cv2.resize(self.image, dsize=None, fx=1 / factor, fy=1 / factor),
            origin=self.origin,
            resolution=self.resolution * factor
        )

    def get_image(self, copy: bool = False) -> BGRImageArray:
        return self.image.copy() if copy else self.image

    def get_size_xy(self) -> Tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]

    def draw_points(self, points_xy: Array['N,2', int], color: BGRColorTuple) -> 'ImageBuilder':
        self.image[points_xy[:, 1], points_xy[:, 0]] = color
        return self

    def draw_circle(self, center_xy: XYPointTuple, radius: float, colour: BGRColorTuple, thickness: int = 1) -> 'ImageBuilder':
        center_ji = self._xy_to_ji(center_xy)
        cv2.circle(self.image, center_ji, radius=round(radius / self.resolution), color=colour, thickness=thickness)
        return self

    def draw_line(self, start_xy: Tuple[float, float], end_xy: Tuple[float, float], color: BGRColorTuple, thickness: int = 1) -> 'ImageBuilder':
        start_ji, end_ji = self._xy_to_ji(start_xy), self._xy_to_ji(end_xy)
        cv2.line(self.image, pt1=start_ji, pt2=end_ji, color=color, thickness=thickness)
        return self

    def draw_arrow(self, start_xy: Tuple[float, float], end_xy: Tuple[float, float], color: BGRColorTuple, thickness: int = 1, tip_frac=0.25)  -> 'ImageBuilder':
        start_ji, end_ji = self._xy_to_ji(start_xy), self._xy_to_ji(end_xy)
        cv2.arrowedLine(self.image, pt1=start_ji, pt2=end_ji, color=color, thickness=thickness, tipLength=tip_frac)
        return self

    def draw_box(self, box: BoundingBox | RelativeBoundingBox, colour: BGRColorTuple = BGRColors.RED,
                 secondary_colour: Optional[BGRColorTuple] = None,
                 text_background_color: Optional[BGRColorTuple] = None,
                 text_color: Optional[BGRColorTuple] = None,
                 text_scale = 0.7,
                 label: Optional[str] = None,
                 thickness: int = 1, box_id: Optional[int] = None,
                 as_circle: bool = False,
                 include_labels = True, show_score_in_label: bool = True,  score_as_pct: bool = False) -> 'ImageBuilder':
        if text_color is None:
            text_color = colour
        if isinstance(box, RelativeBoundingBox):
            box = box.to_bounding_box((self.image.shape[1], self.image.shape[0]))
        # xmin, xmax, ymin, ymax = xx_yy_box
        jmin, imin = self._xy_to_ji((box.x_min, box.y_min))
        jmax, imax = self._xy_to_ji((box.x_max, box.y_max))
        imean, jmean = box.to_ij()
        if as_circle:
            cv2.circle(self.image, center=(jmean, imean), radius=round((jmax-jmin)/2), color=colour, thickness=thickness)
            if secondary_colour is not None:
                cv2.circle(self.image, center=(jmean, imean), radius=round((jmax-jmin)/2)+thickness, color=secondary_colour, thickness=thickness)
        else:
            cv2.rectangle(self.image, pt1=(jmin, imin), pt2=(jmax, imax), color=colour, thickness=thickness)
            if secondary_colour is not None:
                cv2.rectangle(self.image, pt1=(jmin-thickness, imin-thickness), pt2=(jmax+thickness, imax+thickness), color=secondary_colour, thickness=thickness)

        # if box.label or box_id is not None:
        if label is None:
            label = ','.join(str(i) for i in [box_id, box.label, None if not show_score_in_label else f"{box.score:.0%}" if score_as_pct else f"{box.score:.2f}"] if i is not None)
        if include_labels:

            put_text_at(self.image, text=label,
                        position_xy=(jmean, imin if box.y_min > box.y_max-box.y_min else imax),
                        anchor_xy=(0.5, 0.) if as_circle else (0., 0.),
                        scale=text_scale*self.image.shape[1]/640,
                        color=text_color,
                        shadow_color = BGRColors.BLACK,
                        background_color=text_background_color,
                        thickness=2
                        )
            # cv2.putText(self.image, text=label, org=(imin, jmin), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=.7*self.image.shape[1]/640,
            #             color=colour, thickness=thickness)

        return self

    def label_points(self, points_xy: Union[Sequence[Tuple[int, int]], Mapping[Tuple[int, int], str]], radius=10, color: Union[BGRColorTuple, Iterable[BGRColorTuple]]=BGRColors.WHITE, thickness=2
                     ) -> 'ImageBuilder':

        if isinstance(color, tuple) and len(color)==3 and all(isinstance(c, int) for c in color):
            color = itertools.cycle([color])
        if isinstance(points_xy, (list, tuple, np.ndarray)):
            points_xy = {(x, y): str(i) for i, (x, y) in enumerate(points_xy)}
        for ((x, y), label), c in zip(points_xy.items(), color):
            cv2.circle(self.image, center=(round(x), round(y)), radius=radius, color=c, thickness=thickness)
            put_text_at(self.image, text=label, position_xy=(round(x)+10, round(y)+10), color=c, shadow_color=None)
        return self

    def draw_bounding_boxes(self,
                            boxes: Iterable[BoundingBox | RelativeBoundingBox],
                            colour: BGRColorTuple = BGRColors.WHITE,
                            secondary_colour: Optional[BGRColorTuple] = BGRColors.BLACK,
                            text_background_colors: Optional[Iterable[BGRColorTuple]] = None,
                            text_colors: Optional[Iterable[BGRColorTuple]] = None,
                            thickness: int = 2,
                            text_scale=0.7,
                            score_as_pct: bool = False,
                            include_labels: bool = True,
                            show_score_in_label: bool = False,
                            include_inset = False,
                            inset_zoom_factor = 3,
                            as_circles: bool = False,
                            ) -> 'ImageBuilder':

        original_image = self.image.copy()
        if text_background_colors is None:
            text_background_colors = (None for _ in itertools.count(0))
        if text_colors is None:
            text_colors = (None for _ in itertools.count(0))
        for bb, bg, tc in zip(boxes, text_background_colors, text_colors):
            self.draw_box(bb, colour=colour, text_color=tc, secondary_colour=secondary_colour, text_background_color=bg, thickness=thickness, score_as_pct=score_as_pct, show_score_in_label=show_score_in_label,
                          include_labels=include_labels, text_scale=text_scale, as_circle=as_circles)
        if include_inset:
            self.draw_corner_inset(
                ImageRow(*(ImageBuilder(b.slice_image(original_image)).rescale(inset_zoom_factor).image for b in boxes)).render(),
                corner='br', border_color=colour, secondary_border_color=secondary_colour, border_thickness=thickness)
        return self

    def draw_border(self, color: BGRColorTuple, thickness: int = 2, external: bool = False, sides='ltrb') -> 'ImageBuilder':
        # border_ixs = list(range(thickness))+list(range(-thickness, 0))
        # self.image[border_ixs, border_ixs] = color

        l, t, r, b = ((s in sides)*thickness for s in 'ltrb')

        if external:
            new_image = np.empty(shape=(self.image.shape[0]+t+b, self.image.shape[1]+l+r, self.image.shape[2]), dtype=np.uint8)
            new_image[t:-b or None, l:-r or None] = self.image
            self.image = new_image
        h, w = self.image.shape[:2]
        self.image[:t, :] = color
        self.image[h-b:, :] = color
        self.image[:, :l] = color
        self.image[:, w-r:] = color
        return self
        # return self.draw_box(BoundingBox.from_ltrb(0, 0, self.image.shape[1]-1, self.image.shape[0]-1), thickness=thickness, colour=color, include_labels=False)

    def draw_zoom_inset_from_box(self, box: BoundingBox, scale_factor: int, border_color=BGRColors.GREEN, border_thickness: int = 2, corner = 'br', backup_corner='bl') -> 'ImageBuilder':
        # TODO: Make it nor crash when box is too big
        assert corner in ('br', 'tr', 'bl', 'tl')
        assert backup_corner in ('br', 'tr', 'bl', 'tl')
        sub_image = box.slice_image(self.image)
        sub_image = cv2.resize(sub_image, dsize=None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
        corner_size = max(sub_image.shape[:2])/min(self.get_size_xy())

        self.draw_bounding_boxes([box], colour=border_color, thickness=border_thickness)

        cx, cy = np.array(box.get_center())/self.get_size_xy()
        inset_corner = ('t' if cy<corner_size else 'b' if cy > (1-corner_size) else '-') + ('l' if cx<corner_size else 'r' if cx > (1-corner_size) else '-')
        self.draw_corner_inset(sub_image, border_color=border_color, corner=backup_corner if corner==inset_corner else corner)
        return self

    def draw_corner_inset(self, image: BGRImageArray, corner='br', border_color=BGRColors.RED, secondary_border_color: Optional[BGRColorTuple] = None, border_thickness: int = 2) -> 'ImageBuilder':
        # TODO: Make it nor crash when box is too big
        assert corner in ('br', 'tr', 'bl', 'tl')
        vc, hc = corner
        vslice = slice(self.image.shape[0]-image.shape[0], self.image.shape[0]) if vc == 'b' else slice(0, image.shape[0])
        hslice = slice(self.image.shape[1]-image.shape[1], self.image.shape[1]) if hc == 'r' else slice(0, image.shape[1])
        vb_slice = slice(max(0, vslice.start-border_thickness), vslice.stop+border_thickness)
        hb_slice = slice(max(0, hslice.start-border_thickness), hslice.stop+border_thickness)
        if secondary_border_color is not None:
            self.image[max(0, vb_slice.start-border_thickness): vb_slice.stop+border_thickness, max(0, hb_slice.start-border_thickness): hb_slice.stop+border_thickness] = secondary_border_color
        self.image[vb_slice, hb_slice] = border_color
        self.image[vslice, hslice] = image
        return self

    def draw_text_label(self, label: str, top_side: bool = True, rel_font_size: float = 0.05, color: BGRColorTuple = BGRColors.WHITE, background_color: Optional[BGRColorTuple] = None, thickness: int = 2) -> 'ImageBuilder':
        text_image = ImageBuilder.from_text(text=label, text_displayer=TextDisplayer(text_color=color, background_color=background_color, scale=rel_font_size*self.image.shape[0]/20., thickness=thickness)).get_image()
        self.image = ImageCol(text_image, self.image).render() if top_side else ImageCol(self.image, text_image).render()
        return self

    def draw_text(self, text: str, loc_xy: Tuple[int, int], colour: BGRColorTuple, anchor_xy: Tuple[float, float] = (0., 0.), shadow_color: Optional[BGRColorTuple] = None, background_color: Optional[BGRColorTuple] = None, thickness=1,
                  scale=1., font=cv2.FONT_HERSHEY_PLAIN
                  ) -> 'ImageBuilder':

        put_text_at(self.image, position_xy=loc_xy, text=text, color=colour, shadow_color=shadow_color, background_color=background_color, anchor_xy=anchor_xy, scale=scale, thickness=thickness, font=font)

        # cv2.putText(self.image, text=text, org=self._xy_to_ji(loc), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=scale, color=colour, thickness=thickness)
        return self

    def draw_corner_text(self, text: str, colour: BGRColorTuple, shadow_color: Optional[BGRColorTuple] = None, background_color: Optional[BGRColorTuple] = None,
                         corner = 'tl', scale=1, thickness=1, font=cv2.FONT_HERSHEY_PLAIN)  -> 'ImageBuilder':
        put_text_in_corner(self.image, text=text, color=colour, shadow_color=shadow_color, background_color=background_color, corner=corner, scale=scale, thickness=thickness, font=font)
        # cv2.putText(self.image, text=text, org=(0, 10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=colour)
        return self

    def draw_image_at(self, image: BGRImageArray, loc: Union[str, Tuple[float, float]], padding: int = 0, pad_colour: BGRColorTuple = DEFAULT_GAP_COLOR)  -> 'ImageBuilder':

        # if padding:  # Unnecessary copy here.
        #     gap_image = create_gap_image(size=(image.shape[1]+2*padding, image.shape[0]+2*padding))
        #     gap_image[padding:-padding, padding:-padding] = image
        #     image = gap_image
        if isinstance(loc, str):
            i, j = {
                'tl': (0, 0),
                'bl': (self.image.shape[0] - image.shape[0] - 2 * padding, 0),
                'tr': (0, self.image.shape[1] - image.shape[1] - 2 * padding),
                'br': (self.image.shape[0] - image.shape[0] - 2 * padding, self.image.shape[1] - image.shape[1] - 2 * padding),
            }[loc]
        else:
            j, i = self._xy_to_ji(loc)
        if padding:
            self.image[i: i + image.shape[0] + 2 * padding, j: j + image.shape[1] + 2 * padding] = pad_colour
        self.image[i + padding: i + padding + image.shape[0], j + padding:j + padding + image.shape[1]] = image
        return self

    def draw_noise(self, amplitude: float, seed: Optional[int] = None):
        noise = np.random.RandomState(seed).randn(*self.image.shape)*amplitude
        self.image = np.clip((self.image.astype(np.float32) + noise), 0, 255).astype(np.uint8)
        return self

    def superimpose(self, image: Union[BGRImageArray, GreyScaleImageArray, BGRFloatImageArray])  -> 'ImageBuilder':
        if image.ndim == 2:
            image = image[:, :, None]
        self.image[:] = np.clip(self.image + image.astype(np.float32), 0, 255).astype(np.uint8)
        return self

    def hstack_with(self, other: Union[ImageBuilder, BGRImageArray]) -> ImageBuilder:
        other_image = other.image if isinstance(other, ImageBuilder) else other
        self.image = ImageRow(self.image, other_image).render()
        return self


    def vstack_with(self, other: Union[ImageBuilder, BGRImageArray]):
        other_image = other.image if isinstance(other, ImageBuilder) else other
        self.image = ImageCol(self.image, other_image).render()
        return self

