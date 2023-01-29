from typing import Union, Mapping, Sequence, Tuple

import numpy as np

from artemis.general.custom_types import BGRImageArray, IndexImageArray, BGRColorTuple
from artemis.image_processing.image_utils import DEFAULT_GAP_COLOR, create_gap_image, BGRColors
from artemis.plotting.data_conversion import put_list_of_images_in_array, put_data_in_image_grid, put_data_in_grid
from artemis.plotting.easy_window import put_text_in_corner


def generate_image_mosaic_and_index_grid(
        mosaic: Union[Mapping[int, BGRImageArray], Sequence[BGRImageArray]],
        gap_color: BGRColorTuple = DEFAULT_GAP_COLOR,
        desired_aspect_ratio = 1.,  # TODO: Make it work
        min_size_xy: Tuple[int, int] = (640, 480),
        padding: int = 1,
       ) -> Tuple[BGRImageArray, IndexImageArray]:

    if isinstance(mosaic, Mapping):
        images = list(mosaic.values())
        ids = list(mosaic.keys())
    else:
        images = mosaic
        ids = list(range(len(mosaic)))

    if len(mosaic)==0:
        img = create_gap_image(size=min_size_xy, gap_colour=gap_color)
        put_text_in_corner(img, text='No Detections', color=BGRColors.WHITE)
        return img, np.full(shape=(min_size_xy[1], min_size_xy[0]), fill_value=-1)

    # First pack them into a big array
    image_array = put_list_of_images_in_array(images, fill_colour=gap_color, padding=0)
    id_array = np.zeros(image_array.shape[:3], dtype=int)
    id_array += np.array(ids)[:, None, None]

    image_grid = put_data_in_image_grid(image_array, fill_colour=gap_color, boundary_width=padding, min_size_xy=min_size_xy)
    id_grid = put_data_in_grid(id_array, fill_value=-1, min_size_xy=min_size_xy)

    return image_grid, id_grid

