from typing import Union, Mapping, Sequence, Tuple, Optional

import numpy as np

from artemis.general.custom_types import BGRImageArray, IndexImageArray, BGRColorTuple
from artemis.image_processing.image_utils import DEFAULT_GAP_COLOR, create_gap_image, BGRColors
from artemis.plotting.data_conversion import put_list_of_images_in_array, put_data_in_image_grid, put_data_in_grid
from artemis.plotting.easy_window import put_text_in_corner


def generate_image_mosaic_and_index_grid(
        mosaic: Union[Mapping[int, BGRImageArray], Sequence[BGRImageArray]],
        gap_color: BGRColorTuple = DEFAULT_GAP_COLOR,
        desired_aspect_ratio = 1.,  # TODO: Make it work
        grid_shape: Tuple[Optional[int], Optional[int]] = (None, None),  # (rows, columns)
        min_size_xy: Tuple[int, int] = (640, 480),
        padding: int = 1,
        end_text: Optional[str] = None,
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
        image_grid, id_grid = img, np.full(shape=(min_size_xy[1], min_size_xy[0]), fill_value=-1)
    else:
        # First pack them into a big array
        image_array = put_list_of_images_in_array(images, fill_colour=gap_color, padding=0)
        id_array = np.zeros(image_array.shape[:3], dtype=int)
        id_array += np.array(ids)[:, None, None]

        image_grid = put_data_in_image_grid(image_array, grid_shape=grid_shape, fill_colour=gap_color, boundary_width=padding, min_size_xy=min_size_xy)
        id_grid = put_data_in_grid(id_array, grid_shape=grid_shape, fill_value=-1, min_size_xy=min_size_xy)

    if end_text is not None:
        n_added_pixels = 20
        image_grid = np.pad(image_grid, ((0, n_added_pixels), (0, 0), (0, 0)), mode='constant', constant_values=0)
        put_text_in_corner(image_grid, text=end_text, color=BGRColors.WHITE, corner='bc')
        id_grid = np.pad(id_grid, ((0, n_added_pixels), (0, 0)), mode='constant', constant_values=-1)

    return image_grid, id_grid

