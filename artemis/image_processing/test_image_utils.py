from eagle_eyes.datasets.videos import DroneVideos
from artemis.image_processing.image_utils import iter_images_from_video, mask_to_boxes, conditional_running_min, slice_image_with_pad
import numpy as np

from artemis.general.utils_for_testing import stringlist_to_mask


def test_iter_images_from_video():

    video_path = DroneVideos.WALK_DAY_FIRST_SHORT.path

    frames_of_interest = [3, 5, 9]
    images_1 = [im for i, im in zip(range(frames_of_interest[-1]+1), iter_images_from_video(path=video_path)) if i in frames_of_interest]
    images_2 = list(iter_images_from_video(path=video_path, frames_of_interest=frames_of_interest))
    assert np.array_equal(images_1, images_2)

    time_interval = (3.0, 10.5)
    images_1 = [im for i, im in zip(range(frames_of_interest[-1]+1), iter_images_from_video(path=video_path, time_interval=time_interval)) if i in frames_of_interest]
    images_2 = list(iter_images_from_video(path=video_path, frames_of_interest=frames_of_interest, time_interval=time_interval))
    assert np.array_equal(images_1, images_2)


def test_conditional_running_min():
    vals = np.array([
        [0, 1, 2, 3],
        [7, 6, 5, 4],
        [8, 9, 10, 11],
        [11, 12, 13, 14],
    ])
    mask = np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 1],
        [0, 0, 0, 0]
    ], dtype=bool)
    result = conditional_running_min(vals, mask, default=-1, axis=1)
    assert np.array_equal(result, [
        [-1, 1, 1, -1],
        [7, -1, -1, 4],
        [-1, 9, 9, 9],
        [-1, -1, -1, -1],
    ])



def test_mask_to_boxes():

    image = stringlist_to_mask(
        " X   ",
        "XXXX ",
        " X   ",
        " X   ",
        "     "
    )
    boxes = mask_to_boxes(image)
    assert set(tuple(bb) for bb in boxes) == {
        (0, 0, 4, 4),
    }

    image = stringlist_to_mask(
        "       XXX  ",
        " XX         ",
        " X    X     ",
        "      XXXX  ",
        "      XX    "
    )
    boxes = mask_to_boxes(image)
    assert set(tuple(bb) for bb in boxes) == {
        (1, 1, 3, 3),
        (7, 0, 10, 1),
        (6, 2, 10, 5)
    }


    image = stringlist_to_mask(
        "     ",
        "     ",
        "     ",
        "     ",
        "     "
    )
    boxes = mask_to_boxes(image)
    assert set(tuple(bb) for bb in boxes) == set()


def test_slice_image_with_pad():
    vals = np.array([
        [0, 1, 2, 3],
        [7, 6, 5, 4],
        [8, 9, 10, 11],
        [11, 12, 13, 14],
    ])
    result = slice_image_with_pad(image=vals, xxyy_box=(-1, 3, -1, 2), gap_color=-1)
    assert np.array_equal(result, [
        [-1, -1, -1, -1],
        [-1, 0, 1, 2],
        [-1, 7, 6, 5],
    ])
    result = slice_image_with_pad(image=vals, xxyy_box=(1, 3, 1, 2), gap_color=-1)
    assert np.array_equal(result, [
        [6, 5],
    ])
    result = slice_image_with_pad(image=vals, xxyy_box=(2, 6, 3, 5), gap_color=-1)
    assert np.array_equal(result, [
        [13, 14, -1, -1],
        [-1, -1, -1, -1],
    ])
    arr = np.random.rand(4, 4, 3)
    result = slice_image_with_pad(image=arr, xxyy_box=(2, 6, 3, 5), gap_color=(-1., -1., -1.))
    assert np.array_equal(result[:1, :2], arr[3:, 2:])



if __name__ == "__main__":
    # test_iter_images_from_video()
    # test_mask_to_boxes()
    # test_conditional_running_min()
    test_slice_image_with_pad()


