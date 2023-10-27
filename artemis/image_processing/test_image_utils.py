from artemis.general.custom_types import BGRImageArray
from artemis.plotting.cv2_plotting import just_show
from artemis.plotting.easy_window import ImageRow
from eagle_eyes.datasets.videos import DroneVideos
from artemis.image_processing.image_utils import iter_images_from_video, mask_to_boxes, conditional_running_min, slice_image_with_pad, ImageViewInfo, load_artemis_image
import numpy as np
import os
from artemis.general.utils_for_testing import stringlist_to_mask
import cv2

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



def test_image_view_info(show: bool = False):

    image = load_artemis_image()

    pixel_xy_of_bottom_of_ear = (512, 228)

    # image = cv2.imread(cv2.samples.find_file("starry_night.jpg"))
    frame1 = ImageViewInfo.from_initial_view(window_disply_wh=(500, 500), image_wh=(image.shape[1], image.shape[0]))
    display_xy_of_bottom_of_ear = frame1.pixel_xy_to_display_xy(pixel_xy_of_bottom_of_ear)
    # recon_pixel_xy_of_bottom_of_ear = frame1.display_xy_to_pixel_xy(display_xy_of_bottom_of_ear, limit=False)
    frame2 = frame1.zoom_by(relative_zoom=1.5, invariant_display_xy=display_xy_of_bottom_of_ear, limit=False)
    frame3 = frame2.zoom_by(relative_zoom=1.5, invariant_display_xy=display_xy_of_bottom_of_ear, limit=False)
    frame4 = frame3.zoom_by(relative_zoom=1.5, invariant_display_xy=display_xy_of_bottom_of_ear, limit=False)
    frame5 = frame4.zoom_by(relative_zoom=1.5, invariant_display_xy=display_xy_of_bottom_of_ear, limit=False)
    f5_recon_pixel_xy_of_bottom_of_ear = frame5.display_xy_to_pixel_xy(display_xy_of_bottom_of_ear)
    assert tuple(np.round(f5_recon_pixel_xy_of_bottom_of_ear).astype(int)) == pixel_xy_of_bottom_of_ear
    # frame1 = frame.create_display_image(image)

    frame6 = frame3.pan_by_display_relshift(display_rel_xy=(0.5, 0), limit=True)
    frame7 = frame6.pan_by_display_relshift(display_rel_xy=(0.5, 0), limit=True)
    frame8 = frame7.pan_by_display_relshift(display_rel_xy=(0, .5), limit=True)
    frame9 = frame8.pan_by_display_relshift(display_rel_xy=(0, .5), limit=True)

    frame10 = frame9.zoom_by(relative_zoom=0.75, invariant_display_xy=(0, 0))
    frame11 = frame5.zoom_out()

    crops = [f.create_display_image(image) for f in [frame1, frame2, frame3, frame4, frame5, frame6, frame7, frame8, frame9, frame10, frame11]]
    # crops = [f.create_display_image(image) for f in [frame1, frame3]]

    # crop1 = frame.create_display_image(image)
    # crop2 = frame.zoom_by(relative_zoom=1.5, invariant_display_xy=(500, 250)).create_display_image(image)

    if show:
        display_img = ImageRow(image, *crops, wrap=6).render()
        just_show(display_img, hang_time=10)


from video_scanner.general_utils.file_utils import imread_any_path
from video_scanner.general_utils.utils_for_app_testing import DroneDataDirectory


def test_read_file_with_cyrillic_path():
    # Latin path
    path = DroneDataDirectory().get_file('test_data\casara_on\DJI_202305261131_040_Targets\DJI_20230526120735_0001_W.JPG')
    image = imread_any_path(path)
    assert image is not None
    assert image.shape == (3000, 4000, 3)

    # Path containing cyrillic
    path = DroneDataDirectory().get_file('test_data\\test_дir\dji_2023-07-06_18-09-26_0007.jpg')
    image = imread_any_path(path)
    assert image is not None
    assert image.shape == (3024, 4032, 3)


if __name__ == "__main__":
    # test_iter_images_from_video()
    # test_mask_to_boxes()
    # test_conditional_running_min()
    # test_slice_image_with_pad()
    # test_image_view_info(show=True)
    test_read_file_with_cyrillic_path()

