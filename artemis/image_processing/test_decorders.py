import cv2
import numpy as np
import pytest

from artemis.general.should_be_builtins import all_equal
from artemis.general.utils_for_testing import hold_tempdir
from video_scanner.general_utils.utils_for_app_testing import get_or_download_sample_video
from artemis.image_processing.decorders import PyAvDecorder, DecordDecorder, FrameListDecorder, robustly_get_decorder


def iter_frames_with_cv(path):
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame


def test_pyav_decorder():
    vid_path = get_or_download_sample_video()

    decorders = {
        "decord": DecordDecorder(vid_path),
        "cv": FrameListDecorder(tuple(iter_frames_with_cv(vid_path)), fps=30),
        "pyav": PyAvDecorder(vid_path),
        "cached_pyav": robustly_get_decorder(vid_path, prefer_decord=False, use_cache=True),
        "cached_decord": robustly_get_decorder(vid_path, prefer_decord=True, use_cache=True)
    }
    base_decorder = decorders["cv"]
    assert all_equal(167 == len(decorder) for decorder in decorders.values()), f"Different number of frames: {[len(decorder) for decorder in decorders.values()]}"
    frame_ixs_to_test = [5, 20, 70, 30, 0, -2, 155, 30]
    for frame_ix in frame_ixs_to_test:
        base_frame = base_decorder[frame_ix]
        for name, decorder in decorders.items():
            frame = decorder[frame_ix]
            is_equal_to_base = np.array_equal(base_frame, frame)
            is_close_to_base = abs(base_frame.astype(float)-frame.astype(float)).mean() < 1
            assert is_equal_to_base or is_close_to_base, f"Frame {frame_ix} is different between {name} and cv"
            print(f"Frame {frame_ix} from {name} is {'close' if is_close_to_base else 'equal'}")
        print(f"Frame {frame_ix} passed")

    for name, decorder in decorders.items():
        print(f"Testing {name} for index errors")
        with pytest.raises(IndexError):
            decorder[167]
        with pytest.raises(IndexError):
            decorder[999]
        with pytest.raises(IndexError):
            decorder[-168]


if __name__ == '__main__':
    test_pyav_decorder()
