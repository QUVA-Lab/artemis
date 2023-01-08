import os
from dataclasses import dataclass, replace
from typing import Optional, Tuple, Sequence, Iterator

import cv2

from artemis.general.custom_types import TimeIntervalTuple, BGRImageArray
from artemis.image_processing.image_utils import iter_images_from_video, fit_image_to_max_size
from artemis.image_processing.video_reader import VideoReader, VideoFrameInfo
from artemis.remote.utils import ARTEMIS_LOGGER


def parse_time_delta_str_to_sec(time_delta_str: str) -> Optional[float]:
    if time_delta_str in ('start', 'end'):
        return None
    else:
        start_splits = time_delta_str.split(':')
        if len(start_splits) == 1:
            return float(time_delta_str)
        elif len(start_splits) == 2:
            return 60 * float(start_splits[0]) + float(start_splits[1])
        elif len(start_splits) == 3:
            return 3600*float(start_splits[0]) + 60*float(start_splits[1]) + float(start_splits[2])
        else:
            raise Exception(f"Bad format: {time_delta_str}")


def parse_interval(interval_str: str) -> TimeIntervalTuple:

    start, end = (s.strip('') for s in interval_str.split('-'))
    return parse_time_delta_str_to_sec(start), parse_time_delta_str_to_sec(end)


@dataclass
class VideoSegment:
    """ """
    path: str
    time_interval: TimeIntervalTuple = None, None
    frame_interval: Tuple[Optional[int], Optional[int]] = (None, None)
    rotation: int = 0  # Number of Clockwise 90deg rotations to apply to the raw video
    keep_ratio: float = 1.
    use_scan_selection: bool = False
    max_size: Optional[Tuple[int, int]] = None
    frames_of_interest: Optional[Sequence[int]] = None
    verbose: bool = False

    def check_passthrough(self):
        assert os.path.exists(os.path.expanduser(self.path)), f"Path {self.path} does not exist."
        return self

    def iter_images(self, max_size: Optional[Tuple[int, int]] = None, max_count: Optional[int] = None) -> Iterator[BGRImageArray]:
        yield from iter_images_from_video(self.path, time_interval=self.time_interval, max_size=max_size or self.max_size, rotation=self.rotation, frame_interval=(None, max_count))

    def get_reader(self, buffer_size_bytes: int = 1024**3, use_cache: bool = True) -> VideoReader:
        return VideoReader(self.path, time_interval=self.time_interval, frame_interval=self.frame_interval,
                           buffer_size_bytes=buffer_size_bytes, use_cache=use_cache, max_size_xy=self.max_size)

    def recut(self, start_time: Optional[float] = None, end_time: Optional[float] = None):
        return replace(self, time_interval=(start_time, end_time), frame_interval=self.frame_interval)

    def iter_frame_info(self) -> Iterator[VideoFrameInfo]:
        """
        TODO: Replace with
            yield from self.get_reader().iter_frames()
        """

        assert not self.use_scan_selection, "This does not work.  See bug: https://github.com/opencv/opencv/issues/9053"
        path = os.path.expanduser(self.path)
        cap = cv2.VideoCapture(path)
        start_frame, stop_frame = self.frame_interval
        start_time, end_time = self.time_interval
        if self.max_size is not None:  # Set cap size.  Sometimes this does not work so we also have the code below.
            sx, sy = self.max_size if self.rotation in (0, 2) else self.max_size[::-1]
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

        iter_frames_of_interest = iter(self.frames_of_interest) if self.frames_of_interest is not None else None

        initial_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        while cap.isOpened():

            if iter_frames_of_interest is not None and self.use_scan_selection:
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
            if self.max_size is not None:
                image = fit_image_to_max_size(image, self.max_size)
            if self.keep_ratio != 1:
                if self.verbose:
                    print(f'Real surplus: {frame_ix - self.keep_ratio * unique_frame_ix}')
                frame_surplus = round(frame_ix - self.keep_ratio * unique_frame_ix)
                if frame_surplus < 0:  # Frame debt - yield this one twice
                    if self.verbose:
                        print('Yielding extra frame due to frame debt')
                    yield VideoFrameInfo(image, seconds_into_video=(initial_frame + unique_frame_ix) / fps, frame_ix=initial_frame + unique_frame_ix, fps=fps)
                    frame_ix += 1
                elif frame_surplus > 0:  # Frame surplus - skip it
                    if self.verbose:
                        print('Skipping frame due to frame surplus')
                    continue

            if iter_frames_of_interest is None or self.use_scan_selection or (not self.use_scan_selection and frame_ix in self.frames_of_interest):
                if self.rotation != 0:
                    image = cv2.rotate(image, rotateCode={1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}[self.rotation])
                else:
                    image = image
                yield VideoFrameInfo(image, seconds_into_video=(initial_frame + unique_frame_ix) / fps, frame_ix=initial_frame + unique_frame_ix, fps=fps)
            frame_ix += 1
