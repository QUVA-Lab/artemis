

"""
Make a swap-in replacement for the decord video reader.


Why?  Because decord
- Hogs huge amounts of memory on larger videos
- Doesn't install with pyinstaller unless you do hacky stuff:
    https://github.com/dmlc/decord/issues/253
- Still gives errors on some users machines (Mac M1 air with older OS) even with the hacky stuff
    (Note we use eva-decord which ostensibly supports M1 https://pypi.org/project/eva-decord/)
"""
import itertools
from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional

import av
import os
from artemis.general.custom_types import BGRImageArray
import cv2

from artemis.image_processing.image_utils import fit_image_to_max_size
from artemis.image_processing.video_frame import VideoFrameInfo

class IDecorder(metaclass=ABCMeta):

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, item: int) -> BGRImageArray:
        ...




class PyAvDecorder(IDecorder):

    def __init__(self,
                 path: str,  # Path to the video
                 threshold_frames_to_scan: 30,  # Number of frames that we're willing to scan into the future before seeking
                 ):
        self._path = os.path.expanduser(path)
        assert os.path.exists(self._path), f"Cannot find a video at {path}"
        self._threshold_frames_to_scan = threshold_frames_to_scan
        self.container = av.container.open(self._path)
        # self.stream = self.container.streams.video[0]

        # self._cap = cv2.VideoCapture(path)
        # self._frame_cap.get(cv2.CAP_PROP_FPS)
        video_obj = self.container.streams.video[0]
        self._fps = float(video_obj.guessed_rate)

        self._n_frames = video_obj.frames
        # self._cached_last_frame: Optional[VideoFrameInfo] = None  # Helps fix weird bug... see notes below

    def __len__(self):
        return self._n_frames

    def _iter_frame_data(self):
        for frame in self.container.decode(self.container.streams.video[0]):
            yield frame

    def __getitem__(self, index: int) -> BGRImageArray:
        """
        Request a frame of the video.  If the requested frame is out of bounds, this will return the frame
        on the closest edge.
        """
        if self._is_destroyed:
            raise Exception("This object has been explicitly destroyed.")
        # print(f"Requesting frame {index}")
        n_frames = len(self)
        if index < 0:
            index = n_frames + index

        if 0 <= index - self._next_index_to_be_read < self._threshold_frames_to_scan:
            # Scan forward through current iterator until we hit frame
            frame_data = None
            for _ in range(self._next_index_to_be_read, index + 1):
                try:
                    frame_data = next(self._iterator)
                    self._next_index_to_be_read += 1
                except StopIteration:
                    raise Exception(f"Could not get frame at index {index}, despite n_frames being {n_frames}")
            assert frame_data is not None, "Did not read a frame - this should be impossible"
            image = frame_data.to_rgb().to_ndarray(format='bgr24')
            return image
        else:
            # Seek to this frame
            max_seek_search = 200  # I have no idea what's up with this.  100 failed some time
            stream = self.container.streams.video[0]
            pts = int(index * stream.duration / stream.frames)
            self.container.seek(pts, stream=stream)
            self._iterator = self._iter_frame_data()
            for j, f in enumerate(self._iterator):
                if j > max_seek_search:
                    raise RuntimeError(f'Did not find target frame {index} within {max_seek_search} frames of seek')
                if f.pts >= pts - 1:
                    self._iterator = itertools.chain([f], self._iterator)
                    break
            self._next_index_to_be_read = index
            return self.__getitem__(index)

    def destroy(self):
        self._iterator = None
        self._frame_cache = None
        self._is_destroyed = True
