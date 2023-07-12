
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
import os
from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, Iterator

import av
import cv2

from artemis.general.custom_types import BGRImageArray
from artemis.general.item_cache import CacheDict
from artemis.image_processing.image_utils import fit_image_to_max_size


class IDecorder(metaclass=ABCMeta):

    @abstractmethod
    def __len__(self) -> int:
        """ Get the number of frames in the video. """
        ...

    @abstractmethod
    def __getitem__(self, item: int) -> BGRImageArray:
        """ Lookup a frame by index.  Index should be in [-len(self), len(self)), otherwise an IndexError will be raised."""
        ...

    @abstractmethod
    def get_avg_fps(self) -> float:
        ...

    @abstractmethod
    def get_frame_timestamp(self, frame_index: int) -> float:
        """ Get the timestamp of a frame in seconds, relative to the start of the video. """
        ...

    def __iter__(self) -> Iterator[BGRImageArray]:
        """ Iterate over frames in the video (you can override this to be more efficient). """
        return iter(self[i] for i in range(len(self)))


class FrameListDecorder(IDecorder):

    def __init__(self,
                frames: Tuple[BGRImageArray, ...],
                fps: float,
                ):
        self._frames = frames
        self._fps = fps

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, item: int) -> BGRImageArray:
        return self._frames[item]

    def get_avg_fps(self) -> float:
        return self._fps

    def get_frame_timestamp(self, frame_index: int) -> float:
        return frame_index/self._fps


class PyAvDecorder(IDecorder):

    def __init__(self,
                 path: str,  # Path to the video
                 threshold_frames_to_scan: int = 30,  # Number of frames that we're willing to scan into the future before seeking
                 iter_with_opencv: bool = True,  # More efficient with same result
                 ):
        self._path = os.path.expanduser(path)
        assert os.path.exists(self._path), f"Cannot find a video at {path}"
        self._threshold_frames_to_scan = threshold_frames_to_scan
        self.container = av.container.open(self._path)
        # self.stream = self.container.streams.video[0]
        self._is_destroyed = False  # Forget why we want this... memory leak?
        # self._cap = cv2.VideoCapture(path)
        # self._frame_cap.get(cv2.CAP_PROP_FPS)
        video_obj = self.container.streams.video[0]
        self._fps = float(video_obj.guessed_rate)
        self._next_index_to_be_read = 0
        self._iter_with_opencv = iter_with_opencv

        self._n_frames = video_obj.frames
        self._iterator = self._iter_frame_data()
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
            if index < 0:
                raise IndexError(f"Index {index - n_frames} is out of bounds for video with {n_frames} frames")

        if 0 <= index - self._next_index_to_be_read < self._threshold_frames_to_scan:
            # Scan forward through current iterator until we hit frame
            frame_data = None
            for _ in range(self._next_index_to_be_read, index + 1):
                try:
                    frame_data = next(self._iterator)
                    self._next_index_to_be_read += 1
                except StopIteration:
                    if index < n_frames:
                        raise Exception(f"Could not get frame at index {index}, despite n_frames being {n_frames}")
                    else:
                        raise IndexError(f"Index {index} is out of bounds for video with {n_frames} frames")
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

    def get_avg_fps(self) -> float:
        return self._fps

    def get_frame_timestamp(self, frame_index: int) -> float:
        return frame_index / self._fps  # Todo: This is not quite right, but it's close enough for now

    def __iter__(self) -> Iterator[BGRImageArray]:
        if self._iter_with_opencv:
            cap = cv2.VideoCapture(self._path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
            cap.release()
        else:
            self.container.seek(0)
            for frame in self._iterator:
                yield frame.to_rgb().to_ndarray(format='bgr24')

    def destroy(self):
        self._iterator = None
        self._frame_cache = None
        self._is_destroyed = True

try:
    import decord
except:
    DecordDecorder = None
else:
    class DecordDecorder(decord.VideoReader, IDecorder):

        def __getitem__(self, item):
            return super().__getitem__(item).asnumpy()[:, :, ::-1]


class CachedDecorder(IDecorder):

    def __init__(self,
                decorder: IDecorder,
                 buffer_size_bytes=1024 ** 3,
                 max_size_xy: Optional[Tuple[int, int]] = None,
                 ):
        self._decorder = decorder
        self._frame_cache: CacheDict[int, BGRImageArray] = CacheDict(buffer_size_bytes=buffer_size_bytes, always_allow_one_item=True)
        self._max_size_xy = max_size_xy

    def __len__(self):
        return len(self._decorder)

    def __getitem__(self, index: int) -> BGRImageArray:
        if index in self._frame_cache:
            return self._frame_cache[index]
        else:
            image = self._decorder[index]
            if self._max_size_xy is not None:
                image = fit_image_to_max_size(image, self._max_size_xy)
            self._frame_cache[index] = image
            return image

    def get_avg_fps(self) -> float:
        return self._decorder.get_avg_fps()

    def get_frame_timestamp(self, frame_index: int) -> float:
        return self._decorder.get_frame_timestamp(frame_index)


def robustly_get_decorder(
        path: str,
        prefer_decord: bool = True,
        use_cache: bool = True,
        buffer_size_bytes: int = 1024 ** 3,
        max_size_xy: Optional[Tuple[int, int]] = None,

    ) -> IDecorder:
    """
    Get a decorder for a video.  If decord is installed, use that, otherwise use pyav.
    """
    if prefer_decord and DecordDecorder is not None:
        decorder = DecordDecorder(path)
    else:
        decorder = PyAvDecorder(path)

    if use_cache:
        decorder = CachedDecorder(decorder, buffer_size_bytes=buffer_size_bytes, max_size_xy=max_size_xy)

    return decorder
