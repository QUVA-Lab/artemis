import datetime
import itertools
import os
import threading
import time
from _py_abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, Iterator, Sequence, Callable
import av
import cv2
from more_itertools import first

import exif
import numpy as np
from artemis.general.custom_types import TimeIntervalTuple
from artemis.general.should_be_builtins import seconds_to_time_marker
from artemis.general.utils_utils import byte_size_to_string
from artemis.image_processing.image_utils import fit_image_to_max_size, read_image_time_or_none
from artemis.general.item_cache import CacheDict
from artemis.general.parsing import parse_time_delta_str_to_sec
from artemis.image_processing.livestream_recorder import LiveStreamRecorderAgent
from artemis.image_processing.video_frame import VideoFrameInfo, FrameGeoData
from artemis.image_processing.media_metadata import read_image_geodata_or_none


@dataclass
class VideoMetaData:
    duration: float
    n_frames: int
    fps: float
    n_bytes: int
    size_xy: Tuple[int, int]

    @classmethod
    def from_null(cls):
        return cls(duration=float('inf'), n_frames=-1, fps=float('inf'), n_bytes=-1, size_xy=(0, 0))

    def get_duration_string(self) -> str:
        return str(datetime.timedelta(seconds=int(self.duration))) if self.duration != float('inf') else '∞'

    def get_size_xy_string(self) -> str:
        return f"{self.size_xy[0]}x{self.size_xy[1]}"

    def get_size_string(self) -> str:
        return byte_size_to_string(self.n_bytes, decimals_precision=1)


def get_actual_frame_interval(
        n_frames_total: int,
        fps: float,
        time_interval: TimeIntervalTuple = (None, None),
        frame_interval: Tuple[Optional[int], Optional[int]] = (None, None),
) -> Tuple[int, int]:
    assert time_interval == (None, None) or frame_interval == (
        None, None), "You can provide a time interval or frame inteval, not both"
    if time_interval != (None, None):
        tstart, tstop = time_interval
        return get_actual_frame_interval(n_frames_total=n_frames_total, fps=fps,
                                         frame_interval=(round(tstart * fps), round(tstop * fps)))

        # return self.time_to_nearest_frame(tstart) if tstart is not None else 0, self.time_to_nearest_frame(
        #     tstop) + 1 if tstop is not None else self.get_n_frames()
    elif frame_interval != (None, None):
        istart, istop = frame_interval
        istart = 0 if istart is None else n_frames_total + istart if istart < 0 else istart
        istop = n_frames_total if istop is None else n_frames_total + istop if istop < 0 else istop
        return istart, istop
    else:
        return 0, n_frames_total


@dataclass
class IVideoReader(metaclass=ABCMeta):
    """
    Interface for something behaving like a video reader.
    """

    @abstractmethod
    def get_metadata(self) -> VideoMetaData:
        """ Get data showing number of frames, size, etc. """

    @abstractmethod
    def get_n_frames(self) -> int:
        """ Get the number of frames in the video """

    @abstractmethod
    def time_to_nearest_frame(self, t: float) -> int:
        """ Get the frame index nearest the time t """

    @abstractmethod
    def frame_index_to_nearest_frame(self, index: int) -> int:
        """ Get the frame index nearest the time t """

    @abstractmethod
    def frame_index_to_time(self, frame_ix: int) -> float:
        """ Get the time corresponding to the frame index """

    @abstractmethod
    def get_progress_indicator(self, frame_ix) -> str:
        """ Return a human-readable string that indicates progress at this frame. """

    @abstractmethod
    def time_indicator_to_nearest_frame(self, time_indicator: str) -> Optional[int]:
        """ Get the frame index nearest the time-indicator
        e.g. "0:32.5" "32.5s", "53%", "975" (frame number)
        Returns None if the time_indicator is invalid
        Note - frame-index in string is 1-based, output is 0-based, so "975" -> 974
        """

    @abstractmethod
    def iter_frame_ixs(self) -> Iterator[int]:
        """ Iterate through frame indices """

    @abstractmethod
    def iter_frames(self) -> Iterator[VideoFrameInfo]:
        """ Iterate through the frames of the video """

    @abstractmethod
    def cut(self, time_interval: TimeIntervalTuple = (None, None),
            frame_interval: Tuple[Optional[int], Optional[int]] = (None, None)) -> 'IVideoReader':
        """ Cut the video to the given time interval """

    @abstractmethod
    def request_frame(self, index: int) -> VideoFrameInfo:
        """
        Request a frame of the video.  If the requested frame is out of bounds, this will return the frame
        on the closest edge.
        """

    @abstractmethod
    def destroy(self):
        """ Destroy the video reader (prevents memory leaks) """

    def is_live(self) -> bool:
        """ Return True if this this can grow in size as it is read.  False if it is a fixed size. """
        return False


def time_indicator_to_nearest_frame(time_indicator: str, n_frames: int, fps: Optional[float] = None, frame_times: Optional[Sequence[float]] = None) -> Optional[int]:
    """ Get the frame index nearest the time-indicator
    e.g. "0:32.5" "32.5s", "53%", "975" (frame number)
    Returns None if the time_indicator is invalid
    """
    assert (fps is None) or (frame_times is None), "You must provide either fps or frame_times.  Not both."

    def lookup_frame_ix(t: float) -> Optional[int]:
        if frame_times is None:
            return round(t * fps)
        elif fps is not None:
            return np.searchsorted(frame_times, t, side='left')
        else:
            return None

    if time_indicator in ('s', 'start'):
        return 0
    elif time_indicator in ('e', 'end'):
        return n_frames - 1
    elif ':' in time_indicator:
        sec = parse_time_delta_str_to_sec(time_indicator)
        return lookup_frame_ix(sec)
    elif time_indicator.endswith('s'):
        sec = float(time_indicator.rstrip('s'))
        return lookup_frame_ix(sec)
    elif time_indicator.endswith('%'):
        percent = float(time_indicator.rstrip('%'))
        return round(percent / 100 * n_frames)
    elif time_indicator.isdigit():
        return max(0, int(time_indicator)-1)
    else:
        return None


class VideoReader(IVideoReader):
    """
    The reader efficiently provides access to video frames.
    It uses pyav: https://pyav.org/docs/stable/

    Usage:
        reader = VideoReader(path=video_segment.path, use_cache=use_cache)
        # Iterate in order
        for frame in reader.iter_frames(time_interval=(1, 2)):
            cv2.imshow('frame', frame.image)
            cv2.waitKey(1)
        # Request individually
        frame = reader.request_frame(20)  # Ask for the 20th frame
        cv2.imshow('frame', frame.image)
        cv2.waitKey(1)

    Implementation:
    - Providing them in order should be fast
    - Requesting the same frame twice or backtracking a few frames should be VERY FAST (ie - use a cache)
    - Requesting random frames should be reasonably fast (do not scan from start)

    Note: Due to bug in OpenCV with GET_PROP_POS_FRAMES
        https://github.com/opencv/opencv/issues/9053
        We use "av": conda install av -c conda-forge
    """
    # TODO: Remove, and replace with something that uses an IDecorder instead

    def __init__(self,
                 path: str,
                 time_interval: TimeIntervalTuple = (None, None),
                 frame_interval: Tuple[Optional[int], Optional[int]] = (None, None),
                 buffer_size_bytes=1024 ** 3,
                 threshold_frames_to_scan=30,
                 max_size_xy: Optional[Tuple[int, int]] = None,
                 use_cache: bool = True):

        self._path = os.path.expanduser(path)
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"Cannot find a video at {path}")

        self.container = av.container.open(self._path)
        # self.stream = self.container.streams.video[0]

        # self._cap = cv2.VideoCapture(path)
        self._frame_cache: CacheDict[int, VideoFrameInfo] = CacheDict(buffer_size_bytes=buffer_size_bytes,
                                                                      always_allow_one_item=True)
        self._next_index_to_be_read: int = 0
        self._threshold_frames_to_scan = threshold_frames_to_scan
        # self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(self.container.streams.video[0].guessed_rate)

        self._n_frames = self.container.streams.video[0].frames
        # self._cached_last_frame: Optional[VideoFrameInfo] = None  # Helps fix weird bug... see notes below
        self._max_size_xy = max_size_xy
        self._use_cache = use_cache
        self._iterator = self._iter_frame_data()
        self._start, self._stop = get_actual_frame_interval(time_interval=time_interval,
                                                            fps=self._fps,
                                                            frame_interval=frame_interval,
                                                            n_frames_total=self.container.streams.video[0].frames)
        self._metadata: Optional[VideoMetaData] = None
        self._is_destroyed = False
        # Call a function to notify us when this object gets destroyed
        # def is_des

        # self._n_frames =

    def get_metadata(self) -> VideoMetaData:
        file_stats = os.stat(self._path)

        try:
            width = self.container.streams.video[0].codec_context.width
            height = self.container.streams.video[0].codec_context.height
        except Exception as err:
            print("Error getting width and height from video stream. Using first frame instead. Error: ", err)
            firstframe = self.request_frame(0)
            width, height = firstframe.image.shape[1], firstframe.image.shape[0]

        if self._metadata is None:
            self._metadata = VideoMetaData(
                duration=self._n_frames/self._fps,
                n_frames=max(1, self._n_frames),  # It seems to give 0 for images which aint right
                fps=self._fps,
                n_bytes=file_stats.st_size,
                size_xy=(width, height)
            )
        return self._metadata

    def get_progress_indicator(self, frame_ix, just_seconds: bool = False) -> str:
        seconds_into_video = frame_ix / self._fps if self._fps else 0
        seconds_str = f"{seconds_into_video:.2f}s" if just_seconds else seconds_to_time_marker(seconds_into_video)
        total_frames = self.get_n_frames()
        if total_frames is None:
            return f"t={seconds_str}, frame={frame_ix+1}"
        else:
            total_seconds = total_frames / self._fps if self._fps else 0
            total_seconds_str = f"{total_seconds:.2f}s" if just_seconds else seconds_to_time_marker(total_seconds)
            return f"t={seconds_str}/{total_seconds_str}, frame={frame_ix+1}/{total_frames}"

    def get_n_frames(self) -> int:
        return self._stop - self._start

    def time_to_nearest_frame(self, t: float) -> int:
        return max(0, min(self.get_n_frames() - 1, round(t * self._fps)))

    def frame_index_to_nearest_frame(self, index: int) -> int:
        return max(0, min(self.get_n_frames() - 1, index))

    def frame_index_to_time(self, frame_ix: int) -> float:
        return frame_ix / self._fps

    def time_indicator_to_nearest_frame(self, time_indicator: str) -> Optional[int]:
        """ Get the frame index nearest the time-indicator
        e.g. "0:32.5" "32.5s", "53%", "975" (frame number)
        Returns None if the time_indicator is invalid
        """
        return time_indicator_to_nearest_frame(time_indicator, n_frames=self.get_n_frames(), fps=self._fps)

    def iter_frame_ixs(self) -> Iterator[int]:
        start, stop = get_actual_frame_interval(fps=self._fps, n_frames_total=self.get_n_frames())
        """ 
        TODO: Get rid of arguments - just use cut instead 
        """
        return range(start, stop)

    def iter_frames(self) -> Iterator[VideoFrameInfo]:
        """
        TODO: Get rid of arguments - just use cut instead
        """
        for i in self.iter_frame_ixs():
            yield self.request_frame(i)

    def cut(self, time_interval: TimeIntervalTuple = (None, None),
            frame_interval: Tuple[Optional[int], Optional[int]] = (None, None)) -> 'VideoReader':
        fstart, fstop = get_actual_frame_interval(time_interval=time_interval, frame_interval=frame_interval,
                                                   n_frames_total=self.get_n_frames(), fps=self._fps)
        newstart = self._start + fstart
        return VideoReader(path=self._path, frame_interval=(newstart, newstart + (fstop - fstart)),
                           threshold_frames_to_scan=self._threshold_frames_to_scan,
                           buffer_size_bytes=self._frame_cache.buffer_size_bytes)

    def _iter_frame_data(self):
        for frame in self.container.decode(self.container.streams.video[0]):
            yield frame

    def request_frame(self, index: int) -> VideoFrameInfo:
        """
        Request a frame of the video.  If the requested frame is out of bounds, this will return the frame
        on the closest edge.
        """
        if self._is_destroyed:
            raise Exception("This object has been explicitly destroyed.")
        # print(f"Requesting frame {index}")
        if index < 0:
            index = self.get_n_frames() + index
        index = max(0, min(self.get_n_frames() - 1, index))
        index_in_file = index + self._start

        # if index == self.get_n_frames() - 1 and self._cached_last_frame is not None:
        #     return self._cached_last_frame  # There's a weird bug preventing us from loading the last frame again
        if index_in_file in self._frame_cache:
            return self._frame_cache[index_in_file]
        elif 0 <= index_in_file - self._next_index_to_be_read < self._threshold_frames_to_scan:
            frame = None
            for _ in range(self._next_index_to_be_read, index_in_file + 1):
                try:
                    frame_data = next(self._iterator)
                except StopIteration:
                    raise Exception(
                        f"Could not get frame at index {index_in_file}, despite n_frames being {self.get_n_frames()}")

                image = frame_data.to_rgb().to_ndarray(format='bgr24')

                if self._max_size_xy is not None:
                    image = fit_image_to_max_size(image, self._max_size_xy)
                frame = VideoFrameInfo(
                    image=image,
                    seconds_into_video=self._next_index_to_be_read / self._fps,
                    frame_ix=self._next_index_to_be_read,
                    fps=self._fps
                )
                if self._use_cache:
                    self._frame_cache[frame.frame_ix] = frame
                self._next_index_to_be_read += 1

            assert frame is not None, f"Error loading video frame at index {index_in_file}"
            return frame
        else:
            max_seek_search = 200  # I have no idea what's up with this.  100 failed some time
            stream = self.container.streams.video[0]
            pts = int(index_in_file * stream.duration / stream.frames)
            self.container.seek(pts, stream=stream)
            self._iterator = self._iter_frame_data()
            for j, f in enumerate(self._iterator):
                if j > max_seek_search:
                    raise RuntimeError(f'Did not find target frame {index} within {max_seek_search} frames of seek')
                if f.pts >= pts - 1:
                    self._iterator = itertools.chain([f], self._iterator)
                    break
            self._next_index_to_be_read = index_in_file
            return self.request_frame(index)

    def destroy(self):
        self._iterator = None
        self._frame_cache = None
        self._is_destroyed = True


def get_time_ordered_image_paths(image_paths: Sequence[str], fallback_fps: float = 1.
                                 ) -> Tuple[Sequence[str], Sequence[float]]:
    image_times = [read_image_time_or_none(path) for path in image_paths]
    if any(t is None for t in image_times):
        image_times = [i / fallback_fps for i in range(len(image_paths))]
    ixs = np.argsort(image_times)
    return [image_paths[ix] for ix in ixs], image_times[ixs]


@dataclass
class ImageSequenceReader(IVideoReader):
    """ Reads through a seqence of images as if they were a video."""

    def __init__(self,
                 image_paths: Sequence[str],
                 fallback_fps: float = 1.,
                 reorder = False,
                 new_file_checker: Optional[Callable[[], Sequence[str]]] = None,
                 cache_size: int = 1,
                 geodata_reader: Callable[[str], FrameGeoData] = read_image_geodata_or_none
                 ):
        self._image_paths = image_paths
        if reorder:
            image_paths, self._image_times = get_time_ordered_image_paths(image_paths, fallback_fps)
            self._image_paths = list(image_paths)
        else:
            self._image_times = [i / fallback_fps for i in range(len(image_paths))]
        self._new_file_checker = new_file_checker
        self._fallback_fps = fallback_fps
        self._cache = CacheDict(buffer_length=cache_size)
        self._lock = threading.Lock()
        self._geodata_cache = {}
        self._geodata_reader = geodata_reader

    def check_and_add_new_files(self) -> int:
        with self._lock:
            if self._new_file_checker is None:
                return 0
            new_files = self._new_file_checker()
            for new_file in new_files:
                if new_file not in self._image_paths:
                    self._image_paths.append(new_file)
                    self._image_times.append(read_image_time_or_none(new_file) or (len(self._image_paths)-1) / self._fallback_fps)
            return len(new_files)

    def get_sorted_paths(self) -> Sequence[str]:
        return self._image_paths

    def get_metadata(self) -> VideoMetaData:
        try:
            image_meta = exif.Image(self._image_paths[0])
            if hasattr(image_meta, 'pixel_x_dimension'):
                size_xy = image_meta.pixel_x_dimension, image_meta.pixel_y_dimension
            elif hasattr(image_meta, 'image_width'):
                size_xy = image_meta.image_width, image_meta.image_height
            else:
                raise Exception("Cant read image size")
        except:  # Load it and find out
            if self._image_paths:
                img = cv2.imread(self._image_paths[0])
                size_xy = (img.shape[1], img.shape[0]) if img is not None else (-1, -1)
            else:
                size_xy = (-1, -1)

        duration = self._image_times[-1] - self._image_times[0] if self._image_times else 0.
        return VideoMetaData(
            duration=self._image_times[-1] - self._image_times[0] if self._image_times else 0.,
            n_frames=len(self._image_paths),
            fps=len(self._image_paths) /duration if duration > 0 else 0.,
            size_xy=size_xy,
            n_bytes=sum(os.path.getsize(path) if path else 0 for path in self._image_paths)
        )

    def get_image_paths(self) -> Sequence[str]:
        return self._image_paths

    def get_progress_indicator(self, frame_ix) -> str:
        if self._image_paths:
            return f"Frame {frame_ix+1}/{self.get_n_frames()}: {os.path.split(self._image_paths[frame_ix])[-1]}"
        else:
            return "No frames loaded yet"

    def get_n_frames(self) -> int:
        return len(self._image_paths)

    def time_to_nearest_frame(self, t: float) -> int:
        return np.searchsorted(self._image_times, t, side='left')

    def frame_index_to_nearest_frame(self, index: int) -> int:
        return min(max(0, index), self.get_n_frames() - 1)

    def frame_index_to_time(self, frame_ix: int) -> float:
        return self._image_times[frame_ix]

    def time_indicator_to_nearest_frame(self, time_indicator: str) -> Optional[int]:
        return time_indicator_to_nearest_frame(time_indicator, n_frames=self.get_n_frames(), frame_times=self._image_times)

    def iter_frame_ixs(self) -> Iterator[int]:
        return range(self.get_n_frames())

    def iter_frames(self) -> Iterator[VideoFrameInfo]:
        if not self.is_live():
            for i in self.iter_frame_ixs():
                yield self.request_frame(i)
        else:
            i = 0
            while True:
                self.check_and_add_new_files()
                for j in range(i, self.get_n_frames()):
                    yield self.request_frame(j)
                if not self.is_live():
                    break
                i = self.get_n_frames()
                time.sleep(0.1)

    def cut(self, time_interval: TimeIntervalTuple = (None, None), frame_interval: Tuple[Optional[int], Optional[int]] = (None, None)) -> 'ImageSequenceReader':
        if time_interval[0] is not None:
            frame_interval = (self.time_to_nearest_frame(time_interval[0]), frame_interval[1])
        if time_interval[1] is not None:
            frame_interval = (frame_interval[0], self.time_to_nearest_frame(time_interval[1]))
        return ImageSequenceReader(self._image_paths[frame_interval[0]:frame_interval[1]])

    def request_frame(self, index: int) -> VideoFrameInfo:
        with self._lock:
            # image = cv2.imread(self._image_paths[index]) if index not in self._cache else self._cache[index]
            if index in self._cache:
                image = self._cache[index]
            else:
                if index >= len(self._image_paths):
                    index = len(self._image_paths) - 1
                if not self._image_paths[index]:
                    index = first((i for i in range(index, -1, -1) if self._image_paths[i]), default=0)
                if not os.path.exists(self._image_paths[index]):
                    raise FileNotFoundError(f"Could not find image at path: '{self._image_paths[index]}'")
                image = cv2.imread(self._image_paths[index])
                self._cache[index] = image
            assert image is not None, f"Could not load image at path: '{self._image_paths[index]}'"
            return VideoFrameInfo(
                image=image,
                seconds_into_video=self._image_times[index],
                frame_ix=index,
                fps=self.get_metadata().fps
            )

    def read_frame_geodata_or_none(self, frame_ix: int) -> FrameGeoData:
        if frame_ix not in self._geodata_cache:
            self._geodata_cache[frame_ix] = self._geodata_reader(self._image_paths[frame_ix])
        return self._geodata_cache[frame_ix]

    def stop_live(self):
        self._new_file_checker = None

    def destroy(self):
        pass

    def is_live(self) -> bool:
        return self._new_file_checker is not None


@dataclass
class LiveVideoReader(IVideoReader):

    cap: cv2.VideoCapture
    frames_seen_so_far: int = 0
    record: bool = True
    _iterator: Optional[Iterator[VideoFrameInfo]] = None
    _last_frame: Optional[VideoFrameInfo] = None

    @classmethod
    def get_metadata(self) -> VideoMetaData:
        return VideoMetaData(
            duration=np.inf,
            n_frames=0,
            fps=np.inf,
            size_xy=(-1, -1),
            n_bytes=-1
        )

    def get_n_frames(self) -> int:
        return 1

    def time_to_nearest_frame(self, t: float) -> int:
        return 0

    def frame_index_to_nearest_frame(self, index: int) -> int:
        return 0

    def frame_index_to_time(self, frame_ix: int) -> float:
        return 0.

    def time_indicator_to_nearest_frame(self, time_indicator: str) -> Optional[int]:
        return 0

    def iter_frame_ixs(self) -> Iterator[int]:
        yield 0

    def iter_frames(self) -> Iterator[VideoFrameInfo]:
        # cap = cv2.VideoCapture(self.stream_url)
        t_start = time.monotonic()
        count = 0
        while True:
            count += 1
            ret, frame = self.cap.read()
            if not ret:
                break
            elapsed = time.monotonic() - t_start
            yield VideoFrameInfo(
                image=frame,
                seconds_into_video=elapsed,
                frame_ix=self.frames_seen_so_far,
                fps=self.get_metadata().fps
            )
            self.frames_seen_so_far += 1

    def cut(self, time_interval: TimeIntervalTuple = (None, None), frame_interval: Tuple[Optional[int], Optional[int]] = (None, None)) -> 'IVideoReader':
        raise NotImplementedError("Can't cut a live stream")

    def request_frame(self, index: int) -> VideoFrameInfo:
        if self._iterator is None:
            self._iterator = self.iter_frames()
        return next(self._iterator)
        # assert self._last_frame is not None, "Can't request a frame from a live stream without iterating through it"
        # return self._last_frame

    def destroy(self):
        pass


@dataclass
class LiveRecordingVideoReader(IVideoReader):

    agent: LiveStreamRecorderAgent

    # stream_url: str
    # record_url: str
    _frames_seen_so_far: int = 0
    _recorded_video_reader: Optional[VideoReader] = None
    # _iterator: Optional[Iterator[VideoFrameInfo]] = None

    # _last_frame: Optional[VideoFrameInfo] = None

    def _get_recorded_video_reader(self) -> VideoReader:
        if self._recorded_video_reader is None:
            self._recorded_video_reader = VideoReader(self.agent.writing_video_path)
        return self._recorded_video_reader

    def get_metadata(self) -> VideoMetaData:
        return VideoMetaData(
            duration=np.inf,
            n_frames=self._frames_seen_so_far,
            fps=np.inf,
            size_xy=(-1, -1),
            n_bytes=-1
        )

    def get_n_frames(self) -> int:
        return self._frames_seen_so_far

    def time_to_nearest_frame(self, t: float) -> int:
        return 0

    def frame_index_to_nearest_frame(self, index: int) -> int:
        return 0

    def frame_index_to_time(self, frame_ix: int) -> float:
        return 0.

    def time_indicator_to_nearest_frame(self, time_indicator: str) -> Optional[int]:
        return 0

    def iter_frame_ixs(self) -> Iterator[int]:
        yield 0

    def iter_frames(self) -> Iterator[VideoFrameInfo]:
        raise NotImplementedError()
        # cap = cv2.VideoCapture(self.stream_url)
        # t_start = time.monotonic()
        # count = 0
        # while True:
        #     count += 1
        #     ret, frame = cap.read()
        #     if not ret:
        #         break
        #     elapsed = time.monotonic() - t_start
        #     yield VideoFrameInfo(
        #         image=frame,
        #         seconds_into_video=elapsed,
        #         frame_ix=self._frames_seen_so_far,
        #         fps=self.get_metadata().fps
        #     )
        #     self._frames_seen_so_far += 1

    def cut(self, time_interval: TimeIntervalTuple = (None, None), frame_interval: Tuple[Optional[int], Optional[int]] = (None, None)) -> 'IVideoReader':
        raise NotImplementedError("Can't cut a live stream")

    def request_frame(self, index: int) -> VideoFrameInfo:
        print("Requesting frame ", index)
        if index==-1 or index==self._frames_seen_so_far:
            frame = self.agent.get_last_frame_blocking()
            self._frames_seen_so_far = frame.frame_ix + 1
            print("Requesting last frame from live stream")
            return frame
        else:
            print("Requesting frame from recorded video")
            return self._get_recorded_video_reader().request_frame(index)

    def destroy(self):
        pass


