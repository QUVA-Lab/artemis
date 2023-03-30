import datetime
import itertools
import os
from dataclasses import dataclass
from typing import Tuple, Optional, Iterator
import av

from artemis.general.custom_types import BGRImageArray, TimeIntervalTuple
from artemis.general.utils_utils import bytes_to_string
from artemis.image_processing.image_utils import fit_image_to_max_size
from artemis.general.item_cache import CacheDict
from artemis.general.parsing import parse_time_delta_str_to_sec


@dataclass
class VideoFrameInfo:
    image: BGRImageArray
    seconds_into_video: float
    frame_ix: int
    fps: float

    def get_size_xy(self) -> Tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]

    def get_progress_string(self, total_frames: Optional[int] = None) -> str:
        if total_frames is None:
            return f"t={self.seconds_into_video:.2f}s, frame={self.frame_ix}"
        else:
            return f"t={self.seconds_into_video:.2f}s/{total_frames/self.fps:.2f}s, frame={self.frame_ix}/{total_frames}"

@dataclass
class VideoMetaData:
    duration: float
    n_frames: int
    fps: float
    n_bytes: int
    size_xy: Tuple[int, int]

    def get_duration_string(self) -> str:
        return str(datetime.timedelta(seconds=int(self.duration)))

    def get_size_xy_string(self) -> str:
        return f"{self.size_xy[0]}x{self.size_xy[1]}"

    def get_size_string(self) -> str:
        return bytes_to_string(self.n_bytes, decimals_precision=1)

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
class VideoReader:
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

    def __init__(self,
                 path: str,
                 time_interval: TimeIntervalTuple = (None, None),
                 frame_interval: Tuple[Optional[int], Optional[int]] = (None, None),
                 buffer_size_bytes=1024 ** 3,
                 threshold_frames_to_scan=30,
                 max_size_xy: Optional[Tuple[int, int]] = None,
                 use_cache: bool = True):

        self._path = os.path.expanduser(path)
        assert os.path.exists(self._path), f"Cannot find a video at {path}"

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
        if time_indicator in ('s', 'start'):
            return 0
        elif time_indicator in ('e', 'end'):
            return self.get_n_frames() - 1
        elif ':' in time_indicator:
            sec = parse_time_delta_str_to_sec(time_indicator)
            return round(sec * self.get_metadata().fps)
        elif time_indicator.endswith('s'):
            sec = float(time_indicator.rstrip('s'))
            return round(sec * self.get_metadata().fps)
        elif time_indicator.endswith('%'):
            percent = float(time_indicator.rstrip('%'))
            return round(percent / 100 * self.get_n_frames())
        elif all(c in '0123456789' for c in time_indicator):
            return int(time_indicator)
        else:
            return None

    def iter_frame_ixs(self, time_interval: TimeIntervalTuple = (None, None),
                       frame_interval: Tuple[Optional[int], Optional[int]] = (None, None)
                       ) -> Iterator[int]:
        start, stop = get_actual_frame_interval(fps=self._fps, time_interval=time_interval,
                                                frame_interval=frame_interval,
                                                n_frames_total=self.get_n_frames())
        """ 
        TODO: Get rid of arguments - just use cut instead 
        """
        return range(start, stop)

    def iter_frames(self,
                    time_interval: TimeIntervalTuple = (None, None),
                    frame_interval: Tuple[Optional[int], Optional[int]] = (None, None)
                    ) -> Iterator[VideoFrameInfo]:
        """
        TODO: Get rid of arguments - just use cut instead
        """
        for i in self.iter_frame_ixs(time_interval=time_interval, frame_interval=frame_interval):
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
                    raise RuntimeError(f'Did not find target within {max_seek_search} frames of seek')
                if f.pts >= pts - 1:
                    self._iterator = itertools.chain([f], self._iterator)
                    break
            self._next_index_to_be_read = index_in_file
            return self.request_frame(index)

    def destroy(self):
        self._iterator = None
        self._frame_cache = None
        self._is_destroyed = True


    # def __del__(self):
    #     print(f"Video reader {id(self)} closed!")
