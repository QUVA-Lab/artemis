"""
To get this working (MacOS):

1) Download Local RTMP Server, run it
2) Stream a video locally: ffmpeg -stream_loop -1 -i /Users/peter/drone/dji/raw/dji_2023-02-19_23-51-29_0737.mp4 -f flv rtmp://127.0.0.1/live
3) Run this script
"""
import multiprocessing
import os.path
import queue
from contextlib import contextmanager
from functools import partial
from multiprocessing import Queue
from typing import Optional

import cv2
import numpy as np
from dataclasses import dataclass, field

from artemis.general.debug_utils import easy_profile
from artemis.image_processing.video_frame import VideoFrameInfo
import logging

LIVESTREAM_LOGGER = logging.getLogger("livestream_recorder")


def read_stream_and_save_to_disk(
        stream_url: str,
        writing_video_path: Optional[str] = None,
        poison_pill_input_queue: Optional["Queue[bool]"] = None,
        latest_frame_return_queue: Optional["Queue[VideoFrameInfo]"] = None,
        verbose: bool = False
):
    """ Hey ChatGPT can you fill in this function please?  """

    LIVESTREAM_LOGGER.setLevel(logging.DEBUG if verbose else logging.WARNING)

    LIVESTREAM_LOGGER.debug(f"Trying to read stream {stream_url}")
    cap = cv2.VideoCapture(stream_url)
    LIVESTREAM_LOGGER.debug("Got stream!")
    writer = None

    if cap.isOpened() is False:
        LIVESTREAM_LOGGER.info("Error opening the stream or video")
        return
    LIVESTREAM_LOGGER.info("Launching reader thread")
    frame_ix = 0
    last_frame = None
    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                LIVESTREAM_LOGGER.debug(f"Process: Found frame of size {frame.shape}")
                if writing_video_path is not None and writer is None:
                    # Initialize our video writer
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(os.path.expanduser(writing_video_path), fourcc, cap.get(cv2.CAP_PROP_FPS),
                                             (frame.shape[1], frame.shape[0]), True)
                if last_frame is not None and np.array_equal(frame, last_frame):
                    LIVESTREAM_LOGGER.debug("Process: Found duplicate frame, skipping")
                    continue
                last_frame = frame

                # Write the output frame to disk
                if writer is not None:
                    writer.write(frame)

                seconds_into_video = frame_ix / cap.get(cv2.CAP_PROP_FPS)
                frame_info = VideoFrameInfo(image=frame, seconds_into_video=seconds_into_video,
                                            frame_ix=frame_ix, fps=cap.get(cv2.CAP_PROP_FPS))
                if latest_frame_return_queue is not None:
                    if not latest_frame_return_queue.full():
                        latest_frame_return_queue.put(frame_info)
                frame_ix += 1
            else:
                LIVESTREAM_LOGGER.debug("Process: Found no frame from stream")
                break

            if poison_pill_input_queue is not None and not poison_pill_input_queue.empty():
                if poison_pill_input_queue.get():
                    break
    finally:
        cap.release()
        if writer is not None:
            with easy_profile(f"Process: Finalizing video {writing_video_path}...", log_entry=True):
                writer.release()
        else:
            LIVESTREAM_LOGGER.debug("Process: Done!")
    LIVESTREAM_LOGGER.warning("Ending Livestream Process")


@dataclass
class LiveStreamRecorderAgent:
    stream_url: str
    writing_video_path: Optional[str] = None
    poison_pill_input_queue: Optional["Queue[bool]"] = field(default_factory=lambda: Queue(maxsize=1))
    latest_frame_return_queue: Optional["Queue[VideoFrameInfo]"] = field(default_factory=lambda: Queue(maxsize=2))

    def launch(self):
        self.process = multiprocessing.Process(
            target=partial(read_stream_and_save_to_disk,
                           stream_url=self.stream_url,
                           writing_video_path=self.writing_video_path,
                           poison_pill_input_queue=self.poison_pill_input_queue,
                           latest_frame_return_queue=self.latest_frame_return_queue
                           )
        )
        self.process.start()

    def get_last_frame(self) -> Optional[VideoFrameInfo]:
        try:
            return self.latest_frame_return_queue.get_nowait()
        except queue.Empty:
            return None

    def get_last_frame_blocking(self, timeout: Optional[float] = None) -> VideoFrameInfo:
        return self.latest_frame_return_queue.get(timeout=timeout)

    @contextmanager
    def launch_and_iter_frames_context(self):
        try:
            yield self.launch_and_iter_frames()
        finally:
            self.kill()

    def launch_and_iter_frames(self):
        self.launch()
        while True:
            try:
                # print("Main: Trying to get frame")
                frame_info = self.latest_frame_return_queue.get(timeout=0.1)
            except queue.Empty:
                # print("Main: But no frame")
                continue
            yield frame_info

    def kill(self):
        print("Stopping Livestrean process...")
        self.poison_pill_input_queue.put(True)
        while not self.latest_frame_return_queue.empty():
            try:
                self.latest_frame_return_queue.get_nowait()
            except queue.Empty:
                continue
        self.process.terminate()  # Forcefully stop the child process
        self.process.join()  # Wait for the child process to stop
        print("Stopped Livestrean process.")


def demo_livestream_viewer():
    agent = LiveStreamRecorderAgent(
        stream_url="rtmp://127.0.0.1:1935/live",
        writing_video_path='~/Downloads/demo_livestream_record.mp4',
    )
    with agent.launch_and_iter_frames_context() as frame_iterator:
        for frame_info in frame_iterator:
            cv2.imshow('frame', frame_info.image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print("Out of loop")
    print("Out of context manager")


if __name__ == '__main__':
    demo_livestream_viewer()
