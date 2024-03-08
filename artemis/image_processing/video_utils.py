from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional, Callable
from more_itertools import first

from artemis.general.custom_types import BGRImageArray
from artemis.image_processing.image_utils import fade_image


@dataclass
class VideoFader:
    intro_fade_frames: int
    outro_fade_frames: int
    pause_during_fade: bool = True

    def iter_fade_video(self, video: Iterable[BGRImageArray]) -> Iterable[BGRImageArray]:

        if self.pause_during_fade:
            iter_in = iter(video)
            try:
                img = first(iter_in)
            except StopIteration:
                return
            yield from (fade_image(img, i/self.intro_fade_frames) for i in range(self.intro_fade_frames))
            for img in iter_in:
                yield img
            yield from (fade_image(img, (self.outro_fade_frames-i-1)/self.outro_fade_frames) for i in range(self.outro_fade_frames))

        else:

            frame_queue = deque()

            for i, img in enumerate(video):
                faded_img = fade_image(img, i/self.intro_fade_frames) if i<self.intro_fade_frames else img
                frame_queue.append(faded_img)

                if len(frame_queue)>self.outro_fade_frames:
                    yield frame_queue.popleft()

            for i in range(len(frame_queue)):
                img = frame_queue.popleft()
                yield fade_image(img, (self.outro_fade_frames-i+1)/self.outro_fade_frames)


@dataclass
class LastFrameHanger:
    n_steps_to_hang: int

    def iter_hang_last_frame(self, video: Iterable[BGRImageArray]) -> Iterable[BGRImageArray]:
        img = None
        for img in video:
            yield img
        if img is not None:
            yield from (img for _ in range(self.n_steps_to_hang))

@dataclass
class PassThroughVideoCaller:

    side_func: Optional[Callable[[BGRImageArray], None]]
    mod_func: Optional[Callable[[BGRImageArray], BGRImageArray]] = (lambda x: x)

    def iter_frames(self, video: Iterable[BGRImageArray]) -> Iterable[BGRImageArray]:

        for img in video:
            self.side_func(img)
            yield self.mod_func(img)



