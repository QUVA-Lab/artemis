from dataclasses import dataclass
from typing import Tuple, Optional

from artemis.general.custom_types import BGRImageArray


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
