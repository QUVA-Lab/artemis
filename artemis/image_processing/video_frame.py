from dataclasses import dataclass
from datetime import datetime
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


@dataclass
class FrameGeoData:
    lat_long: Optional[Tuple[float, float]]
    epoch_time_us: int
    altitude_from_home: Optional[float] = None
    altitude_from_sea: Optional[float] = None
    # def has_latlong(self) -> bool:  # Hope you're not flying off the west coast of Africa
    #     return self.latitude != 0 and self.longitude != 0

    def get_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.epoch_time_us/1000000)

    def get_timestamp(self) -> float:
        return self.epoch_time_us/1e6

    def get_time_str(self) -> str:
        return self.get_datetime().strftime('%Y-%m-%d %H:%M:%S.%f')

    def get_latlng_str(self) -> str:
        if self.lat_long is not None:
            return f'{self.lat_long[0]:.5f}, {self.lat_long[1]:.5f}'
        else:
            return 'Unknown'

    def get_altitude_str(self) -> str:
        return f"{self.altitude_from_home:.1f}m (home)" if self.altitude_from_home is not None \
            else f"{self.altitude_from_sea:.1f}m (sea)" if self.altitude_from_sea is not None \
            else "?m"

    def get_latlng_alt_str(self) -> str:
        return self.get_latlng_str()+f", {self.get_altitude_str()}"
