from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional
import utm
import pytz

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
    bearing: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    # def has_latlong(self) -> bool:  # Hope you're not flying off the west coast of Africa
    #     return self.latitude != 0 and self.longitude != 0

    def __str__(self):
        return f"FrameGeoData(date={self.get_time_str()}, (lat, long)={self.get_latlng_str()}, alt={self.get_altitude_str()})"

    def get_datetime(self, localize: bool = True) -> datetime:
        dt = datetime.fromtimestamp(self.epoch_time_us/1000000)  # In UTC
        if localize and self.lat_long is not None:  # Use pytz
            # TODO: Avoid circular import
            from artemis.image_processing.media_metadata import get_timezone_finder_singleton
            tzf = get_timezone_finder_singleton()
            timezone_str = tzf.timezone_at(lat=self.lat_long[0], lng=self.lat_long[1])
            if timezone_str is not None:
                dt = pytz.timezone(timezone_str).localize(dt)
        return dt

    def get_timestamp(self) -> float:
        return self.epoch_time_us/1e6

    def get_time_str(self) -> str:
        # 2-decimals of precision is enough for 1/100th of a second
        # return self.get_datetime().strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]
        # Include zone
        dt = self.get_datetime()
        if dt.tzinfo is None:
            return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-4] + ' UTC'
        else:
            return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-4] + ' ' + dt.tzinfo.tzname(dt)
        # return self.get_datetime().strftime('%Y-%m-%d %H:%M:%S.%f %Z')

    def get_latlng_str(self, format='dd') -> str:
        if self.lat_long is not None:
            if format == 'dd':
                return f'{self.lat_long[0]:.5f}, {self.lat_long[1]:.5f}'
            elif format == 'dms':
                is_north = self.lat_long[0] >= 0
                is_east = self.lat_long[1] >= 0
                lat_d, lat_m, lat_s = self.lat_long[0], self.lat_long[0] % 1 * 60, self.lat_long[0] % 1 * 60 % 1 * 60
                lng_d, lng_m, lng_s = self.lat_long[1], self.lat_long[1] % 1 * 60, self.lat_long[1] % 1 * 60 % 1 * 60
                return f'{lat_d:.0f}°{lat_m:.0f}\'{lat_s:.2f}"{"N" if is_north else "S"}, {lng_d:.0f}°{lng_m:.0f}\'{lng_s:.2f}"{"E" if is_east else "W"}'
            else:
                raise ValueError(f"Unknown format: {format}")
        else:
            return 'Unknown'

    def get_utm_str(self) -> str:
        if self.lat_long is not None:
            easting, northing, zone_number, zone_letter = utm.from_latlon(self.lat_long[0], self.lat_long[1])
            return f"{zone_number}{zone_letter} {easting:.0f} {northing:.0f} "
        else:
            return 'Unknown'

    def get_camera_attitude_str(self) -> str:
        return f"Bearing:{self.bearing:.1f}°, Pitch:({self.pitch:.1f}°, Roll:{self.roll:.1f}°)" if self.pitch is not None and self.roll is not None and self.bearing is not None else ""

    def get_map_link(self, program: str = 'google_maps') -> str:
        # if choice == op_gmaps:
        #     path = f"https://www.google.com/maps/search/?api=1&query={lat},{long}"
        # elif choice == op_gearth:
        #     path = f"https://earth.google.com/web/search/{lat},{long}"
        # elif choice == op_sartopo:
        #     # https://sartopo.com/map.html#ll=49.1297,-123.97042&z=14&b=mbt
        #     path = f"https://sartopo.com/map.html#ll={lat},{long}&z=14&b=mbt"
        if self.lat_long is not None:
            if program == 'google_maps':
                return f"https://www.google.com/maps/search/?api=1&query={self.lat_long[0]:.6f},{self.lat_long[1]:.6f}"
            elif program == 'google_earth':
                return f"https://earth.google.com/web/search/{self.lat_long[0]:.6f},{self.lat_long[1]:.6f}"
            elif program == 'sartopo':
                # https://sartopo.com/map.html#ll=49.1297,-123.97042&z=14&b=mbt
                return f"https://sartopo.com/map.html#ll={self.lat_long[0]:.6f},{self.lat_long[1]:.6f}&z=14&b=mbt"
            else:
                raise ValueError(f"Unknown program: {program}")
        else:
            return 'Unknown'

    def get_altitude_str(self) -> str:
        return f"{self.altitude_from_home:.1f}m (home)" if self.altitude_from_home is not None \
            else f"{self.altitude_from_sea:.1f}m (sea)" if self.altitude_from_sea is not None \
            else "?m"

    def get_latlng_alt_str(self) -> str:
        return self.get_latlng_str()+f", {self.get_altitude_str()}"
