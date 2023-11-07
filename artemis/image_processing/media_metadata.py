from datetime import datetime
from typing import Optional, Tuple

from datetime import datetime, timedelta
from timezonefinder import TimezoneFinder
import pytz

import exif
from pymediainfo import MediaInfo

from artemis.image_processing.video_frame import FrameGeoData


def degrees_minutes_seconds_to_degrees(degrees: float, minutes: float, seconds: float) -> float:
    return degrees + minutes/60 + seconds/3600


def read_exif_data_from_path(path: str, max_bytes_to_read_for_exif: Optional[int]=None,
                             ) -> Optional[exif.Image]:
    try:
        exif_data = exif.Image(path)
    except Exception as err:
        print(f"Got error trying to read exif data from {path}: {err}")
        return None
    if not exif_data.has_exif:
        exif_data = None
    return exif_data


def is_daylight_saving(localized_datetime: datetime) -> bool:
    tz = localized_datetime.tzinfo
    return tz.dst(localized_datetime) != timedelta(0)


_TIMEZONE_FINDER = None


def get_timezone_finder_singleton() -> TimezoneFinder:
    global _TIMEZONE_FINDER
    if _TIMEZONE_FINDER is None:
        _TIMEZONE_FINDER = TimezoneFinder()
    return _TIMEZONE_FINDER


def get_utc_epoch(
        lat_long: Optional[Tuple[float, float]],
        local_timestamp_str: str,
        local_timestamp_format: str='%Y-%m-%d %H:%M:%S',
        requires_dst_correction: bool = False,
        fail_on_no_timezone: bool = False,
    ) -> float:
    """ Tries its best to get a UTC timestamp from the kind of metadata that comes with images and videos. """
    # Initialize timezone finder and find the timezone
    tf = get_timezone_finder_singleton()
    tz_str = tf.timezone_at(lat=lat_long[0], lng=lat_long[1]) if lat_long is not None else None
    if tz_str is None:
        if fail_on_no_timezone:
            raise ValueError("No timezone found for lat/long")
        print("Warning: No timezone found for lat/long, using UTC, even though it's probably wrong")
        tz_str = 'UTC'
    # Parse the local timestamp string into a datetime object
    local_tz = pytz.timezone(tz_str)
    if '.' in local_timestamp_str:
        unlocalized_datetime = datetime.strptime(local_timestamp_str, local_timestamp_format+'.%f')
    else:
        unlocalized_datetime = datetime.strptime(local_timestamp_str, local_timestamp_format)
    # unlocalized_datetime = datetime.strptime(local_timestamp_str, local_timestamp_format)

    # Localize the timestamp to the local timezone
    localized_datetime = local_tz.localize(unlocalized_datetime)

    if requires_dst_correction and is_daylight_saving(localized_datetime):
        localized_datetime += timedelta(hours=1)

    return localized_datetime.timestamp()

    # # Convert the timestamp to UTC timezone
    # utc_timestamp = local_timestamp.astimezone(pytz.utc)
    #
    # # Convert the UTC datetime object to epoch time
    # epoch_timestamp = (utc_timestamp - datetime(1970, 1, 1, tzinfo=pytz.utc)).total_seconds()
    #
    # print(f"Shifted timestamp by {(epoch_timestamp - local_timestamp.timestamp())/3600} houres to get UTC time")
    #
    # return epoch_timestamp

# # Example usage
# lat, lon = 40.7128, -74.0060  # New York City
# local_timestamp_str = "2023-09-22 12:34:56"
# epoch_timestamp = get_utc_epoch(lat, lon, local_timestamp_str)
# print("Epoch timestamp in UTC:", epoch_timestamp)



class ExifParseError(Exception):
    """ Raised """


def convert_exif_to_geodata_or_none(exif_data: exif.Image, requires_dst_correction: bool = False) -> Optional[FrameGeoData]:
    try:
        gps_latitude = degrees_minutes_seconds_to_degrees(*exif_data.gps_latitude)
        gps_longitude = degrees_minutes_seconds_to_degrees(*exif_data.gps_longitude)
        if exif_data.gps_latitude_ref == 'S':
            gps_latitude = -gps_latitude
        if exif_data.gps_longitude_ref == 'W':
            gps_longitude = -gps_longitude
        lat_long = (gps_latitude, gps_longitude)
    except Exception as err:
        print(f"Error parsing GPS data: {err}")
        lat_long = None
    try:
        gps_altitude = exif_data.gps_altitude
    except Exception as err:
        print(f"Error parsing GPS altitude: {err}")
        gps_altitude = None

    try:
        # Seems the images already have UTC time in them, so we don't need to do this
        epoch_timestamp = get_utc_epoch(lat_long=lat_long, local_timestamp_str=exif_data.datetime,
                                        local_timestamp_format='%Y:%m:%d %H:%M:%S', requires_dst_correction=requires_dst_correction)
        epoch_time_us = int(epoch_timestamp * 1000000)
    except Exception as err:
        print(f"Error parsing GPS timestamp: {err}")
        epoch_time_us = 0
    return FrameGeoData(lat_long=lat_long, epoch_time_us=epoch_time_us, altitude_from_sea=gps_altitude)


def read_image_geodata_or_none(image_path: str, requires_dst_correction: bool = False) -> Optional[FrameGeoData]:
    """ Read the exif data from the image to extract lat, long, timestamp """
    exif_data = read_exif_data_from_path(image_path)
    return convert_exif_to_geodata_or_none(exif_data, requires_dst_correction=requires_dst_correction) if exif_data else None




    #
    # with open(image_path, 'rb') as image_file:
    #     try:
    #         exif_data = exif.Image(image_file)
    #     except Exception as err:
    #         print(f"Error reading exif data from {image_path}: {err}")
    #         return None
    #     if exif_data.has_exif:
    #         try:
    #             gps_latitude = degrees_minutes_seconds_to_degrees(*exif_data.gps_latitude)
    #             gps_longitude = degrees_minutes_seconds_to_degrees(*exif_data.gps_longitude)
    #             if exif_data.gps_latitude_ref == 'S':
    #                 gps_latitude = -gps_latitude
    #             if exif_data.gps_longitude_ref == 'W':
    #                 gps_longitude = -gps_longitude
    #         except Exception as err:
    #             print(f"Error parsing GPS data from {image_path}: {err}")
    #             return None
    #         try:
    #             gps_altitude = exif_data.gps_altitude
    #         except Exception as err:
    #             print(f"Error parsing GPS altitude from {image_path}: {err}")
    #             gps_altitude = None
    #         # parse string like '2022:11:20 14:03:13' into datetime
    #         datetime_obj = datetime.strptime(exif_data.datetime, '%Y:%m:%d %H:%M:%S')
    #         epoch_time_us = int(datetime_obj.timestamp()*1000000)
    #         return FrameGeoData((gps_latitude, gps_longitude), epoch_time_us, altitude_from_sea=gps_altitude)
    # return None
    #
