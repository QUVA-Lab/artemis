from typing import Optional, Callable

import cv2
from scipy._lib.decorator import contextmanager

from artemis.general.custom_types import BGRImageArray
from artemis.plotting.cv_keys import Keys, cvkey_to_key
from artemis.image_processing.image_utils import ShowFunction
from artemis.general.utils_utils import get_context_name


def cvkey_to_str(keycode: int) -> str:
    return chr(keycode) if keycode != -1 else None


_ALTERNATE_SHOW_FUNC: Optional[Callable[[BGRImageArray, str], None]] = None


@contextmanager
def hold_alternate_show_func(alternate_func: ShowFunction):
    global _ALTERNATE_SHOW_FUNC
    old = _ALTERNATE_SHOW_FUNC
    _ALTERNATE_SHOW_FUNC = alternate_func
    try:
        yield alternate_func
    finally:
        _ALTERNATE_SHOW_FUNC = old


def just_show(image, name: Optional[str] = None, hang_time: float = 0, upsample_factor: int = 1, enable_alternate: bool = True) -> Optional[Keys]:
    if name is None:
        name = get_context_name(levels_up=2)
    if upsample_factor != 1:
        image = cv2.resize(image, dsize=None, fx=upsample_factor, fy=upsample_factor, interpolation=cv2.INTER_NEAREST)
    if enable_alternate and _ALTERNATE_SHOW_FUNC is not None:
        return _ALTERNATE_SHOW_FUNC(image, name)
    else:
        cv2.imshow(name, image)
        keycode = cv2.waitKey(1 + int(hang_time * 1000) if not hang_time == float('inf') else 1000000000)
        return cvkey_to_key(keycode)
