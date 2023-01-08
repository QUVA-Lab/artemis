import numpy as np
from matplotlib import pyplot as plt

from artemis.general.custom_types import BGRImageArray


def fig_to_bgr_array(fig: plt.Figure) -> BGRImageArray:
    renderer = fig.canvas.renderer
    arr = np.fromstring(renderer.tostring_rgb(), dtype=np.uint8).reshape(int(renderer.height), int(renderer.width), 3)[:, :, ::-1]
    return arr
