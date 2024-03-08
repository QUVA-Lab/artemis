from artemis.plotting.easy_window import ImageRow, ImageCol
from artemis.image_processing.image_processing_utils import compute_aloneness_factor
from artemis.image_processing.image_utils import BGRColors, heatmap_to_color_image
from artemis.plotting.cv2_plotting import just_show
from artemis.image_processing.image_builder import ImageBuilder
import numpy as np

def test_aloneness_factor():

    sx, sy = 480, 480
    rng = np.random.RandomState(1234)
    builder = ImageBuilder.from_blank(size=(sx, sy), color=BGRColors.BLACK)

    builder.draw_circle(center_xy=(100, 100), radius=6, colour=BGRColors.GREEN, thickness=-1)
    builder.draw_circle(center_xy=(400, 100), radius=6, colour=BGRColors.GREEN, thickness=-1)
    builder.draw_circle(center_xy=(400, 120), radius=6, colour=BGRColors.RED, thickness=-1)
    builder.draw_circle(center_xy=(420, 100), radius=6, colour=BGRColors.GREEN, thickness=-1)
    builder.draw_line(start_xy=(100, 400), end_xy=(600, 300), color=BGRColors.RED, thickness=4)

    builder.superimpose(rng.randn(sy, sx, 3)**2 * 20)

    img = builder.get_image()

    heatmap = img.mean(axis=2)/255

    factor = compute_aloneness_factor(heatmap=heatmap, outer_box_width=100, inner_box_width=5, suppression_factor=3, n_iter=5)
    # for _ in range(10):
    # factor = compute_aloneness_factor(heatmap=factor, outer_box_width=100, inner_box_width=5, suppression_factor=5)
    # factor = non_maximal_suppression(heatmap=heatmap, outer_box_width=100, inner_box_width=5, n_iter=2, suppression_factor=100.)

    disp_image = ImageCol(ImageRow(img, heatmap_to_color_image(heatmap, show_range=True)),
                          ImageRow(heatmap_to_color_image(factor, show_range=True), img*factor[:, :, None])).render()
    just_show(disp_image, hang_time=100)


if __name__ == '__main__':
    test_aloneness_factor()
