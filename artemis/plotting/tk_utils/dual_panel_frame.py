import tkinter as tk
from typing import Optional, Callable, Tuple

from artemis.general.custom_types import BGRImageArray
from artemis.image_processing.image_utils import ImageViewInfo
from artemis.plotting.tk_utils.alternate_zoomable_image_view import ZoomableImageFrame
from artemis.plotting.tk_utils.constants import ThemeColours
from artemis.plotting.tk_utils.ui_utils import RespectableButton, ToggleLabel


class DualPanelFrame(tk.Frame):

    def __init__(self,
                 parent_frame: tk.Frame,
                 lock_view_frames: bool = False,
                 left_label: str = "",
                 right_label: str = "",
                 frame_click_callback: Optional[Callable[[Tuple[int, int], int], None]] = None
                 ):

        tk.Frame.__init__(self, parent_frame, bg=ThemeColours.BACKGROUND)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self._lock_view_frames = lock_view_frames

        # control_frame = tk.Frame(self, bg=ThemeColours.BACKGROUND)
        # control_frame.grid(row=0, column=0, columnspan=2, sticky=tk.EW)
        #
        # # TkImageWidget(control_frame, image=cv2.imread(AssetFiles.LOGO_IMAGE_PATH_SQUARE), width=50).pack(side=tk.LEFT)
        #
        # tk.Label(control_frame, text="Eagle Eyes Roamer", bg=ThemeColours.BACKGROUND, fg=ThemeColours.TEXT, justify=tk.CENTER, font=("Helvetica", 24)
        #          ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # self._export_button = RespectableButton(
        #     control_frame,
        #     text="Export Situation Map as GeoTiff",
        #     shortcut='e',
        #     command=lambda: print("Exporting...")
        # )
        # self._export_button.pack(side=tk.RIGHT)

        # recompute_button = tk.Label(control_frame, text="Recompute (P)", relief=tk.RAISED, cursor='hand2', bg=ThemeColours.BACKGROUND, fg=ThemeColours.TEXT)
        # recompute_button.pack(side=tk.RIGHT)

        # clear_button = tk.Label(control_frame, text="Clear (Y)", relief=tk.RAISED, cursor='hand2', bg=ThemeColours.BACKGROUND, fg=ThemeColours.TEXT)
        # clear_button.bind("<Button-1>", lambda e: self._clear_labels())
        # clear_button.pack(side=tk.RIGHT)

        self.left_view = ZoomableImageFrame(self, after_view_change_callback=lambda vf: self._set_view_frame(vf, view=self.right_view), double_click_callback=lambda xy: self._on_frame_click(xy, 0))
        self.left_view.grid(column=0, row=1, sticky=tk.NSEW)
        tk.Label(self, text=left_label, bg=ThemeColours.BACKGROUND, fg=ThemeColours.TEXT).grid(column=0, row=2, sticky=tk.NSEW)
        self.right_view = ZoomableImageFrame(self, after_view_change_callback=lambda vf: self._set_view_frame(vf, view=self.left_view), double_click_callback=lambda xy: self._on_frame_click(xy, 1))
        self.right_view.grid(column=1, row=1, sticky=tk.NSEW)
        tk.Label(self, text=right_label, bg=ThemeColours.BACKGROUND, fg=ThemeColours.TEXT).grid(column=1, row=2, sticky=tk.NSEW)

        self._frame_click_callback = frame_click_callback

    # def mark_keypoint_on_map(self, event: tk.Event):
    #     current_frame = self.map_view.get_image_view_frame_or_none()
    #     if current_frame is not None:
    #         pixel_xy = current_frame.display_xy_to_pixel_xy((event.x, event.y))
    #         self._map_keypoints[len(self._map_keypoints)] = tuple(int(d) for d in pixel_xy)
    #
    #         self._redraw()

    # def _clear_labels(self):
    #     if self._imseg is not None and self._selected_heatmap is not None:
    #         self._imseg.segmentation_heatmaps[self._selected_heatmap].label_map[:, :] = 0
    #         self._imseg.segmentation_heatmaps[self._selected_heatmap].prob_map[:, :] = 0.
    #         del self._cached_prob_maps_images[self._selected_heatmap]
    #         self._redraw()
    #
    # def _mark_keypoint(self, event: tk.Event):
    #
    #     point_id = int(event.char)
    #     is_in_image_frame = event.widget is self.left_view
    #     is_in_map_frame = event.widget is self.right_view
    #     widget: Optional[ZoomableImageFrame] = self.left_view if is_in_image_frame else self.right_view if is_in_map_frame else None
    #     if widget is None:
    #         return
    #     current_frame = widget.get_image_view_frame_or_none()
    #     pixel_xy = current_frame.display_xy_to_pixel_xy((event.x, event.y))
    #     pixel_xy = tuple(int(d) for d in pixel_xy)
    #     if is_in_image_frame:
    #         self._image_keypoints[point_id] = pixel_xy
    #         print("Image Keypoints: {}".format(self._image_keypoints))
    #     elif is_in_map_frame:
    #         self._map_keypoints[point_id] = pixel_xy
    #         print("Map Keypoints: {}".format(self._map_keypoints))
    #
    #     self._redraw(redraw_map=is_in_map_frame, redraw_image=is_in_image_frame)

        # current_frame = self.map_view.get_image_view_frame_or_none()
        # if current_frame is not None and self._imseg is not None and self._selected_heatmap is not None:
        #     pixel_xy = current_frame.display_xy_to_pixel_xy((event.x, event.y))
        #     # draw_circle_on_label_inplace(self._imseg.segmentation_heatmaps[self._selected_heatmap].label_map, pixel_xy, radius=radius, fill_value=value)
            # self._redraw()

    def _on_frame_click(self, xy: Tuple[int, int], side: int):
        if self._frame_click_callback is not None:
            self._frame_click_callback(xy, side)


    def _set_view_frame(self, image_view_info: ImageViewInfo, view: ZoomableImageFrame):
        if self._lock_view_frames and view.get_image_view_frame_or_none() != image_view_info:
            view.set_image_frame(image_view_info)

    def update_left(self, image: BGRImageArray):
        self.left_view.set_image(image)

    def update_right(self, image: BGRImageArray):
        self.right_view.set_image(image)
    #
    # def _redraw(self, redraw_left: bool = True, redraw_right: bool = True):
    #     if redraw_left:
    #         self.left_view.set_image(image)
    #     if redraw_right and self. is not None:
    #         self.right_view.set_image(map_image)



