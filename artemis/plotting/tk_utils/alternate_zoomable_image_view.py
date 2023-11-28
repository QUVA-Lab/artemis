# -*- coding: utf-8 -*-
# Advanced zoom example. Like in Google Maps.
# It zooms only a tile, but not the whole image. So the zoomed tile occupies
# constant memory and not crams it with a huge resized image for the large zooms.
import time
import tkinter as tk
from contextlib import contextmanager
from math import copysign
from tkinter import EventType, Event
from typing import Optional, Tuple, Callable, Mapping, Dict, List

import cv2
from PIL import ImageTk

from artemis.fileman.smart_io import smart_load_image
from artemis.general.custom_types import BGRImageArray, BGRColorTuple
from artemis.image_processing.image_utils import ImageViewInfo, BGRColors
from artemis.plotting.tk_utils.machine_utils import is_windows_machine
from artemis.plotting.tk_utils.tk_error_dialog import tk_show_eagle_eyes_error_dialog, ErrorDetail
from artemis.plotting.tk_utils.tk_utils import bind_callbacks_to_widget
from artemis.plotting.tk_utils.ui_utils import bgr_image_to_pil_image


class ZoomableImageFrame(tk.Label):

    def __init__(self,
                 parent_frame: tk.Frame,
                 image: Optional[BGRImageArray] = None,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 gap_color: BGRColorTuple = BGRColors.DARK_GRAY,
                 scrollbar_color: BGRColorTuple = BGRColors.GRAY,
                 zoom_jump_factor: float = 1.2,
                 max_zoom: float = 40.0,
                 pan_jump_factor=0.2,
                 mouse_scroll_speed: float = 2.0,
                 error_handler: Optional[Callable[[ErrorDetail], None]] = tk_show_eagle_eyes_error_dialog,
                 zoom_scrolling_mode: bool = False,  # Use mouse scrollwheel to zoom,
                 after_view_change_callback: Optional[Callable[[ImageViewInfo], None]] = None,
                 additional_canvas_callbacks: Optional[Mapping[str, Callable[[Event], None]]] = None,
                 single_click_callback: Optional[Callable[[Tuple[int, int]], None]] = None,
                 double_click_callback: Optional[Callable[[Tuple[int, int]], None]] = None,
                 mouse_callback: Optional[Callable[[Event, Tuple[int, int]], bool]] = None,  # Takes the event and the pixel xy
                 scroll_indicator_width_pix: int = 10,
                 rel_area_change_to_reset_zoom: float = 0.25,
                 margin_gap: int = 4,  # Prevents infinite config-loop
                 ):

        # self.label = tk.Label(parent_frame)
        super().__init__(master=parent_frame, width=width, height=height)
        # self.label.pack()
        # assert height is not None or width is not None, "You must specify height, width, or both to display image"
        # self.height = height
        # self.width = width
        self._after_view_change_callback = after_view_change_callback
        self._mouse_scroll_speed = mouse_scroll_speed
        self._image_view_frame: Optional[ImageViewInfo] = None
        self._recent_configured_whs: List[Tuple[int, int]] = []
        self._last_configured_wh: Optional[Tuple[int, int]] = None
        self._scroll_indicator_width_pix = scroll_indicator_width_pix
        self._gap_color = gap_color
        self._scrollbar_color = scrollbar_color
        self._margin_gap = margin_gap
        self._first_pass = True
        self._rel_area_change_to_reset_zoom = rel_area_change_to_reset_zoom
        self._single_click_callback = single_click_callback
        self._double_click_callback = double_click_callback
        self._mouse_callback = mouse_callback
        self._zoom_jump_factor = zoom_jump_factor
        self._hold_off_redraw = False  # Disales redraw while true.  Use through self.hold_off_redraw_context()
        self._zoom_scrolling_mode = zoom_scrolling_mode
        self._max_zoom = max_zoom
        self._drag_start_display_xy: Optional[Tuple[int, int]] = None
        self._image: Optional[BGRImageArray] = None
        self._is_configuration_still_being_negotiated = True
        if image is not None:
            self.set_image(image)

        self._binding_dict: Dict[str, Callable[[Event], None]] = {
            **(additional_canvas_callbacks or {}),
            **{
                '<z>': lambda event: self.set_image_frame(self._image_view_frame.zoom_by(zoom_jump_factor, invariant_display_xy=self._event_to_display_xy(event))),
                '<x>': lambda event: self.set_image_frame(self._image_view_frame.zoom_by(1 / zoom_jump_factor, invariant_display_xy=self._event_to_display_xy(event))),
                '<c>': lambda event: self.set_image_frame(self._image_view_frame.zoom_out()),
                '<w>': lambda event: self.set_image_frame(self._image_view_frame.pan_by_display_relshift(display_rel_xy=(0, -pan_jump_factor), limit=True)),
                '<a>': lambda event: self.set_image_frame(self._image_view_frame.pan_by_display_relshift(display_rel_xy=(-pan_jump_factor, 0), limit=True)),
                '<s>': lambda event: self.set_image_frame(self._image_view_frame.pan_by_display_relshift(display_rel_xy=(0, pan_jump_factor), limit=True)),
                '<d>': lambda event: self.set_image_frame(self._image_view_frame.pan_by_display_relshift(display_rel_xy=(pan_jump_factor, 0), limit=True)),
                '<Button-1>': self._on_click,  # For some reason, this is not working...
                # '<Button-1>': lambda event: print("Single click"),  # This never gets called
                '<Double-Button-1>': self._on_double_click,
                # Handle mouse-drag and release
                '<B1-Motion>': self._on_mouse_drag_and_release,
                '<ButtonRelease-1>': self._on_mouse_drag_and_release,
                '<MouseWheel>': self._handle_mousewheel_event,
                '<Button-5>': self._handle_mousewheel_event,
                '<Button-4>': self._handle_mousewheel_event,
                '<Configure>': self._on_configure,  # This may be unnecessary - and it can cause dangerous loops
                # Add number-pad callbacks: 5 to zoom in, 0 to zoom out, 1-9 to pan
                "<KP_5>": lambda event: self.set_image_frame(self._image_view_frame.zoom_by(zoom_jump_factor, invariant_display_xy=None)),
                "<KP_0>": lambda event: self.set_image_frame(self._image_view_frame.zoom_by(1 / zoom_jump_factor, invariant_display_xy=None)),
                **{f"<KP_{i}>": lambda event, i=i: self.set_image_frame(self._image_view_frame.pan_by_display_relshift(display_rel_xy=(pan_jump_factor*(((i-1) % 3)-1), -pan_jump_factor*(((i-1)//3)-1)), limit=True)) for i in [1, 2, 3, 4, 6, 7, 8, 9]},
                # Add callbacks for entering/exiting focus:



                # '<Double-Button-1>': double_click,
                # '<ButtonPress-1>': lambda x: print("Single Click"),
                #    '<B1-Motion>': self.move_to,
                #    '<MouseWheel>': self.wheel,
                #    '<Button-5>': self.wheel,
                #    '<Button-4>': self.wheel,
                #    },
                # **{f"<{key}>": self.onKeyPress for key in 'zxcwasd'}
            }}
        bind_callbacks_to_widget(callbacks=self._binding_dict, widget=self, bind_all=False, error_handler=error_handler)
        self.bind("<1>", lambda event: self.focus_set())
        # self.rebind()

    @classmethod
    def launch_as_standalone(cls, image: BGRImageArray):
        root = tk.Tk()
        root.geometry("1280x720")
        frame = ZoomableImageFrame(root)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        frame.set_image(image)
        root.mainloop()

    def _on_configure(self, event: Event):


        # Avoid getting trapped in configuration loops...
        # print(f"Configure called at {time.monotonic() % 100 :.1f}")
        # print(f"  Master size {self.master.winfo_width()}x{self.master.winfo_height()}")
        # print(f"  Our size    {self.winfo_width()}x{self.winfo_height()}")
        if (self.winfo_width(), self.winfo_height()) not in self._recent_configured_whs:
            # print("  Redrawing...")
            # print("Configure called with size: ", self.winfo_width(), self.winfo_height())
            self.redraw()
            self._recent_configured_whs = self._recent_configured_whs[-2:] + [(self.winfo_width(), self.winfo_height())]
        # Set last configured wh (used to determine if zoom should be reset)
        self._last_configured_wh = (self.winfo_width(), self.winfo_height())

    def _on_mouse_drag_and_release(self, event: Event):

        if self._mouse_callback is not None and self._image_view_frame is not None:
            display_xy = self._event_to_display_xy(event)
            px, py = self._image_view_frame.display_xy_to_pixel_xy(display_xy)
            eat_event = self._mouse_callback(event, (int(px), int(py)))
            if eat_event:
                return

        is_drag = event.type == EventType.Motion
        is_release = event.type == EventType.ButtonRelease
        if is_drag:
            if self._drag_start_display_xy is None:
                self._drag_start_display_xy = self._event_to_display_xy(event)
            else:
                display_xy = self._event_to_display_xy(event)
                display_rel_xy = self._drag_start_display_xy[0]-display_xy[0], self._drag_start_display_xy[1]-display_xy[1]
                self._drag_start_display_xy = display_xy
                new_frame = self._image_view_frame.pan_by_display_shift(display_shift_xy=display_rel_xy, limit=True)
                self.set_image_frame(new_frame)
        elif is_release:
            self._drag_start_display_xy = None

    def _on_click(self, event: Event):
        # Never gets called for some reason
        if self._single_click_callback is not None:
            display_xy = self._event_to_display_xy(event)
            px, py = self._image_view_frame.display_xy_to_pixel_xy(display_xy)
            self._single_click_callback((int(px), int(py)))

    def _on_double_click(self, event: Event):
        if self._double_click_callback is not None:
            display_xy = self._event_to_display_xy(event)
            px, py = self._image_view_frame.display_xy_to_pixel_xy(display_xy)
            self._double_click_callback((int(px), int(py)))
        else:
            self.set_image_frame(self._image_view_frame.zoom_by(self._zoom_jump_factor, invariant_display_xy=self._event_to_display_xy(event)))

    def _event_to_display_xy(self, event: Event) -> Tuple[int, int]:
        return event.x, event.y
        # offset_x, offset_y = (self.winfo_width()-self.width)//2, (self.winfo_height()-self.height)//2
        # return event.x-offset_x, event.y-offset_y

    def _handle_mousewheel_event(self, event: Event):
        ''' Pan with mouse wheel '''
        if self._image_view_frame is None:
            return
        # print(f"Event with state {event.state}")
        # State "8" indicates that the command key is held
        # Make real state invariant to numlock
        real_state = event.state & ~0x0008 if is_windows_machine() else event.state
        # print(f"Mousewheel event: {event.delta}, type: {event.type}, state: {event.state}, real_state: {real_state}, serial: {event.serial}")
        modified_scroll_state = 4 if is_windows_machine() else 8  # Command-Scroll on mac, Control-Scroll on windows
        is_zoom_scroll = (self._zoom_scrolling_mode and real_state==0) or (not self._zoom_scrolling_mode and real_state==modified_scroll_state)
        # print(f"Got scroll event with state {event.state} and delta {event.delta}")
        # 3 is a good empirical fudge factor on windows.
        delta = copysign(3.0, event.delta) if is_windows_machine() else event.delta
        # This is horribly confusing... must be a way to simplify.
        new_frame = None
        if is_zoom_scroll:
            # if event.state == 0:  # Vertical
            if is_windows_machine() and self._zoom_scrolling_mode:
                delta = -delta
            is_zoom_in = delta < 0
            zoom_factor = (self._zoom_jump_factor - 1)*abs(delta)*self._mouse_scroll_speed + 1
            rzoom = zoom_factor if is_zoom_in else 1/zoom_factor
            new_frame = self._image_view_frame.zoom_by(relative_zoom=rzoom, invariant_display_xy=self._event_to_display_xy(event), max_zoom=self._max_zoom)

        else:
            is_vertical_pan = (self._zoom_scrolling_mode and real_state==modified_scroll_state) or (not self._zoom_scrolling_mode and real_state==0)
            is_horizontal_pan = real_state==1

            if is_windows_machine() or event.type == EventType.MouseWheel:
                if not is_windows_machine() and self._zoom_scrolling_mode:  # I am so confused
                    delta = -delta
                step = -delta * self._mouse_scroll_speed
                if is_vertical_pan:
                    new_frame = self._image_view_frame.pan_by_display_shift(display_shift_xy=(0, step))
                elif is_horizontal_pan:
                    new_frame = self._image_view_frame.pan_by_display_shift(display_shift_xy=(step, 0))
        if new_frame is not None:
            self.set_image_frame(new_frame)

    def set_image(self, image: BGRImageArray, redraw_now: bool = True):
        zoom_reset_needed = self._image is not None and self._image.shape != image.shape
        self._image = image
        if redraw_now:
            if zoom_reset_needed:
                self.reset_zoom()
            else:
                self.redraw()

    @contextmanager
    def hold_off_redraw_context(self):
        try:
            previous_state = self._hold_off_redraw
            self._hold_off_redraw = True
            yield
        finally:
            self._hold_off_redraw = previous_state
            self.redraw()

    def set_image_frame(self, image_view_frame: Optional[ImageViewInfo], fix_configuration: bool = True):
        if fix_configuration:
            self._is_configuration_still_being_negotiated = False
        self._image_view_frame = image_view_frame
        self.redraw()
        if self._after_view_change_callback is not None:
            self._after_view_change_callback(image_view_frame)

    def reset_zoom(self):
        self.set_image_frame(None)
        self._is_configuration_still_being_negotiated = True

    def zoom_to_pixel(self, pixel_xy: Tuple[int, int], zoom_level: float, adjust_to_boundary: bool = True):
        if self._image_view_frame is not None:
            new_frame = self._image_view_frame.zoom_to_pixel(pixel_xy=pixel_xy, zoom_level=zoom_level)
            if adjust_to_boundary:
                new_frame = new_frame.adjust_pan_to_boundary()
                # new_frame = new_frame.adjust_to_window(self.winfo_width(), self.winfo_height())
            self.set_image_frame(new_frame)

    def get_image_view_frame_or_none(self) -> Optional[ImageViewInfo]:
        return self._image_view_frame

    def rebind(self):
        # self.unbind_keys()
        bind_callbacks_to_widget(self._binding_dict, widget=self)
        # self.bind("<1>", lambda event: self.focus_set())
        self.bind()

    def unbind_keys(self):
        self.unbind_all(list(self._binding_dict.keys()))

    #
    # def zoom_by(self, zoom_factor: float, invariant_display_xy: Tuple[int, int]):
    #     self._image_view_frame = self._image_view_frame.zoom_by(zoom_factor, invariant_display_xy=invariant_display_xy)
    #     self.redraw()
    #
    # def zoom_out(self):
    #     self._image_view_frame = self._image_view_frame.zoom_out()
    #     self.redraw()

    def redraw(self):
        if self._hold_off_redraw:
            return
        # width, height =
        # height, width = (self.master.winfo_width(), self.master.winfo_height()) if self._first_pass else (self.winfo_width(), self.winfo_height())
        width, height = (self.winfo_width(), self.winfo_height())
        self._first_pass = False

        # if width == 1 or height == 1:  # Window has not yet rendered and does not have a size, so do not draw
        if width < 10 or height < 10:  # Window has not yet rendered and does not have a size, so do not draw
            # print(f"Skipping redraw on Zoomable Image Container because width={width}, height={height}")
            return
        if self._image is None:
            # print("Skipping becase no image yet")
            return
        # print(f'Redrawing image of shape {self._image.shape} with width={width}, height={height}')
        # aspect_ratio = self._image.shape[1] / self._image.shape[0]
        # self.height = self.height or round(self.width/aspect_ratio)
        # self.width = self.width or round(self.height*aspect_ratio)
        # width = self.width or self.master.winfo_width()

        # if height<2 or width<2:
        #     return
        # print(f"Redrawing Zoomable Image Container with width={width}, height={height}")
        if self._image_view_frame is None or self._is_configuration_still_being_negotiated:
            self._image_view_frame = ImageViewInfo.from_initial_view(window_disply_wh=(width, height), image_wh=(self._image.shape[1], self._image.shape[0]))
        else:

            relative_area_change = 1. if self._last_configured_wh is None else (width * height) / (self._last_configured_wh[0] * self._last_configured_wh[1])
            keep_old_zoom = 1/(1+self._rel_area_change_to_reset_zoom) <= relative_area_change <= 1+self._rel_area_change_to_reset_zoom
            self._image_view_frame = self._image_view_frame.adjust_frame_and_image_size(new_image_wh=(self._image.shape[1], self._image.shape[0]), new_frame_wh=(width, height))
            if not keep_old_zoom:
                self._image_view_frame = self._image_view_frame.zoom_out()
        disp_image = self._image_view_frame.create_display_image(self._image, gap_color=self._gap_color, scroll_fg_color = self._scrollbar_color)
        # disp_image = cv2.cvtColor(disp_image, cv2.COLOR_BGR2RGB)
        # im_resized = put_image_in_box(self._image, (self.winfo_width(), self.winfo_height()))
        # print(f"Zoomable Display image shape: {disp_image.shape}, with width={width}, height={height}")
        imagetk = ImageTk.PhotoImage(bgr_image_to_pil_image(disp_image), master=self)
        self.config(image=imagetk)
        # self.config(image=imagetk, width=width-self._margin_gap, height=height-self._margin_gap)
        self._imagetk = imagetk


if __name__ == "__main__":

    # Get sample image from web
    import requests
    import numpy as np
    from io import BytesIO

    img=smart_load_image("https://upload.wikimedia.org/wikipedia/commons/8/8a/%22Ride_the_elephant._Ride_Holy_Moses%22_LCCN2018647619.tif", use_cache=True)
    assert img is not None, "Could not load image"
    root = tk.Tk()
    root.geometry("800x600")

    frame = ZoomableImageFrame(root,
                               zoom_scrolling_mode=False,
                               # image=img,
                               # width=800,
                               # height=600,
                               # zoom_to_parent_initially=True,
                               # image_single_click_callback=lambda xy: print(f"Single click on {xy}"),
                               # image_double_click_callback=lambda xy: print(f"Double click on {xy}"),
                               )

    # frame = tk.Label(text='aaa')
    # frame.update_image(img)
    # frame = ZoomableImageTkFrame(root, image=img, zoom_to_parent_initially=True)
    frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    frame.set_image(img[:, :, ::-1])
    root.mainloop()
