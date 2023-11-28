import tkinter as tk
from typing import Union, Mapping, Optional

import numpy as np

from artemis.general.custom_types import BGRImageArray
from artemis.plotting.tk_utils.alternate_zoomable_image_view import ZoomableImageFrame
from artemis.plotting.tk_utils.tk_utils import hold_tkinter_root_context
from artemis.plotting.tk_utils.ui_utils import RespectableButton


class SwitchableImageViewer(tk.Frame):

    def __init__(self, master: tk.Frame):
        tk.Frame.__init__(self, master=master)
        # self.columnconfigure(0, weight=1)
        # self.columnconfigure(1, weight=1)
        # self.rowconfigure(0, weight=0)
        # self.rowconfigure(1, weight=1)

        self.switch_button = RespectableButton(
            master=self,
            text='<State>',
            command=lambda: self.switch_view()
        )
        # Right arrow to switch forward, left to switch back
        self.bind_all('<Right>', lambda e: self.switch_view(1))
        self.bind_all('<Left>', lambda e: self.switch_view(-1))
        # self.switch_button.grid(row=0, column=0, sticky=tk.NSEW)
        self.switch_button.pack()
        self.image_view = ZoomableImageFrame(self)
        # self.image_view.grid(row=1, column=0, sticky=tk.NSEW)
        self.image_view.pack(fill=tk.BOTH, expand=tk.YES)

        self._current_images: Mapping[str, BGRImageArray] = {}
        self._active_panel: Optional[str] = None

    def _panel_name_to_ix(self, name: str) -> int:
        return list(self._current_images.keys()).index(name)

    def switch_view(self, increment: int = 1):
        current_view_ix = self._panel_name_to_ix(self._active_panel) if self._active_panel is not None else -increment
        next_view_ix = (current_view_ix + increment) % len(self._current_images)
        panel = list(self._current_images.keys())[next_view_ix]
        self.set_active_panel(panel)

    def set_active_panel(self, name: str):
        self._active_panel = name
        view_ix = self._panel_name_to_ix(name)
        self.switch_button.configure(text=f"{name} ({view_ix+1}/{len(self._current_images)})")
        self._redraw()

    def set_images(self, images: Mapping[str, BGRImageArray], active_panel: Optional[str] = None):


        self._current_images = images
        if active_panel is not None:
            self.set_active_panel(active_panel)
        elif self._active_panel is None:
            self.set_active_panel(next(iter(images.keys())))
        self._redraw()

    def _redraw(self):
        if self._current_images and self._active_panel is not None:
            self.image_view.set_image(self._current_images[self._active_panel])
        # else:
        #     self.image_view.set_image(image=)



def tkshow(
        images: Union[BGRImageArray, Mapping[str, BGRImageArray]],
        hang_time: Optional[int] = None,
        title: Optional[str] = None,
):

    if isinstance(images, np.ndarray):
        images = {'image': images}

    #
    # for name, image in images.items():
    #     if isinstance(image, tf.Tensor):
    #         images[name] = image.numpy()

    with hold_tkinter_root_context() as root:
        # Set geometry
        root.geometry("1200x800")
        root.title(title or 'Left/Right to switch views, Z/X/C to zoom, W/A/S/D to pan, Escape to close')

        # top_level = tk.Toplevel()
        # viewer = ZoomableImageFrame(root)
        viewer = SwitchableImageViewer(root)
        viewer.pack(fill=tk.BOTH, expand=tk.YES)
        viewer.set_images(images)
        # viewer.set_image(images['image'])

        # Close on Escape
        viewer.bind_all('<Escape>', lambda e: viewer.master.destroy())

        if hang_time is not None:
            root.after(int(hang_time)*1000, lambda: viewer.master.destroy())

        viewer.mainloop()
        return viewer

