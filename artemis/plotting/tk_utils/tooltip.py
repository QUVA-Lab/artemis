import tkinter as tk
from typing import Tuple

from artemis.plotting.tk_utils.constants import ThemeColours


class ToolTip(object):

    def __init__(self, widget: tk.Widget, background: str = '#ffffe0', borderwidth: int = 1, font: Tuple[str, str, str] = ('tahom', '10', 'normal'), justify: str = 'left', relief: str = 'solid', text: str = '', anchor=tk.NW):

        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        self._background = background
        self._borderwidth = borderwidth
        self._font = font
        self._font_size = int(font[1])
        self._justify = justify
        self._relief = relief
        self._text = text
        self._below_cursor = 'n' in anchor.lower()
        self._right_of_cursor = 'w' in anchor.lower()

    def showtip(self):
        "Display text in tooltip window"
        # self.text = text
        if self.tipwindow or not self._text:
            return
        x, y, cx, cy = self.widget.bbox("insert")

        self.tipwindow = tw = tk.Toplevel(self.widget)
        text_width = len(self._text)*self._font_size*0.4
        x_offset = 57
        # print(f'Tip window width {self.tipwindow.winfo_width()}')
        # is_on_right_edge = x_offset + x + text_width > self.tipwindow.winfo_width()
        right_of_cursor = self._right_of_cursor

        x = x + self.widget.winfo_rootx() + (x_offset if right_of_cursor else -x_offset - text_width)
        y = y + cy + self.widget.winfo_rooty() + (27 if self._below_cursor else -27 - self._font_size)




        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        # Anchor window to left of cursor
        # if self._right_of_cursor:
        #     tw.wm_geometry("+%d+%d" % (x+len(self._text), y))
        label = tk.Label(tw, text=self._text, justify=tk.RIGHT,
                      background=self._background, relief=tk.SOLID, borderwidth=1,
                      font=self._font)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


def show_toast_as_temporary_tooltip(text: str, widget: tk.Widget, duration_ms: int = 2000):
    tooltip = ToolTip(widget, text=text, anchor=tk.NW, background=ThemeColours.TOOLTIP_BACKGROUND, borderwidth=1, font=('tahom', '14', 'normal'), justify='left', relief='solid')
    tooltip.showtip()
    widget.after(duration_ms, tooltip.hidetip)


def create_tooltip(widget, text, anchor=tk.NW, **kwargs):
    toolTip = ToolTip(widget, text=text, anchor=anchor, **kwargs)

    def enter(event):
        toolTip.showtip()

    def leave(event):
        toolTip.hidetip()

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)