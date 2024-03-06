from enum import Enum

from artemis.plotting.tk_utils.tk_basic_utils import hold_tkinter_root_context
from video_scanner.app_utils.constants import ThemeColours


def test_button_panel(wait: bool = False):

    with hold_tkinter_root_context() as root:
        from artemis.plotting.tk_utils.ui_utils import ButtonPanel
        bp = ButtonPanel(root, max_buttons_before_expand=3, as_row=False)

        bp.add_button('Button 1', lambda: print('Button 1'))
        bp.add_button('Button 2', lambda: print('Button 2'))
        bp.add_button('Button 3', lambda: print('Button 3'))
        bp.add_button('Button 4', lambda: print('Button 4'))
        bp.add_button('Button 5', lambda: print('Button 5'))
        bp.pack()
        root.update()
        if wait:
            root.wait_window()

        bp.destroy()
        root.update()


def test_multi_state_toggle(manual: bool = False):
    with hold_tkinter_root_context() as root:
        from artemis.plotting.tk_utils.ui_utils import MultiStateToggle

        class States(Enum): # You can also use an Enum here
            STATE_1 = 'State 1'
            STATE_2 = 'State 2'
            STATE_3 = 'State 3'

        mst = MultiStateToggle(root, States, on_state_change_callback=lambda s: print(f'Switched to {s.value}'))
        mst.pack()
        root.update()

        assert mst.get_state() == States.STATE_1
        mst.set_state(States.STATE_2)
        assert mst.get_state() == States.STATE_2
        if manual:
            root.wait_window()


if __name__ == '__main__':
    # test_button_panel(wait=True)
    test_multi_state_toggle(manual=True)
