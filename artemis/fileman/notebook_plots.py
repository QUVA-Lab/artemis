from IPython.core.display import HTML
from IPython.display import display
from fileman.notebook_utils import get_relative_link_from_local_path, get_relative_link_from_relative_path
from fileman.saving_plots import save_and_show, get_local_figures_dir, set_show_callback

__author__ = 'peter'


def always_link_figures(state = True, **link_and_show_arg):
    """
    Call this function to always
    :param state: True to display links to plots
    :param show: Set to False if you just want to print the link, and not actually show the plot.
    :param link_and_show_arg: Passed down to link_and_show/save_and_show
    """

    set_show_callback(lambda fig = None: link_and_show(fig=fig, **link_and_show_arg) if state else None)


def link_and_show(embed = False, **save_and_show_kwargs):
    """
    Use this function to show a plot in IPython Notebook, and provide a link to download the figure.
    See function save_and_show for parameters.
    """

    base_dir = get_local_figures_dir()
    full_figure_loc = save_and_show(print_loc = False, base_dir=base_dir, show = not embed, **save_and_show_kwargs)
    relative_link = get_relative_link_from_local_path(full_figure_loc)
    figure_folder_loc = get_relative_link_from_relative_path('figures')

    if embed:
        show_embedded_figure(relative_link)
    else:
        display(HTML("See <a href='%s' target='_blank'>this figure</a>.  See <a href='%s' target='_blank'>all figures</a>"
                % (relative_link, figure_folder_loc)))


def show_embedded_figure(relative_link):
    figure_folder_loc = get_relative_link_from_relative_path('figures')
    display(HTML("<iframe src='%s' width=600 height=460></iframe>" % (relative_link, )))
    display(HTML("See <a href='%s' target='_blank'>this figure</a>.  See <a href='%s' target='_blank'>all figures</a>"
            % (relative_link, figure_folder_loc)))
