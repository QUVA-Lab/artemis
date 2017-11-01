# import prettytable
from prettytable.prettytable_old import PrettyTable

import textwrap

from artemis.general.mymath import clip_to_sum
from artemis.general.should_be_builtins import izip_equal


def wrap_rows(rows, max_width, column_pad = 3, total_pad = 2):

    n_cols = len(rows[0])
    total_width = max_width - total_pad - n_cols*column_pad
    max_column_widths = [max(len(str(el)) for el in col) for col in zip(*rows)]

    # Ok, we gotta bring this down so that max_width == total width
    new_column_widths = clip_to_sum(max_column_widths, total=total_width)
    if sum(max_column_widths)<=max_width:
        return rows
    else:
        max_column_width = None if max_width is None else (max_width-total_pad)/(n_cols+column_pad)
        new_rows = []
        for row in rows:
            new_row = [textwrap.fill(d, width=max_col_wid) if len(str(d))>max_column_width else d for d, max_col_wid in izip_equal(row, new_column_widths)]
            new_rows.append(new_row)
        return new_rows


def table_str(rows, headers=None, max_width=None, hrules ='frame', vrules ='frame'):
    # if width is not None:
    #     if headers is None:
    #         rows = wrap_rows(rows, max_width=width)
    #     else:
    #         headers_and_rows = wrap_rows([headers]+rows, max_width=width)
    #         headers, rows = headers_and_rows[0], headers_and_rows[1:]

    # str2enum = {
    #     'all': prettytable.ALL,
    #     'frame': prettytable.prettytable.FRAME,
    #     'none': prettytable.NONE,
    #     'header': prettytable.HEADER,
    #     }
    tab = PrettyTable(rows=rows, header = headers is not None, hrules=hrules, vrules=vrules, max_table_width=max_width if max_width is not None else 0)
    return str(tab)

#
# def squeeze_table(table_str, max_width):
#
#
#
# def extract_table_contents(contents, )