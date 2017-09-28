import itertools

from six import string_types

from artemis.general.should_be_builtins import all_equal_deprecated, all_equal
from six.moves import xrange


def build_table(lookup_fcn, row_categories, column_categories, clear_repeated_headers = True, prettify_labels = True,
            row_header_labels = None, remove_unchanging_cols = False):
    """
    Build the rows of a table.  You can feed these rows into tabulate to generate pretty things.

        Example (requires installing tabulate (pip install tabulate):
        For the table of total utility in prisoner's dillema (see https://en.wikipedia.org/wiki/Prisoner%27s_dilemma):

        def lookup_function(prisoner_a_choice, prisoner_b_choice):
            total_utility = \
                2 if prisoner_a_choice=='cooperate' and prisoner_b_choice=='cooperate' else \
                3 if prisoner_a_choice != prisoner_b_choice else \
                4 if prisoner_b_choice=='betray' and prisoner_a_choice=='betray' \
                else bad_value((prisoner_a_choice, prisoner_b_choice))
            return total_utility

        rows = build_table(lookup_function, row_categories=['cooperate', 'betray'], column_categories=['cooperate', 'betray'])
        print tabulate.tabulate(rows)

        ---------  ---------  ------
                   Cooperate  Betray
        Cooperate  2          3
        Betray     3          4
        ---------  ---------  ------

        See more examples in test_tables.

    :param lookup_fcn: A function of the form:
        data = lookup_fcn(row_info, column_info)
        Where:
            row_info is a tuple of data identifying the row.
            col_info is a tuple of data identifying the column
    :param row_categories: A list<list<str>> of categories that will make up the rows
    :param column_categories: A list<list<str>> of catefories that will make up the columns
    :param clear_repeated_headers: True to not repeat row headers.
    :param row_header_labels: Labels for the row headers.
    :param remove_unchanging_cols: Remove columns for which all d
    :return: A list of rows.
    """
    # Now, build that table!
    single_row_category = all(isinstance(c, string_types) for c in row_categories)
    single_column_category = all(isinstance(c, string_types) for c in column_categories)

    if single_row_category:
        row_categories = [row_categories]
    if single_column_category:
        column_categories = [column_categories]
    if row_header_labels is not None:
        assert len(row_header_labels) == len(row_categories)
    rows = []
    column_headers = list(zip(*itertools.product(*column_categories)))
    for i, c in enumerate(column_headers):
        row_header = row_header_labels if row_header_labels is not None and i==len(column_headers)-1 else [' ']*len(row_categories)
        row = row_header+(blank_out_repeats(c) if clear_repeated_headers else list(c))
        rows.append([prettify_label(el) for el in row] if prettify_labels else row)
    last_row_data = [' ']*len(row_categories)
    for row_info in itertools.product(*row_categories):
        if clear_repeated_headers:
            row_header, last_row_data = zip(*[(h, h) if lh!=h else (' ', lh) for h, lh in zip(row_info, last_row_data)])
        else:
            row_header = row_info
        if prettify_labels:
            row_header = [prettify_label(str(el)) for el in row_header]
        data = [lookup_fcn(row_info[0] if single_row_category else row_info, column_info[0] if single_column_category else column_info) for column_info in itertools.product(*column_categories)]
        rows.append(list(row_header) + data)
    assert all_equal((len(r) for r in rows)), "All rows must have equal length.  They now have lengths: {}".format([len(r) for r in rows])

    if remove_unchanging_cols:
        for col_ix in range(len(rows[0]))[::-1]:
            if all_equal([row[col_ix] for row in rows[len(column_headers):]]):
                for row in rows:
                    del row[col_ix]
    return rows


def prettify_label(label):
    return (label[0].upper() + label[1:]).replace('_', ' ')


def blank_out_repeats(sequence, replace_with=' '):

    new_sequence = list(sequence)
    for i in xrange(len(new_sequence)-1, 0, -1):
        if new_sequence[i]==new_sequence[i-1]:
            new_sequence[i] = replace_with
    return new_sequence
