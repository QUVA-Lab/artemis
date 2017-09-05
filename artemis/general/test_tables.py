from artemis.general.should_be_builtins import bad_value
from artemis.general.tables import build_table


def test_build_table(show_table=True):

    def lookup_function(prisoner_a_choice, prisoner_b_choice):
        total_utility = \
            2 if prisoner_a_choice=='cooperate' and prisoner_b_choice=='cooperate' else \
            3 if prisoner_a_choice != prisoner_b_choice else \
            4 if prisoner_b_choice=='betray' and prisoner_a_choice=='betray' \
            else bad_value((prisoner_a_choice, prisoner_b_choice))
        return total_utility

    rows = build_table(lookup_function, row_categories=['cooperate', 'betray'], column_categories=['cooperate', 'betray'])

    assert rows == [[' ', 'Cooperate', 'Betray'], ['Cooperate', 2, 3], ['Betray', 3, 4]]

    if show_table:
        import tabulate
        print(tabulate.tabulate(rows))


if __name__ == '__main__':
    test_build_table()
