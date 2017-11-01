from prettytable.prettytable import PrettyTable

_TEST_TABLE_ROWS = []
_TEST_TABLE_ROWS.append(['joe', 'fulcndncdilsancduilsabvfdalcbsahlb dsahkl bfgfdsgfds gfdsg gfds gfds gfds gfds gfds gfdsgfdsdhslb fda', 5.23432])
_TEST_TABLE_ROWS.append(['jill', 'dsavas fdsnjvl fdsjhlv fdhslv fdhjlsfdgbv dsahgfdba', 5.23432])

_TEST_TABLE_HEADERS = ['Name', 'Description', 'Number']


def test_limited_width():

    table = PrettyTable(_TEST_TABLE_ROWS, max_table_width=80, field_names=_TEST_TABLE_HEADERS, max_table_width_style='robin_hood')
    print table

    width = str(table).index('\n')
    print width


if __name__ == '__main__':
    test_limited_width()
