import numpy as np
from tabulate import tabulate
from artemis.general.dead_easy_ui import DeadEasyUI


class TableExplorerUI(DeadEasyUI):

    def __init__(self, table_data, col_headers=None, row_headers=None, col_indices=None, row_indices = None):

        assert all(len(r)==len(table_data[0]) for r in table_data), "All rows of table data must have the same length.  Got lengths: {}".format([len(r) for r in table_data])
        table_data = np.array(table_data, dtype=object)
        assert table_data.ndim==2, "Table must consist of 2d data"

        assert col_headers is None or len(col_headers)==table_data.shape[1]
        assert row_headers is None or len(row_headers)==table_data.shape[0]

        self._table_data = table_data
        self._col_indices = np.array(col_indices) if col_indices is not None else None
        self._row_indices = np.array(row_indices) if row_indices is not None else None
        self._col_headers = np.array(col_headers) if col_headers is not None else None
        self._row_headers = np.array(row_headers) if row_headers is not None else None
        self._old_data_buffer = []

    @property
    def n_rows(self):
        return self._table_data.shape[0]

    @property
    def n_cols(self):
        return self._table_data.shape[1]

    def _get_full_table(self):
        n_total_rows = 1 + int(self._col_headers is not None) + self._table_data.shape[0]
        n_total_cols = 1 + int(self._row_headers is not None) + self._table_data.shape[1]
        table_data = np.empty((n_total_rows, n_total_cols), dtype=object)
        table_data[:2, :2] = ''
        table_data[0, -self.n_cols:] = self._col_indices if self._col_indices is not None else ['{}'.format(i) for i in range(1, self.n_cols+1)]
        table_data[-self.n_rows:, 0] = self._row_indices if self._row_indices is not None else ['{}'.format(i) for i in range(1, self.n_rows+1)]
        if self._col_headers is not None:
            table_data[1, -self.n_cols:] = self._col_headers
        if self._row_headers is not None:
            table_data[-self.n_rows:, 1] = self._row_headers
        table_data[-self.n_rows:, -self.n_cols:] = self._table_data
        return table_data

    def _get_menu_string(self):
        table_str = tabulate(self._get_full_table())
        return '{}\n'.format(table_str)

    def _backup(self):
        self._old_data_buffer.append((self._table_data, self._row_headers, self._row_indices, self._col_headers, self._col_indices))

    def undo(self):
        if len(self._old_data_buffer)==0:
            print("Can't undo, no history")
        else:
            self._table_data, self._row_headers, self._row_indices, self._col_headers, self._col_indices = self._old_data_buffer.pop()

    def _parse_indices(self, user_range):
        if isinstance(user_range, str):
            user_range = user_range.split(',')
        return [int(i)-1 for i in user_range]

    def _reindex(self, row_ixs=None, col_ixs=None):
        self._backup()
        if row_ixs is not None:
            self._table_data = self._table_data[row_ixs, :]
            if self._row_headers is not None:
                self._row_headers = self._row_headers[row_ixs]
            if self._row_indices is not None:
                self._row_indices = self._row_indices[row_ixs]
        if col_ixs is not None:
            self._table_data = self._table_data[:, col_ixs]
            if self._col_headers is not None:
                self._col_headers = self._col_headers[col_ixs]
            if self._col_indices is not None:
                self._col_indices = self._col_indices[col_ixs]

    def delcol(self, user_range):
        self._reindex(col_ixs=[i for i in range(self.n_cols) if i not in self._parse_indices(user_range)])

    def delrow(self, user_range):
        self._reindex(row_ixs=[i for i in range(self.n_rows) if i not in self._parse_indices(user_range)])

    def shufrows(self, user_range):
        indices = self._parse_indices(user_range)
        self._reindex(row_ixs=indices + [i for i in range(self.n_rows) if i not in indices])

    def shufcols(self, user_range):
        indices = self._parse_indices(user_range)
        self._reindex(col_ixs=indices + [i for i in range(self.n_cols) if i not in indices])

    def sortrows(self, by_cols=None, shuffle_cols=True):
        key_order_indices = self._parse_indices(by_cols) if by_cols is not None else range(self.n_cols)

        sorting_data = self._table_data[:, key_order_indices[::-1]].copy()
        for col in range(sorting_data.shape[1]):
            if np.mean([np.isreal(x) for x in sorting_data[:, col]]) % 1 != 0:  # Indicating not some numeric and some non-numeric data
                sorting_data[:, col] = [(not np.isreal(x), x) for x in sorting_data[:, col]]

        indices = np.lexsort(sorting_data.T)
        self._reindex(row_ixs=indices)
        if shuffle_cols:
            self.shufcols(by_cols)

    def sortcols(self, by_rows=None, shuffle_rows=True):
        key_order_indices = self._parse_indices(by_rows) if by_rows is not None else range(self.n_rows)
        indices = np.lexsort(self._table_data[key_order_indices[::-1], :])
        self._reindex(col_ixs=indices)
        if shuffle_rows:
            self.shufrows(by_rows)


if __name__ == '__main__':

    ui = TableExplorerUI(
        col_headers=['param1', 'size', 'cost'],
        row_headers=['exp1', 'exp2', 'exp3'],
        table_data= [[4, 'Bella', 100], [3, 'Abe', 120], [4, 'Clarence', 117]],
    )
    ui.launch()
