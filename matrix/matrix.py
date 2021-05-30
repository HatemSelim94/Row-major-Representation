from __future__ import annotations
from typing import List, Tuple


class Matrix:
    def __init__(self, row: int, col: int,
                 data: List[float] = None) -> None:
        """
        Initializes a Matrix object.

        The storage order is row-major.

        Args:
            row: The number of rows.
            col: The number of columns.
            data: (Optional) A list containing the rows * cols matrix elements
                             in row-major order.
                             If this argument is not provided, the constructor
                             creates a list containing rows * cols zeros.
        """
        self._row = row
        self._col = col
        if data is None:
            data = [0]*row*col
        self._data = data.copy()

    def __add__(self, other: Matrix) -> Matrix:
        """
        The matrix plus operator

        Args:
            other: The Matrix instance to which this matrix(self) is added
        Returns:
            A Matrix instance containing the matrix addition "self + other".
        """
        return Matrix(self.row, self.col,
                      [elem1 + elem2 for elem1, elem2 in
                       zip(self.data, other.data)])

    def __sub__(self, other: Matrix) -> Matrix:
        """
        The matrix sub operator

        Args:
            other: The Matrix instance which will be subtracted from this matrix(self)
        Returns:
            A Matrix instance containing the matrix subtraction "self - other".
        """
        return Matrix(self.row, self.col,
                      [elem1 - elem2 for elem1, elem2 in
                       zip(self.data, other.data)])

    def __mul__(self, other: Matrix) -> Matrix:
        """
        The matrix product operator.

        Note: Python calls this function when the "mat1 @ mat2" syntax is used, where
              "mat1" is an n x m, "mat2" an m x q Matrix instance.

        Args:
            other: The Matrix instance with which to multiply this matrix (self) from
                   the right, i.e. self @ other.

        Returns:
            A Matrix instance containing the matrix product "self @ other".
        """
        result = Matrix(self.row, other.col)
        for i in range(result.row):
            for j in range(result.col):
                row_data = self.get_row(i)
                col_data = other.get_col(j)
                result.set_elem(i, j,
                                sum([elem1*elem2 for elem1, elem2
                                     in zip(row_data, col_data)]))
        return result

    def __getitem__(self, row_col: Tuple[int, int]) -> float:
        """
        Get the element at (row, col).

        Note: Python calls this function when the mat[row, col] syntax is used, where
              "mat" is a Matrix instance.

        Args:
            row_col: The 2-tuple containing the row, column indices.

        Returns:
            The element at position (row, col).
        """
        element_liner_id: int = self.linear_index(row_col[0], row_col[1])
        return self._data[element_liner_id-1]

    def __setitem__(self, row_col: Tuple[int, int], value: float) -> None:
        """
        Set the element at (row, col) to the specified value.

        Args:
            row_col: The 2-tuple containing the row, column indices.
            value: The value to assign.

        Note: Python calls this function when the mat[row, col] = value syntax is used,
        where "mat" is a Matrix instance.
        """
        element_liner_id: int = self.linear_index(row_col[0], row_col[1])
        self._data[element_liner_id-1] = value

    @property
    def col(self) -> int:
        """
        Get the number of cols.

        Returns:
            The number of cols.
        """
        return self._col

    @property
    def row(self) -> int:
        """
        Get the number of rows.

        Returns:
            The number of rows.
        """
        return self._row

    @property
    def data(self) -> List[float]:
        """
        Get the (row-major) data buffer.

        Returns:
            The data buffer.
        """

        return self._data

    def __get_row_ids(self, r) -> Tuple[int, int, int]:
        """
        Get the indices of elements of the r-th row
        Args:
            r: row number
        Returns:
            start, end, step form is used to describe the indices
        """
        return r*self.col, r*self.col+self.col, 1

    def __get_col_ids(self, c) -> Tuple[int, None, int]:
        """
        Get the indices of elements of the c-th column
        Args:
            c: column number
        Returns:
            start, end, step form is used to describe the indices
        """
        return c, None, self.col

    def linear_index(self, r: int, c: int) -> int:
        """
        Get the index to the (r,c)-th element in the data array.

        Returns:
            The linear index corresponding to the element at (r,c).
        """

        return self.col * (r-1) + c

    def get_elem(self, row, col) -> float:
        start, end, step = self.__get_row_ids(row)
        elem_id = list(range(start, end, step))[col]
        return self.data[elem_id]

    def set_elem(self, row, col, val) -> None:
        start, end, step = self.__get_row_ids(row)
        elem_id = list(range(start, end, step))[col]
        self.data[elem_id] = val

    def get_row(self, r: int) -> List[float]:
        """
        Get the rth row.

        Returns:
            The list of elements in the rth row.
        """
        start, end, step = self.__get_row_ids(r)
        return self.data[start:end:step]

    def get_col(self, c: int) -> List[float]:
        """
        Get the cth column.

        Returns:
            The list of elements in the cth column.
        """
        start, end, step = self.__get_col_ids(c)
        return self.data[start:end:step]

    def set_col(self, c: int, data: List[float]) -> None:
        """
        Set the cth column to the specified list of elements.

        Args:
            c: The column index.
            data: The list of elements to assign to the cth column.
        """
        start, end, step = self.__get_col_ids(c)
        self.data[start:end:step] = data

    def set_row(self, r: int, data: List[float]) -> None:
        """
        Set the rth row to the specified list of elements.

        Args:
            r: The row index.
            data: The list of elements to assign to the rth row.
        """
        start, end, step = self.__get_row_ids(r)
        self.data[start:end:step] = data

    def transpose(self) -> Matrix:
        """
        Transposes the matrix.

        Returns:
            A Matrix instance containing the transpose of this matrix.
        """
        mat_transpose = Matrix(self.col, self.row)
        [mat_transpose.set_row(i, self.get_col(i)) for i in range(self.col)]
        return mat_transpose


def inner(v: List[float], w: List[float]) -> float:
    """
    Computes the inner product of two vectors.

    Args:
        v: The first vector.
        w: The second vector.

    Returns:
        The inner product.
    """
    output: float = sum([i * j for i, j in zip(v, w)])
    return output


def outer(v: List[float], w: List[float]) -> Matrix:
    """
    Computes the outer product of two vectors.

    Args:
        v: The first vector.
        w: The second vector.

    Returns:
        The outer product matrix.
    """
    output_mat = Matrix(len(v), len(w))
    for i in range(output_mat.col):
        output_mat.set_col(i, [j * w[i] for j in v])
    return output_mat
