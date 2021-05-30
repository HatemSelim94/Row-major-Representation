import unittest
from matrix import Matrix


class TestMatrix(unittest.TestCase):

    row = 2
    col = 2
    data1 = [0] * col * row
    data2 = [1, 2, 3, 4]
    data3 = list(range(1, 7))
    mat1 = Matrix(row, col)
    mat2 = Matrix(row, col, data2)
    mat3 = Matrix(3, 2, data3)

    def test_init(self):
        # default initialization
        self.assertEqual(self.mat1.row, self.row)
        self.assertEqual(self.mat1.col, self.col)
        self.assertListEqual(self.mat1.data, self.data1)

        # custom initialization
        self.assertEqual(self.mat2.row, self.row)
        self.assertEqual(self.mat2.col, self.col)
        self.assertListEqual(self.mat2.data, self.data2)

    def test_add(self):
        mat = self.mat1 + self.mat2
        self.assertListEqual(mat.data, self.mat2.data)

    def test_mul(self):
        mat = self.mat3 * self.mat2
        self.assertEqual(mat.row, self.mat3.row)
        self.assertEqual(mat.col, self.mat2.col)
        self.assertListEqual(mat.data, [7, 10, 15, 22, 23, 34])

    def test_get_row(self):
        row = self.mat3.get_row(2)
        self.assertListEqual(row, [5, 6])

    def test_get_col(self):
        col = self.mat3.get_col(1)
        self.assertListEqual(col, [2, 4, 6])

    def test_set_row(self):
        mat = Matrix(3, 3, list(range(1, 10)))
        mat.set_row(2, [0, 0, 0])
        row1 = mat.get_row(0)
        row3 = mat.get_row(2)
        self.assertListEqual(row3, [0, 0, 0])
        self.assertListEqual(row1, [1, 2, 3])

    def test_set_col(self):
        mat = Matrix(3, 3, list(range(1, 10)))
        mat.set_col(2, [0, 0, 0])
        col1 = mat.get_col(0)
        col3 = mat.get_col(2)
        self.assertListEqual(col3, [0, 0, 0])
        self.assertListEqual(col1, [1, 4, 7])

    def test_get_element(self):
        elem = self.mat3.get_elem(2, 1)
        self.assertEqual(elem, 6)

    def test_set_element(self):
        mat = Matrix(3, 3, list(range(1, 10)))
        val = 3
        mat.set_elem(2, 2, val)
        mat_val = mat.get_elem(2, 2)
        self.assertEqual(mat_val, val)

    def test_transpose(self):
        mat_transposed = self.mat3.transpose()
        self.assertEqual(mat_transposed.col, self.mat3.row)
        self.assertEqual(mat_transposed.row, self.mat3.col)
        self.assertListEqual(mat_transposed.get_col(1),
                             self.mat3.get_row(1))
        self.assertListEqual(mat_transposed.get_row(0),
                             self.mat3.get_col(0))

    def test_getitem(self):
        a = self.mat3[3, 2]
        self.assertEqual(a, 6)

    def test_setitem(self):
        mat = Matrix(3, 3, list(range(1, 10)))
        self.assertEqual(mat.data[2], mat[1, 3])

    def test_linear_index(self):
        index = self.mat3.linear_index(3, 2)
        index1 = self.mat2.linear_index(2, 2)
        self.assertEqual(index, 6)
        self.assertEqual(index1, 4)


if __name__ == '__main__':
    unittest.main()
