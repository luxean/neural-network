#pragma once

#include <vector>
#include <stdexcept>
#include <functional>
#include <numeric>
#include <cmath>
#include <cassert>
#include <algorithm>

int test_matrices();

class matrix {

	size_t _rows;
	size_t _cols;
	std::vector<double> _values;


public:
	size_t rows() const {
		return _rows;
	}


	size_t cols() const {
		return _cols;
	}


	size_t size() const {
		return _values.size();
	}


	double operator()(size_t row, size_t col) const {
		return _values[_cols * row + col];
	}


	double& operator()(size_t row, size_t col) {
		return _values[_cols * row + col];
	}


	double operator[](size_t idx) const {
		return _values[idx];
	}


	double& operator[](size_t idx) {
		return _values[idx];
	}


	matrix():
		_rows(0),
		_cols(0)
	{}


	matrix(size_t rows, size_t cols):
		_rows(rows),
		_cols(cols),
		_values(std::vector<double>(rows * cols))
	{}


	matrix(std::vector<double> values):
		_rows(1),
		_cols(values.size()),
		_values(values)
	{}


	matrix(const std::vector<std::vector<double>>& values):
		matrix(values.size(), values[0].size())
	{
		for (size_t row = 0; row < _rows; ++row) {
			for (size_t col = 0; col < _cols; ++col) {
				(*this)(row, col) = values[row][col];
			}
		}
	}


	template <typename init_func_t>
	matrix(size_t rows, size_t cols, init_func_t init_func):
		matrix(rows, cols)
	{
		for (double& val : _values) {
			val = init_func();
		}
	}


	matrix& add_val(double val) {
		if (_rows != 1) {
			throw std::out_of_range("Matrix is not a row vector");
		}

		_values.push_back(val);
		_cols += 1;
		return *this;
	}


	matrix& remove_last() {
		if (_rows != 1) {
			throw std::out_of_range("Matrix is not a row vector");
		}

		_values.pop_back();
		_cols -= 1;
		return *this;
	}


private:
	template <typename op_func_t>
	matrix& operator_base(const matrix& rhs, op_func_t op_func) {
		if (_rows != rhs._rows || _cols != rhs._cols) {
			throw std::out_of_range("Matrix dimensions do not match");
		}

		for (size_t idx = 0; idx < _values.size(); ++idx) {
			op_func(_values[idx], rhs[idx]);
		}

		return *this;
	}


public:
	friend bool operator==(const matrix& lhs, const matrix& rhs) {
		if (lhs._rows != rhs._rows || lhs._cols != rhs._cols) {
			return false;
		}

		for (size_t idx = 0; idx < lhs.size(); ++idx) {
			if (lhs[idx] != rhs[idx]) {
				return false;
			}
		}

		return true;
	}


	matrix& operator+=(const matrix& rhs) {
		return operator_base(rhs, [](double& x, double y) {return x += y; });
	}


	matrix& operator-=(const matrix& rhs) {
		return operator_base(rhs, [](double& x, double y) {return x -= y; });
	}


	friend matrix operator+(matrix lhs, const matrix& rhs) {
		lhs += rhs;
		return lhs;
	}


	friend matrix operator-(matrix lhs, const matrix& rhs) {
		lhs -= rhs;
		return lhs;
	}


	matrix& operator*=(double scalar) {
		for (double& val : _values) {
			val *= scalar;
		}

		return *this;
	}


	friend matrix operator*(matrix lhs, double scalar) {
		return (lhs *= scalar);
	}


	friend matrix operator*(double scalar, matrix rhs) {
		return (rhs *= scalar);
	}


	friend matrix operator*(const matrix& lhs, const matrix& rhs) {
		if (lhs._cols != rhs._rows) {
			throw std::out_of_range("Matrix dimensions do not match");
		}

		size_t rows = lhs._rows;
		size_t cols = rhs._cols;

		matrix res = matrix(rows, cols);

		for (size_t row = 0; row < rows; ++row) {

			for (size_t col = 0; col < cols; ++col) {
				double sum = 0;

				for (size_t i = 0; i < rhs._rows; ++i) {
					sum += lhs(row, i) * rhs(i, col);
				}

				res(row, col) = sum;
			}
		}

		return res;
	}


	matrix transpose() const {
		matrix result = matrix(_cols, _rows);

		for (size_t row = 0; row < _rows; ++row) {
			for (size_t col = 0; col < _cols; ++col) {
				result(col, row) = (*this)(row, col);
			}
		}
		
		return result;
	}


	matrix kronecker_product(const matrix& rhs) const {
		matrix result = matrix(_rows * rhs._rows, _cols * rhs._cols);

		for (size_t row1 = 0; row1 < _rows; ++row1) {
			for (size_t col1 = 0; col1 < _cols; ++col1) {
				for (size_t row2 = 0; row2 < rhs.rows(); ++row2) {
					for (size_t col2 = 0; col2 < rhs.cols(); ++col2) {
						result(rhs.rows() * row1 + row2, rhs.cols() * col1 + col2) = 
							(*this)(row1, col1) * rhs(row2, col2);
					}
				}
			}
		}

		return result;
	}


	matrix& hadamart_product(const matrix& rhs) {
		return operator_base(rhs, [](double& x, double y) {return x *= y;});
	}


	double max() const {
		return *max_element(_values.begin(), _values.end());
	}


	std::vector<double>::iterator begin() {
		return _values.begin();
	}


	std::vector<double>::iterator end() {
		return _values.end();
	}

};
