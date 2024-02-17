#include "matrix.hpp"
#include <cassert>



int test_matrices() {
	matrix m1 = matrix(std::vector<std::vector<double>> {
		std::vector<double> {1, 2},
			std::vector<double> {3, 4}
	});
	matrix m2 = matrix(std::vector<std::vector<double>> {
		std::vector<double> {7, 7},
			std::vector<double> {7, 7}
	});
	matrix res = matrix(std::vector<std::vector<double>> {
		std::vector<double> {8, 9},
			std::vector<double> {10, 11}
	});

	assert(m1 + m2 == res);

	matrix s1 = matrix(std::vector<std::vector<double>> {
		std::vector<double> {1, 2},
			std::vector<double> {3, 4},
			std::vector<double> {5, 6}
	});
	matrix s2 = matrix(std::vector<std::vector<double>> {
		std::vector<double> {7, 8, 9, 10},
			std::vector<double> {11, 12, 13, 14}
	});
	matrix res2 = matrix(std::vector<std::vector<double>> {
		std::vector<double> {29, 32, 35, 38},
			std::vector<double> {65, 72, 79, 86},
			std::vector<double> {101, 112, 123, 134}
	});
	assert(s1 * s2 == res2);

	matrix s2t = matrix(std::vector<std::vector<double>> {
		std::vector<double> {7, 11},
			std::vector<double> {8, 12},
			std::vector<double> {9, 13},
			std::vector<double> {10, 14}

	});

	assert(s2.transpose() == s2t);

	matrix s1k = matrix(std::vector<std::vector<double>> {
		std::vector<double> {1, 2, 3},
			std::vector<double> {4, 5, 6}
	});
	matrix s2k = matrix(std::vector<std::vector<double>> {
		std::vector<double> {7, 8, 9},
			std::vector<double> {10, 11, 12},
			std::vector<double> {13, 14, 15},
			std::vector<double> {16, 17, 18}
	});
	matrix res2k = matrix(std::vector<std::vector<double>> {
		std::vector<double> {7, 8, 9, 14, 16, 18, 21, 24, 27 },
		std::vector<double> {10, 11, 12, 20, 22, 24, 30, 33, 36},
		std::vector<double> {13 , 14 , 15 , 26 , 28 , 30 , 39 , 42 , 45},
		std::vector<double> {16 , 17 , 18 , 32 , 34 , 36 , 48 , 51 , 54},
		std::vector<double> {28 , 32 , 36 , 35 , 40 , 45 , 42 , 48 , 54},
		std::vector<double> {40 , 44 , 48 , 50 , 55 , 60 , 60 , 66 , 72},
		std::vector<double> {52 , 56 , 60 , 65 , 70 , 75 , 78 , 84 , 90},
		std::vector<double> {64, 68, 72, 80, 85, 90, 96, 102, 108}
	});
	assert(s1k.kronecker_product(s2k) == res2k);

	return 0;
}
