#include <iostream>
#include "matrix.hpp"
#include <cassert>
#include "neural_network.hpp"
#include <vector>
#include "activation_functions.hpp"
#include <memory>

using act_func_ptr = std::unique_ptr<activation_function>;
int main() {
	std::vector<act_func_ptr> act_funcs;
	act_funcs.push_back(std::make_unique<selu>(1.0507, 1.6733));
	act_funcs.push_back(std::make_unique<selu>(1.0507, 1.6733));
	act_funcs.push_back(std::make_unique<softmax>());

	neural_network nn = neural_network
	(
		std::vector<size_t> {784, 100, 100, 10},
		std::move(act_funcs)
	);

	nn.sgd(
		25, 
		256,
		0.01,
		0.9,
		0.0002,
		"./data/fashion_mnist_train_vectors.csv",
		"./data/fashion_mnist_train_labels.csv"
	);

	nn.make_predictions(
		"./data/fashion_mnist_test_vectors.csv",
		"actualTestPredictions"
	);

	nn.make_predictions(
		"./data/fashion_mnist_train_vectors.csv",
		"trainPredictions"
	);

	return 0;
}
