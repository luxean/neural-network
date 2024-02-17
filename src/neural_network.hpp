#pragma once

#include <vector>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <string>
#include <sstream>
#include <random>
#include <numeric>
#include <cmath>
#include <memory>
#include "io.hpp"
#include "matrix.hpp"
#include "activation_functions.hpp"



class neural_network {
	using act_func_ptr = std::unique_ptr<activation_function>;

	size_t _layers;
	std::vector<matrix> _weights;
	std::vector<act_func_ptr> _act_functions;

	std::vector<matrix> _outputs;
	std::vector<matrix> _partial_der_potentials;
	std::vector<matrix> _act_func_der_potentials;


public:
	neural_network(const std::vector<size_t>& sizes, std::vector<act_func_ptr> act_functions):
		_layers(sizes.size()),
		_weights(_layers - 1),
		_act_functions(std::move(act_functions)),

		_outputs(_layers),
		_partial_der_potentials(_layers - 1),
		_act_func_der_potentials(_layers - 2)
	{
		std::random_device rd{};
		std::mt19937 gen{ rd() };

		for (size_t i = 0; i < _weights.size(); ++i) {
			std::normal_distribution<double> nd { 
				0, 
				std::sqrt(_act_functions[i]->weight_init_variance(sizes[i] + 1, sizes[i + 1])) 
			};

			_weights[i] = matrix(sizes[i] + 1, sizes[i + 1], [&nd, &gen]() {return nd(gen); });
		}
	}


private:
	matrix& feed_forward(matrix input) {
		_outputs[0] = std::move(input.add_val(1));

		for (size_t i = 0; i < _layers - 2; ++i) {
			matrix potential = _outputs[i] * _weights[i];
			_outputs[i + 1] = _act_functions[i]->apply(static_cast<const matrix> (potential)).add_val(1);

			_act_func_der_potentials[i] = _act_functions[i]->derivative(std::move(potential));
		}

		matrix potential = _outputs[_layers - 2] * _weights[_layers - 2];
		_outputs[_layers - 1] = _act_functions[_layers - 2]->apply(static_cast<const matrix> (potential)).add_val(1);

		return _outputs[_layers - 1].remove_last();
	}
	

	std::vector<matrix> back_propagation(const matrix& desired_outputs) {
		std::vector<matrix> nabla_weights(_layers - 1);

		int layer = _layers - 2;
		_partial_der_potentials[layer] = desired_outputs - _outputs[_layers - 1];

		for (; layer > 0; --layer) {
			_partial_der_potentials[layer - 1] = (_weights[layer] * _partial_der_potentials[layer].transpose())
											.transpose().remove_last();

			_partial_der_potentials[layer - 1].hadamart_product(_act_func_der_potentials[layer - 1]);

			nabla_weights[layer] = _outputs[layer].transpose().kronecker_product(_partial_der_potentials[layer]);
		}

		nabla_weights[0] = _outputs[0].transpose().kronecker_product(_partial_der_potentials[0]);

		return nabla_weights;
	}


	std::vector<matrix> example_gradient(const matrix& input, const matrix& desired_outputs) {
		feed_forward(input);
		return back_propagation(desired_outputs);
	}


private:
	template <typename func_t>
	static void update_vector(std::vector<matrix>& v1, std::vector<matrix>& v2, func_t func) {
		for (size_t i = 0; i < v1.size(); ++i) {
			func(v1[i], v2[i]);
		}
	}


	template <typename func_t>
	static void update_vector(std::vector<matrix>& v1, const std::vector<matrix>& v2, func_t func) {
		for (size_t i = 0; i < v1.size(); ++i) {
			func(v1[i], v2[i]);
		}
	}


public:
	int predict(const matrix& example) {
		matrix out = feed_forward(example);
		double max = out[0];
		int max_idx = 0;

		for (size_t i = 1; i < out.size(); ++i) {
			if (out[i] > max) {
				max = out[i];
				max_idx = i;
			}
		}

		return max_idx;
	}


	void make_predictions(const std::string& examples_filename, const std::string& output_filename) {
		std::vector<matrix> test_examples = load_examples(examples_filename);

		std::ofstream output(output_filename);

		for (matrix& example : test_examples) {
			output << predict(example) << '\n';
		}
	}


	void sgd(int num_epochs, size_t batch_size, 
			 double learning_rate, double influence, double weight_decay, 
			 const std::string& examples_filename, const std::string& labels_filename) {

		std::vector<matrix> examples = load_examples(examples_filename);
		std::vector<matrix> labels = load_labels(labels_filename);

		std::vector<size_t> indeces(labels.size());
		std::iota(std::begin(indeces), std::end(indeces), 0);

		auto rng = std::default_random_engine();

		std::vector<matrix> previous_update(_weights.size());
		for (size_t i = 0; i < _weights.size(); ++i) {
			previous_update[i] = matrix(_weights[i].rows(), _weights[i].cols());
		}

		for (;num_epochs > 0; --num_epochs) {
			std::shuffle(std::begin(indeces), std::end(indeces), rng);

			for (size_t batch = 0; batch < examples.size(); batch += batch_size) {
				std::vector<matrix> gradient = example_gradient(examples[indeces[batch]], labels[indeces[batch]]);

				size_t curr = 0;
				for (; curr < batch_size && curr + batch < examples.size(); ++curr) {
					update_vector(
						gradient,
						example_gradient(examples[indeces[batch + curr]], labels[indeces[batch + curr]]),
						[](matrix& x, const matrix& y) { x += y; }
					);
				}
				
				update_vector(
					previous_update,
					std::move(gradient),
					[curr, learning_rate, influence](matrix& upd, const matrix& g) 
						{ upd = (learning_rate / curr) * g + upd * influence; }
				);

				update_vector(
					_weights, 
					previous_update, 
					[weight_decay](matrix& w, matrix& upd) { (w *= (1 - weight_decay)) += upd; }
				);
			}
		}
	}
};
