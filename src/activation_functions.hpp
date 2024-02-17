#pragma once

#include "matrix.hpp"
#include <cmath>


class activation_function {

public:
	virtual double apply(double x) = 0;
	virtual double derivative(double x) = 0;
	virtual double weight_init_variance(size_t num_inputs, size_t num_neurons) = 0;
	virtual ~activation_function() = default;

private:
	matrix& apply_func_base(matrix& mx, double (activation_function::*func_ptr)(double)) {
		for (size_t idx = 0; idx < mx.size(); ++idx) {
			mx[idx] = (this->*func_ptr)(mx[idx]);
		}
		return mx;
	}

public:
	virtual matrix& apply(matrix& potentials) {
		return apply_func_base(potentials, &activation_function::apply);
	}

	virtual matrix apply(const matrix& potentials) {
		matrix copy = potentials;
		return apply_func_base(copy, &activation_function::apply);
	}

	virtual matrix& derivative(matrix& potentials) {
		return apply_func_base(potentials, &activation_function::derivative);
	}

	virtual matrix derivative(const matrix& potentials) {
		matrix copy = potentials;
		return apply_func_base(copy, &activation_function::derivative);
	}
};

class selu: public activation_function {
	double _lambda;
	double _alpha;

public:
	selu(double lambda, double alpha):
		_lambda(lambda),
		_alpha(alpha)
	{}

public:
	double apply(double x) override {
		if (x < 0) {
			return _lambda * _alpha * (exp(x) - 1);
		}
		return _lambda * x;
	}

	double derivative(double x) override {
		if (x < 0) {
			return _lambda * _alpha * exp(x);
		}
		return _lambda;
	}

	double weight_init_variance(size_t num_inputs, size_t num_neurons) override {
		return 1.0 / num_inputs;
	}
};

class softmax: public activation_function {

private:
	double apply(double _) override {
		return 0;
	}

	double derivative(double _) override {
		return 0;
	}

public:
	double weight_init_variance(size_t num_inputs, size_t num_neurons) override {
		return 2.0 / (num_inputs + num_neurons);
	}

	matrix& apply(matrix& potentials) override {
		double max = potentials.max();

		double sum = 0;
		for (double& val: potentials) {
			val = exp(val - max);
			sum += val;
		}

		std::transform(
			potentials.begin(),
			potentials.end(),
			potentials.begin(),
			[sum](double x) {return x / sum;}
		);

		return potentials;
	}

	matrix apply(const matrix& potentials) override {
		matrix copy = potentials;
		return apply(copy);
	}
};
