#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "matrix.hpp"
#include <fstream>
#include <string>
#include <sstream>


std::vector<double>& standardize(std::vector<double>& values) {
	double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();

	double accum = std::accumulate(
		values.begin(),
		values.end(),
		0.0,
		[mean](double l, double r) {return l + std::pow(r - mean, 2);}
	);

	double stdev = std::sqrt(accum / values.size());

	std::transform(
		values.begin(),
		values.end(),
		values.begin(),
		[mean, stdev](double x) {return (x - mean) / stdev;}
	);

	return values;
}


std::vector<matrix> load_examples(const std::string& filename) {
	std::ifstream file(filename);
	std::string line;

	std::vector<matrix> examples;

	while (std::getline(file, line)) {
		std::vector<double> example;
		std::istringstream line_stream(line);

		for (int val; line_stream >> val;) {
			example.push_back(val);

			if (line_stream.peek() == ',') {
				line_stream.ignore();
			}
		}

		examples.emplace_back(std::move(standardize(example)));
	}

	return examples;
}


std::vector<matrix> load_labels(const std::string& filename) {
	std::ifstream file(filename);

	std::vector<matrix> labels;

	for (int label; file >> label;) {
		std::vector<double> row(10, 0);
		row[label] = 1;

		labels.emplace_back(std::move(row));
	}

	return labels;
}




