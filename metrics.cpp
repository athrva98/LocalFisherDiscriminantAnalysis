#include "metrics.h"

#include <Eigen/Core>
#include <iostream>
#include <map>
#include <functional>
#include <cmath>
#include <omp.h>

Metrics::Metrics() {

}

Metrics::~Metrics() {
}

Eigen::MatrixXd Metrics::manhattan_distances(const Eigen::MatrixXd& X) {
    int n = X.rows();
    Eigen::MatrixXd result(n, n);
#pragma omp parallel for default(none) shared(n, X, result) collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result(i, j) = (X.row(i) - X.row(j)).cwiseAbs().sum();
        }
    }

    return result;
}

Eigen::MatrixXd Metrics::cosine_distances(const Eigen::MatrixXd& x) {
    int n = x.rows();
    Eigen::MatrixXd result(n, n);
#pragma omp parallel for default(none) shared(x, n, result) collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double dot_product = x.row(i).dot(x.row(j));
            double norm_i = x.row(i).norm();
            double norm_j = x.row(j).norm();

            if (norm_i > 0 && norm_j > 0) {
                result(i, j) = 1.0 - (dot_product / (norm_i * norm_j));
            }
            else {
                result(i, j) = 1.0; // Handle division by zero or zero vectors
            }
        }
    }

    return result;
}

Eigen::MatrixXd Metrics::euclidean_distances(const Eigen::MatrixXd& x) {
    int n = x.rows();
    Eigen::MatrixXd result(n, n);
#pragma omp parallel for default(none) shared(x, result, n) collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result(i, j) = (x.row(i) - x.row(j)).norm();
        }
    }
    return result;
}

Eigen::MatrixXd Metrics::haversine_distances(const Eigen::MatrixXd& x) {
    int n = x.rows();
    Eigen::MatrixXd result(n, n);

    // TODO: Throw NotImplementedError
    return result;
}

Eigen::MatrixXd Metrics::nan_euclidean_distances(const Eigen::MatrixXd& x) {
    int n = x.rows();
    Eigen::MatrixXd result(n, n);
#pragma omp parallel for default(none) shared(x, n, result) collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                result(i, j) = (x.row(i) - x.row(j)).norm();
            }
            else {
                result(i, j) = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }

    return result;
}


Eigen::MatrixXd Metrics::pairwise_distances(const Eigen::MatrixXd& X,
    const std::string& metric) {
    if (Metrics::PAIRWISE_DISTANCE_FUNCTIONS.find(metric)
        != Metrics::PAIRWISE_DISTANCE_FUNCTIONS.end()) {
        return Metrics::PAIRWISE_DISTANCE_FUNCTIONS[metric](X);
    }
    else {
        throw std::invalid_argument(
            "Unknown Metric : " + metric);
    }
}

std::map<std::string,
    std::function<Eigen::MatrixXd(
        const Eigen::MatrixXd&)>> Metrics::PAIRWISE_DISTANCE_FUNCTIONS = {
{"cityblock", Metrics::manhattan_distances},
{"cosine", Metrics::cosine_distances},
{"euclidean", Metrics::euclidean_distances},
{"haversine", Metrics::haversine_distances},
{"nan_euclidean", Metrics::nan_euclidean_distances}
};
