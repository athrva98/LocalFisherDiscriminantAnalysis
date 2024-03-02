#pragma once

#include "Eigen/Core"
#include <map>
#include <functional>
#include <stdexcept>


class Metrics {
public:
    Metrics();

    ~Metrics();

    static Eigen::MatrixXd manhattan_distances(
        const Eigen::MatrixXd& X);

    static Eigen::MatrixXd cosine_distances(
        const Eigen::MatrixXd& x);

    static Eigen::MatrixXd euclidean_distances(
        const Eigen::MatrixXd& x);

    static Eigen::MatrixXd haversine_distances(
        const Eigen::MatrixXd& x);

    static Eigen::MatrixXd nan_euclidean_distances(
        const Eigen::MatrixXd& x);

    static std::map<std::string,
        std::function<Eigen::MatrixXd(
            const Eigen::MatrixXd&)>>
        PAIRWISE_DISTANCE_FUNCTIONS;

    static Eigen::MatrixXd pairwise_distances(const Eigen::MatrixXd& X,
        const std::string& metric);
};
