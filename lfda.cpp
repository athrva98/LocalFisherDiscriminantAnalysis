#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "lfda.h"
#include "metrics.h"

// File scope functions
std::unordered_set<double> get_unique_classes(Eigen::VectorXd y) {
    std::unordered_set<double> unique_classes;
    for (int i = 0; i < y.rows(); i++) {
        // if the element i.s not present in the set, insert it
        if (unique_classes.find(static_cast<int>(y(i))) == unique_classes.end()) {
            unique_classes.insert(static_cast<int>(y(i)));
        }
    }
    return unique_classes;
}

Eigen::MatrixXd filter_indices_by_class(
    Eigen::MatrixXd X, Eigen::VectorXd y, int label) {
    std::vector<Eigen::VectorXd> X_class;

    for (int i = 0; i < X.rows(); i++) {
        if (y(i) == label) {
            X_class.push_back(X.row(i));
        }
    }
    Eigen::MatrixXd X_class_eigen;
    X_class_eigen.resize(X_class.size(), X_class.at(0).size());

    for (int i = 0; i < (int)X_class.size(); i++) {
        X_class_eigen.row(i) = X_class.at(i);
    }
    return X_class_eigen;
}

Eigen::MatrixXd calculate_sigma(const Eigen::MatrixXd& dist, int k) {
    int n = dist.rows();
    Eigen::MatrixXd sigma(n, 1);
#pragma omp parallel for shared(dist) firstprivate(n, k)
    for (int col = 0; col < n; ++col) {
        Eigen::VectorXd column_values = dist.col(col);

        std::nth_element(column_values.data(), column_values.data() + k, column_values.data() + n);

        double sigma_val = std::sqrt(column_values(k));
        sigma(col, 0) = std::isnormal(sigma_val) ? sigma_val : 0.0; // Check for normal value
    }

    return sigma;
}

// Member functions of the LFDA class

LFDA::LFDA(const int n_components,
	const int k,
	const EMBEDDING_TYPE embedding) :
	n_components_(n_components),
	k_(k),
	embedding_(embedding)
{
    Eigen::initParallel();
	// implemented as initializer list
}

LFDA::~LFDA() {
    // empty
}

void LFDA::EigenSolver(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, int dim,
    Eigen::MatrixXd& eigenVectors, Eigen::VectorXd& eigenValues) {
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(a, b);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigenvalue decomposition failed.");
    }

    eigenValues = solver.eigenvalues().real().topRows(dim);
    eigenVectors = solver.eigenvectors().real().leftCols(dim);

    // Filter out negative or nan eigenvalues which indicate numerical instability
    for (int i = 0; i < eigenValues.size(); ++i) {
        if (eigenValues(i) < 0 || std::isnan(eigenValues(i))) {
            eigenValues(i) = 0;
        }
    }
}

Eigen::MatrixXd LFDA::AddSmallRegularization(const Eigen::MatrixXd& a, double epsilon) {
    int n = a.rows();
    double reg_strength = epsilon * a.norm(); // Adjust regularization strength based on matrix norm
    Eigen::MatrixXd regularization = reg_strength * Eigen::MatrixXd::Identity(n, n);
    return a + regularization;
}

Eigen::MatrixXd LFDA::OuterProduct(const Eigen::MatrixXd x) {
    Eigen::VectorXd sum = x.colwise().sum();
    return sum * sum.transpose();
}

int LFDA::CheckNComponents(int n_features, int n_components) {
    if (n_components == -1) {
        return n_features;
    }
    if (n_components > 0 && n_components <= n_features) {
        return n_components;
    }
    throw std::invalid_argument(
        "Invalid n_components, must be in [1, "
        + std::to_string(n_features) + "]");
}

void LFDA::Fit(const Eigen::MatrixXd& Xin,
    const Eigen::VectorXd& yin) {

    std::shared_ptr<Eigen::MatrixXd> X = std::make_shared<Eigen::MatrixXd>(Xin);
    std::shared_ptr<Eigen::VectorXd> y = std::make_shared<Eigen::VectorXd>(yin);


    // Check to see if X has sufficient number of rows
    if (!X || !y || X->rows() <= 2 || X->rows() != y->rows()) {
        return;
    }

    std::unordered_set<double> unique_classes = get_unique_classes(*y);

    // Check to see if X has sufficient number of features
    if (X->cols() < this->n_components_) {
        return;
    }

    if (this->k_ <= 0) {
        this->k_ = std::min<int>(7, X->cols() - 1);
    }
    else if (this->k_ >= X->cols()) {
        // k must be <= d - 1
        return;
    }

    Eigen::MatrixXd tSb;
    Eigen::MatrixXd tSw;

    tSb.resize(X->cols(), X->cols());
    tSw.resize(X->cols(), X->cols());

    tSb.setZero(X->cols(), X->cols());
    tSw.setZero(X->cols(), X->cols());

    for (const int& element : unique_classes) {
        Eigen::MatrixXd X_class = filter_indices_by_class(*X, *y, element);

        int nc = X_class.rows();

        Eigen::MatrixXd dist = Metrics::pairwise_distances(X_class, "euclidean");

        int local_k = std::min<int>(this->k_, nc - 1);
        Eigen::MatrixXd sigma = calculate_sigma(dist, local_k);

        Eigen::MatrixXd local_scale = LFDA::OuterProduct(sigma);

        Eigen::MatrixXd A = (-dist.array() / (1e-7 + local_scale.array())).exp();

#pragma omp parallel for default(none) shared(A, local_scale) collapse(2) num_threads(2)
        for (int i = 0; i < A.rows(); ++i) {
            for (int j = 0; j < A.cols(); ++j) {
                if (local_scale(i, j) == 0) {
                    A(i, j) = 0.0;
                }
            }
        }

        Eigen::MatrixXd G = X_class.transpose() * (A.rowwise().sum().array().matrix() * X_class);

        tSb += G / X->rows() +
            (1.0 - static_cast<double>(X_class.rows()) / X->rows())
            * (X_class.transpose() * X_class) + LFDA::OuterProduct(X_class);

        tSw += G / X_class.rows();
    }
    tSb -= (LFDA::OuterProduct(*X) / X->rows()) - tSw;

    tSb = (tSb + tSb.transpose()) / 2;
    tSw = (tSw + tSw.transpose()) / 2;

    Eigen::VectorXd eigen_values;
    Eigen::MatrixXd eigen_vectors;
    LFDA::EigenSolver(tSb, tSw, this->n_components_, eigen_vectors, eigen_values);

    std::vector<int> idx(eigen_values.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(), [&eigen_values](int i1, int i2) {return eigen_values(i1) > eigen_values(i2); });

    Eigen::VectorXd sortedVals(this->n_components_);
    Eigen::MatrixXd sortedVecs(eigen_vectors.rows(), this->n_components_);

    for (int i = 0; i < this->n_components_; ++i) {
        sortedVals(i) = eigen_values(idx[i]);
        sortedVecs.col(i) = eigen_vectors.col(idx[i]);
    }

    eigen_values = sortedVals;
    eigen_vectors = sortedVecs;

    if (this->embedding_ == EMBEDDING_TYPE::WEIGHTED) {
        for (int i = 0; i < this->n_components_; ++i) {
            eigen_vectors.col(i) *= std::sqrt(eigen_values(i));
        }
    }
    else if (this->embedding_ == EMBEDDING_TYPE::ORTHONORMALIZED) {
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(eigen_vectors);
        eigen_vectors = qr.householderQ();
    }

    this->components_ = eigen_vectors.transpose();

    return;
}

Eigen::MatrixXd LFDA::Transform(const Eigen::MatrixXd& X) {
    std::cout << X.rows() << " " << X.cols() << "\n";
    std::cout << this->components_.cols() << " " << this->components_.rows() << "\n";
    Eigen::MatrixXd X_res = X * this->components_.transpose();
    return X_res;
}
