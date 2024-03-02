#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include "Eigen/Dense"

///
///
/// This file defines the various methods used by 
/// The Local Fischer Discriminant Analysis
/// 
/// 


///
/// We support embeddings types
/// a> weighted
/// b> plain
/// c> orthonormalized
/// 
enum class EMBEDDING_TYPE {
	WEIGHTED = 0,
	PLAIN,
	ORTHONORMALIZED
};

/// Implementation of LFDA algorithm
/// The implementation here is a supervised implementation
/// of LDA
/// However, this implementation can be easily adapted to a unsupervised implementation
class LFDA {
public:
	LFDA(const int n_components,
		const int k,
		const EMBEDDING_TYPE embedding);

	LFDA() = delete;

	~LFDA();

	void Fit(const Eigen::MatrixXd& Xin,
		const Eigen::VectorXd& yin);

	Eigen::MatrixXd Transform(const Eigen::MatrixXd& X);

private:
	int n_components_;
	int k_;
	EMBEDDING_TYPE embedding_;

	// to hold the solution eigen vectors after fit
	Eigen::MatrixXd components_;

	// Some supporting math methods
	Eigen::MatrixXd OuterProduct(Eigen::MatrixXd A);

	static Eigen::MatrixXd AddSmallRegularization(const Eigen::MatrixXd& a,
		double epsilon);

	void EigenSolver(const Eigen::MatrixXd& a,
		const Eigen::MatrixXd & b, int dim,
		Eigen::MatrixXd& eigenVectors, Eigen::VectorXd& eigenValues);

	int CheckNComponents(int n_features, int n_components);
};
