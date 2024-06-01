#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <Eigen/Dense>

class KalmanFilter {
public:
    KalmanFilter(int state_dim, int measure_dim);

    void predict();
    void update(const Eigen::VectorXd& z);

    Eigen::VectorXd x; // state vector
    Eigen::MatrixXd F; // state transition matrix
    Eigen::MatrixXd H; // measurement matrix
    Eigen::MatrixXd P; // covariance matrix
    Eigen::MatrixXd R; // measurement noise covariance matrix
    Eigen::MatrixXd Q; // process noise covariance matrix

private:
    int state_dim;
    int measure_dim;
};

#endif // KALMAN_FILTER_H
