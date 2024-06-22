/*
 * File:        kalman_filter.h
 * Author:      Yufeng Ma
 * Date:        2024-06-01
 * Email:       97357473@qq.com
 * Description: This header file defines the KalmanFilter class, which provides
 *              a basic implementation of the Kalman filter algorithm. The class
 *              includes methods for predicting the state and updating it with
 *              new measurements. The key components are:
 *              - State vector (x): Represents the estimated state of the system.
 *              - State transition matrix (F): Models the state evolution.
 *              - Measurement matrix (H): Maps the state to the measurement space.
 *              - Covariance matrix (P): Represents the uncertainty in the state estimate.
 *              - Measurement noise covariance matrix (R): Represents the uncertainty in the measurements.
 *              - Process noise covariance matrix (Q): Represents the uncertainty in the process model.
 *              The class is initialized with the dimensions of the state and
 *              measurement vectors, and provides methods for prediction and update.
 */

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
