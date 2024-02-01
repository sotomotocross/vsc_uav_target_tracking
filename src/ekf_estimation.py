# ekf_estimation.py

import numpy as np

class EKFEstimation:
    def __init__(self, dt):
        """
        Initializes the EKF Estimation.

        Parameters:
        - dt (float): Sampling time.
        """
        self.P = 100.0 * np.eye(4, dtype=float)
        self.x_est = np.array([0.4, 0.4, 0.4, 0.4], dtype=float).reshape(4, 1)
        self.dt = dt

    def estimate(self, e_m, e_m_dot, use_ekf_estimator=True):
        """
        Performs EKF estimation to estimate the velocity.

        Parameters:
        - e_m (float): Centroid error.
        - e_m_dot (float): Gradient (velocity) after removing the camera velocity.
        - use_ekf_estimator (bool): Flag to enable or disable EKF estimation.

        Returns:
        - numpy.ndarray: Estimated velocity.
        """
        if use_ekf_estimator:
            PHI_S1 = 0.001
            PHI_S2 = 0.001

            F_est = np.zeros((4, 4), dtype=float)
            F_est[0, 1] = 1.0
            F_est[1, 0] = -self.x_est[2] ** 2
            F_est[1, 2] = -2.0 * self.x_est[2] * self.x_est[0] + 2.0 * self.x_est[2] * self.x_est[3]
            F_est[1, 3] = self.x_est[2] ** 2

            PHI_k = np.eye(4, 4) + F_est * self.dt

            Q_k = np.array([[0.001, 0.0, 0.0, 0.0],
                            [0.0, 0.5 * PHI_S1 * (
                                    -2.0 * self.x_est[2] * self.x_est[0] + 2.0 * self.x_est[2] * self.x_est[
                                3]) ** 2 * (self.dt ** 2) + 0.333 * self.x_est[2] ** 4 * (self.dt ** 3) * PHI_S2, 0.0,
                             0.5 * self.x_est[2] ** 2 * (self.dt ** 2) * PHI_S2],
                            [0.0, 0.5 * PHI_S1 * (
                                    -2.0 * self.x_est[2] * self.x_est[0] + 2.0 * self.x_est[2] * self.x_est[
                                3]) ** 2 * (self.dt ** 2), PHI_S1 * self.dt, 0.0],
                            [0.0, 0.5 * PHI_S2 * (self.dt ** 2) * self.x_est[2] ** 2, 0.0, PHI_S2 * self.dt]],
                           dtype=float).reshape(4, 4)

            H_k = np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0]], dtype=float).reshape(2, 4)

            R_k = 0.1 * np.eye(2, dtype=float)

            M_k = np.dot(np.dot(PHI_k, self.P), PHI_k.transpose()) + Q_k
            K_k = np.dot(np.dot(M_k, H_k.transpose()), np.linalg.inv(
                np.dot(np.dot(H_k, M_k), H_k.transpose()) + R_k))
            self.P = np.dot((np.eye(4) - np.dot(K_k, H_k)), M_k)

            xdd = -((self.x_est[2] ** 2) * self.x_est[0]) + (self.x_est[2] ** 2) * self.x_est[3]
            xd = self.x_est[1] + self.dt * xdd
            x = self.x_est[0] + self.dt * xd

            x_dash_k = np.array([x, xd, self.x_est[2], self.x_est[3]]).reshape(4, 1)

            z = np.array([e_m, e_m_dot]).reshape(2, 1)

            self.x_est = np.dot(PHI_k, x_dash_k) + np.dot(K_k, z - np.dot(H_k, np.dot(PHI_k, x_dash_k)))

            return self.x_est[1]  # Return the estimated velocity
        else:
            # If not using EKF, return a placeholder or something meaningful
            return np.zeros((8, 1))
