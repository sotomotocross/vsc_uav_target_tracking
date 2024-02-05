# visual_servoing_utils.py

import numpy as np

class VisualServoingUtils:
    def __init__(self, dt, window_size):
        """
        Initializes the Visual Servoing Utilities.

        Parameters:
        - dt (float): Sampling time.
        - window_size (int): Size of the moving average filter window.
        """
        self.dt = dt
        self.window_size = window_size
        self.values = []
        self.sum = np.zeros((8, 1))

    def deriv_error_estimation(self, er_pix, er_pix_prev):
        """
        Estimates the derivative of error.

        Parameters:
        - er_pix (numpy.ndarray): Current error.
        - er_pix_prev (numpy.ndarray): Previous error.

        Returns:
        - numpy.ndarray: Estimated derivative of error.
        """
        if er_pix_prev is not None:
            return (er_pix - er_pix_prev) / self.dt
        else:
            return np.zeros_like(er_pix)

    def moving_average_filter(self, data):
        """
        Applies a moving average filter to the input data.

        Parameters:
        - data (numpy.ndarray): Input data.

        Returns:
        - numpy.ndarray: Filtered result.
        """
        if self.window_size <= 1:
            return data

        self.values.append(data.tolist())  # Convert NumPy array to list and append

        self.sum = np.add(self.sum, data)

        if len(self.values) > self.window_size:
            removed_value = self.values.pop(0)
            self.sum = np.subtract(self.sum, removed_value)

        if len(self.values) == 0:
            return data

        ew_filtered = np.divide(self.sum, len(self.values)).reshape(8, 1)

        return ew_filtered
