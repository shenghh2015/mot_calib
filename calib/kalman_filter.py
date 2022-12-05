# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

# x, y, h, w, vx, vy, vh, w
class KalmanFilter(object):
    """
    A 3D cardboard Kalman filter for tracking bounding boxes in image space.

    The 6-dimensional state space

        dx, dy, da, db, ddx, ddy, dda, ddb,

    contains the cardboard center position on the ground floor (x, y), width w, height h,
    and the cardboard's velocity.

    """

    def __init__(self, dt=1.0, xy_std=0.1, h_std = 0.01, w_std=0.01, vel_scalar = 0.1):

        state_dim  = 8
        cboard_dim = 4
 
        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(state_dim, state_dim)
        for i in range(4):
            self._motion_mat[i, cboard_dim + i] = dt

        self._update_mat = np.eye(cboard_dim, state_dim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.

        # TODO: pick the right values through experiements 
        self._pos_std = xy_std
        self._h_std   = h_std
        self._w_std   = w_std

        self.vel_scalar  = vel_scalar

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos[:4])
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._pos_std,
            2 * self._pos_std,
            2 * self._h_std,
            2 * self._w_std,
            4 * self._pos_std,
            4 * self._pos_std,
            4 * self._h_std,
            4 * self._w_std]

        covariance = np.diag(np.square(std))

        # previous states
        self.prev_mean = np.zeros_like(mean_pos[:4])

        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._pos_std,
            self._pos_std,
            self._h_std,
            self._w_std]
        std_vel = [
            self.vel_scalar * self._pos_std,
            self.vel_scalar * self._pos_std,
            self.vel_scalar * self._h_std,
            self.vel_scalar * self._w_std]

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """

        #TODO: adjust the std based on visibility of measurement
        std = [
            self._pos_std,
            self._pos_std,
            self._h_std,
            self._w_std]

        innovation_cov = np.diag(np.square(std))
        
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))

        output_cov = covariance + innovation_cov

        return mean, output_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T

        innovation     = measurement - projected_mean

        new_mean       = mean + np.dot(innovation, kalman_gain.T)

        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance
