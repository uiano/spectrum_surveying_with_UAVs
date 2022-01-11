import scipy
import numpy as np
import abc
from scipy.spatial.distance import euclidean
from utilities import empty_array
from scipy.stats import norm
import abc
from utilities import empty_array
from map_estimators.map_estimator import MapEstimator
from IPython.core.debugger import set_trace

from utilities import empty_array


class KrigingEstimator(MapEstimator):
    """
    `metric`: indicates the estimated metric:
      - "power": metric = received power
      - "service": metric = posterior probability of received power >= `self.min_service_power`.
    `f_shadowing_covariance` is a function of a distance argument that returns
    the covariance between the map values at two points separated by
    that distance.
    `f_mean`: if None, map assumed zero mean. Else, this is a function of the ind_channel and
        location.
    """
    estimation_fun_type = 's2s'
    def __init__(
            self,
            *args,
            # grid,
            # metric="power",  # values: "power", "service"
            # f_shadowing_covariance=None,
            f_mean=None,            
            dumping_factor=0.001,
            **kwargs
    ):

        # self.grid = grid
        # self.metric = metric
        # self.f_shadowing_covariance = f_shadowing_covariance
        self.f_mean = f_mean
        #self.min_service_power = min_service_power
        self.dumping_factor = dumping_factor

        # self.m_all_measurements = None  # num_sources x num_measurements_so_far
        # self.m_all_measurement_loc = None  # 3 x num_measurements_so_far

        super().__init__(*args,**kwargs)


class BatchKrigingEstimator(KrigingEstimator):
    name_on_figs = "KrigingEstimator"

    def __init__(
            self,

            *args,
            name_on_figs=None,
            **kwargs):
        super(BatchKrigingEstimator, self).__init__(*args,
                                            **kwargs)
        if name_on_figs is not None:
            self.name_on_figs = name_on_figs
        else:
            self.name_on_figs = "KrigingEstimator"
        self.reset()

    def estimate_s2s(self, measurement_loc=None,
                     measurements=None, building_meta_data=None,
                     test_loc=None):
        """
        Args:
            - `measurement_locs` : num_measurements x 3 matrix with the
                   3D locations of the measurements.
            - `measurements` : num_measurements x num_sources matrix
                   with the measurements at each channel.
            - `building_meta_data`: can be none or num_points x 3 matrix with the
                   3D locations of the buildings
            - `m_test_loc`: a num_test_loc x 3 matrix with the
                    locations where the map estimate will be evaluated.

        Returns:
            `d_map_estimate`: dictionary whose fields are:

           - "t_power_map_estimate" :  num_test_loc x num_sources matrix with
                estimated power of each channel at test_loc.
           - "t_power_map_norm_variance" : Contains the variance a posteriori
                normalized to the interval (0,1).
           - "t_service_map_estimate": [only if self.min_service_power
                is not None]. Each entry contains the posterior probability
                that the power is above `self.min_service_power`.
           - "t_service_map_entropy": [only if self.min_service_power
                is not None] Entropy of the service indicator.
        """


        ###############################################################################

        # Ny x Nx buffer matrix for a building meta data
        # if self.m_meta_data is None:
        #     self.m_meta_data = m_meta_data

        # num_sources = measurements.shape[1]
        measurements = measurements.T
        measurement_loc = measurement_loc.T
        # num_test_loc = test_loc.shape[0]
        # get the measurements
        # for loc, meas in zip(measurement_locs, measurements):
        #     self.store_measurement_old(loc, meas)

        # Now estimate metric per channel
        # t_power_map_est = empty_array((num_sources, num_test_loc))
        # t_power_map_var = empty_array((num_sources, num_test_loc))
        # if self.min_service_power is not None:
        #     t_service_map_est = empty_array((num_sources, num_test_loc))
        #     t_service_map_ent = empty_array((num_sources, num_test_loc))
        # for ind_src in range(num_sources):
        #     t_power_map_est[ind_src, :], t_power_map_var[ind_src, :] = \
        #         self.estimate_power_one_channel(ind_src, measurement_locs,
        #                                         measurements, test_loc)
        #     if self.min_service_power is not None:
        #         t_service_map_est[ind_src, :], t_service_map_ent[ind_src, :] = \
        #             self.estimate_service_one_gaussian_channel(t_power_map_est[ind_src, :], t_power_map_var[ind_src, :])
        #         # self.estimate_service_one_gaussian_channel(self.m_all_measurements[ind_src, :], ind_src)
        #
        # # m_entropy = np.mean(t_power_map_var, axis=0)
        #
        # d_map_estimate = {
        #     "t_power_map_estimate":
        #     t_power_map_est.T,
        #     "t_power_map_norm_variance":
        #     t_power_map_var.T / self.f_shadowing_covariance(0)  # ensure in (0,1)
        # }
        # if self.min_service_power:
        #     d_map_estimate["t_service_map_estimate"] = t_service_map_est.T
        #     d_map_estimate["t_service_map_entropy"] = t_service_map_ent.T
        d_map_estimate = self.estimate_metric_per_channel(measurement_loc,
                                                          measurements, building_meta_data,
                                                          test_loc,
                                                          f_power_est=self.estimate_power_one_channel,
                                                          f_service_est=self.estimate_service_one_gaussian_channel)

        return d_map_estimate

    def estimate_power_one_channel(self, ind_channel, m_measurement_loc, m_measurements, test_loc):
        """
        Args:
            `m_measurement_loc`: 3 x num_measurements matrix
            `m_measurements`: num_sources x num_measurements
            `test_loc`: num_test_loc x 3 matrix

        Returns:
            `v_estimated_pow_of_source`: num_test_loc length vector that represents
                the estimated power at each test location for ind_channel
            `v_variance`: a num_test_loc length vector that provides the posterior
                variance at each test location for ind_channel
        """

        num_test_loc = test_loc.shape[0]
        v_measurements = m_measurements[ind_channel, :]
        num_measurements = v_measurements.shape[0]

        if not self.f_mean:
            # average mean = 0
            mean_pobs = np.zeros(shape=(
                num_measurements,
                1))  # mean of power at the locations of the observations

            # mean of power at the locations without observations
            mean_pnobs = np.zeros(shape=(num_test_loc, 1))
        else:
            # mean of power at the locations with observations

            mean_pobs = np.reshape(
                self.f_mean(ind_channel, m_measurement_loc),
                (num_measurements, 1))

            # mean of power at the locations without observations
            # m_all_grid_points = np.reshape(self.grid.t_coordinates, (1, 3, -1))
            mean_pnobs = np.zeros(shape=(num_test_loc, 1))

            for ind_num_grid_points in range(num_test_loc):
                mean_pnobs[ind_num_grid_points] = \
                    self.f_mean(ind_channel, np.reshape(
                        (test_loc[ind_num_grid_points]), (3, 1)))

        m_location = m_measurement_loc

        # First obtain the partitioned covariance matrix (theorem 10.2 of Kay, page 324)
        cov_obs_obs = np.zeros(shape=(num_measurements, num_measurements))
        for ind_measurement in range(num_measurements):
            v_distance = np.linalg.norm(
                m_location - np.expand_dims(m_location[:, ind_measurement], 1),
                ord=2,
                axis=0)
            cov_obs_obs[ind_measurement, :] = self.f_shadowing_covariance(
                v_distance)

        cov_obs_nobs = np.zeros(shape=(num_measurements, num_test_loc))
        m_location_transposed = m_location.T
        for ind_measurement in range(0, num_measurements):
            cov_obs_nobs[ind_measurement, :] = self.f_shadowing_covariance(
                np.linalg.norm(m_location_transposed[ind_measurement] - test_loc,
                               ord=2, axis=1))

        cov_nobs_obs = cov_obs_nobs.T

        # cov_nobs_nobs = self.f_shadowing_covariance(
        #     self.grid.get_distance_matrix())
        cov_nobs_nobs = self.f_shadowing_covariance(
            self.grid.get_distance_matrix_test_loc(test_loc))

        # Dumping factor
        cov_obs_obs += self.dumping_factor * np.eye(num_measurements)

        v_coef = (np.linalg.inv(cov_obs_obs) @ (v_measurements.reshape(
            (num_measurements, 1)) - mean_pobs))

        # MMSE estimator for power at each spatial test_loc P(y/x)
        m_estimated_pow_of_source = mean_pnobs + cov_nobs_obs @ v_coef
        v_estimated_pow_of_source = m_estimated_pow_of_source[:,0]

        m_estimated_cov = cov_nobs_nobs - np.matmul(
            cov_nobs_obs,
            (np.matmul(np.linalg.inv(cov_obs_obs), cov_obs_nobs)))

        # (i,j)-th entry contains the posterior variance of the power at the (i,j)-th grid point
        v_variance = m_estimated_cov.diagonal()

        if (v_variance < 0).any():
            set_trace()
        # m_uncertainty = np.zeros((self.grid.num_points_x, self.grid.num_points_y))

        return v_estimated_pow_of_source, v_variance

    #########################################################
    def estimate_s2s_old(self, measurement_loc=None,
                     measurements=None, building_meta_data=None,
                     test_loc=None):
        """
        Args:
            - `measurement_locs` : num_measurements x 3 matrix with the
                   3D locations of the measurements.
            - `measurements` : num_measurements x num_sources matrix
                   with the measurements at each channel.
            - `building_meta_data`: num_points x 3 matrix with the
                   3D locations of the buildings
            - `m_test_loc`: can be None or a num_test_loc x 3 matrix with the
                    locations where the map estimate will be evaluated.

        Returns:
            `d_map_estimate`: dictionary whose fields are:

           - "power_map_estimate" :  num_test_loc x num_sources matrix with
                estimated power of each channel at test_loc.
           - "power_map_norm_variance" : Contains the variance a posteriori
                normalized to the interval (0,1).
           - "service_map_estimate": [only if self.min_service_power
                is not None]. Each entry contains the posterior probability
                that the power is above `self.min_service_power`.
           - "service_map_entropy": [only if self.min_service_power
                is not None] Entropy of the service indicator.
        """


        ###############################################################################

        # Ny x Nx buffer matrix for a building meta data
        # if self.m_meta_data is None:
        #     self.m_meta_data = m_meta_data

        num_sources = measurements.shape[1]
        self.m_all_measurements = measurements.T
        self.m_all_measurement_loc = measurement_loc.T
        # get the measurements
        # for loc, meas in zip(measurement_locs, measurements):
        #     self.store_measurement_old(loc, meas)

        # Now estimate metric per channel
        t_power_map_est = empty_array(
            (num_sources, self.grid.num_points_y, self.grid.num_points_x))
        t_power_map_var = empty_array(
            (num_sources, self.grid.num_points_y, self.grid.num_points_x))
        if self.min_service_power is not None:
            t_service_map_est = empty_array(
                (num_sources, self.grid.num_points_y, self.grid.num_points_x))
            t_service_map_ent = empty_array(
                (num_sources, self.grid.num_points_y, self.grid.num_points_x))
        for ind_src in range(num_sources):
            t_power_map_est[ind_src, :, :], t_power_map_var[ind_src, :, :] = \
                self.estimate_power_one_channel_old(ind_src)
            if self.min_service_power is not None:
                t_service_map_est[ind_src, :, :], t_service_map_ent[ind_src, :, :] = \
                    self.estimate_service_one_gaussian_channel(t_power_map_est[ind_src, :, :],
                                                               t_power_map_var[ind_src, :, :])
                # self.estimate_service_one_gaussian_channel(self.m_all_measurements[ind_src, :], ind_src)

        # m_entropy = np.mean(t_power_map_var, axis=0)

        d_map_estimate = {
            "t_power_map_estimate":
            t_power_map_est,
            "t_power_map_norm_variance":
            t_power_map_var / self.f_shadowing_covariance(0)  # ensure in (0,1)
        }
        if self.min_service_power:
            d_map_estimate["t_service_map_estimate"] = t_service_map_est
            d_map_estimate["t_service_map_entropy"] = t_service_map_ent

        return d_map_estimate

    def estimate_power_one_channel_old(self, ind_channel):

        num_grid_points = self.grid.t_coordinates.shape[
            1] * self.grid.t_coordinates.shape[2]

        v_measurements = self.m_all_measurements[ind_channel, :]
        num_measurements = v_measurements.shape[0]

        if not self.f_mean:
            # average mean = 0
            mean_pobs = np.zeros(shape=(
                num_measurements,
                1))  # mean of power at the locations of the observations

            # mean of power at the locations without observations
            mean_pnobs = np.zeros(shape=(num_grid_points, 1))
        else:
            # mean of power at the locations with observations

            mean_pobs = np.reshape(
                self.f_mean(ind_channel, self.m_all_measurement_loc),
                (num_measurements, 1))

            # mean of power at the locations with observations
            m_all_grid_points = np.reshape(self.grid.t_coordinates, (1, 3, -1))
            mean_pnobs = np.zeros(shape=(num_grid_points, 1))

            for ind_num_grid_points in range(num_grid_points):
                mean_pnobs[ind_num_grid_points] = \
                    self.f_mean(ind_channel, np.reshape(
                        (m_all_grid_points[0, :, ind_num_grid_points]), (3, 1)))

        m_location = self.m_all_measurement_loc

        # First obtain the partitioned covariance matrix (theorem 10.2 of Kay, page 324)
        cov_obs_obs = np.zeros(shape=(num_measurements, num_measurements))
        for ind_measurement in range(num_measurements):
            v_distance = np.linalg.norm(
                m_location - np.expand_dims(m_location[:, ind_measurement], 1),
                ord=2,
                axis=0)
            cov_obs_obs[ind_measurement, :] = self.f_shadowing_covariance(
                v_distance)

        cov_obs_nobs = np.zeros(shape=(num_measurements, num_grid_points))
        m_location_transposed = m_location.T
        for ind_measurement in range(0, num_measurements):
            cov_obs_nobs[ind_measurement, :] = self.f_shadowing_covariance(
                self.grid.get_distance_to_grid_points(
                    m_location_transposed[ind_measurement, :])).reshape(
                        1, num_grid_points)

        cov_nobs_obs = cov_obs_nobs.T

        cov_nobs_nobs = self.f_shadowing_covariance(
            self.grid.get_distance_matrix())

        # Dumping factor
        cov_obs_obs += self.dumping_factor * np.eye(num_measurements)

        v_coef = (np.linalg.inv(cov_obs_obs) @ (v_measurements.reshape(
            (num_measurements, 1)) - mean_pobs))

        # MMSE estimator for power at each spatial (for here at every grid point) P(y/x)
        m_estimated_pow_of_source = np.reshape(
            (mean_pnobs + cov_nobs_obs @ v_coef),
            (self.grid.num_points_y, self.grid.num_points_x))

        m_estimated_cov = cov_nobs_nobs - np.matmul(
            cov_nobs_obs,
            (np.matmul(np.linalg.inv(cov_obs_obs), cov_obs_nobs)))

        # (i,j)-th entry contains the posterior variance of the power at the (i,j)-th grid point
        m_variance = np.reshape(
            (m_estimated_cov.diagonal()),
            (self.grid.num_points_y, self.grid.num_points_x))

        if (m_variance < 0).any():
            set_trace()
        # m_uncertainty = np.zeros((self.grid.num_points_x, self.grid.num_points_y))

        return m_estimated_pow_of_source, m_variance


class OnlineKrigingEstimator(KrigingEstimator):
    name_on_figs = "OnlineKrigingEstimator"
    def __init__(self, *args, **kwargs):
        super(OnlineKrigingEstimator, self).__init__(*args, **kwargs)
        self.reset()

    def reset(self):

        super().reset()

        self.prev_mean = []
        self.prev_cov = []
        self.count_ind_channel = 0

    def estimate_s2s(self, measurement_loc=None,
                     measurements=None, building_meta_data=None,
                     test_loc=None):
        """
        Args:
            - `measurement_locs` : num_measurements x 3 matrix with the
                   3D locations of the measurements.
            - `measurements` : num_measurements x num_sources matrix
                   with the measurements at each channel.
            - `building_meta_data`: can be none or num_points x 3 matrix with the
                   3D locations of the buildings
            - `m_test_loc`: a num_test_loc x 3 matrix with the
                    locations where the map estimate will be evaluated.

        Returns:
            `d_map_estimate`: dictionary whose fields are:

           - "t_power_map_estimate" :  num_test_loc x num_sources matrix with
                estimated power of each channel at test_loc.
           - "t_power_map_norm_variance" : Contains the variance a posteriori
                normalized to the interval (0,1).
           - "t_service_map_estimate": [only if self.min_service_power
                is not None]. Each entry contains the posterior probability
                that the power is above `self.min_service_power`.
           - "t_service_map_entropy": [only if self.min_service_power
                is not None] Entropy of the service indicator.
        """



        measurements = measurements.T
        measurement_loc = measurement_loc.T

        d_map_estimate = self.estimate_metric_per_channel(measurement_loc,
                                                          measurements, building_meta_data,
                                                          test_loc,
                                                          f_power_est=self.estimate_power_one_channel,
                                                          f_service_est=self.estimate_service_one_gaussian_channel)

        return d_map_estimate

    def estimate_power_one_channel(self, ind_channel, m_measurement_loc, m_measurements, test_loc):
        """
                Args:
                    `m_measurement_loc`: 3 x num_measurements matrix
                    `m_measurements`: num_sources x num_measurements
                    `test_loc`: num_test_loc x 3 matrix

                Returns:
                    `v_estimated_mean_power`: num_test_loc length vector that represents
                        the estimated power at each test location for ind_channel
                    `v_variance`: a num_test_loc length vector that provides the posterior
                        variance at each test location for ind_channel
                """

        # num_grid_points = self.grid.t_coordinates.shape[
        #     1] * self.grid.t_coordinates.shape[2]
        num_grid_points = test_loc.shape[0]

        # v_measurements = self.m_all_measurements[ind_channel, :]
        v_measurements = [m_measurements[ind_channel,-1]]

        num_measurements = len(v_measurements)

        # num_sources = len(self.m_all_measurements)
        num_sources = m_measurements.shape[0]

        m_measurement_loc = m_measurement_loc[:,-1].reshape((3,1))

        if not self.f_mean:
            # average mean = 0
            mean_pobs = np.zeros(shape=(
                num_measurements,
                1))  # mean of power at the locations of the observations

            # mean of power at the locations without observations
            mean_pnobs = np.zeros(shape=(num_grid_points, 1))
        else:
            # mean of power at the locations with observations
            mean_pobs = self.f_mean(ind_channel, m_measurement_loc)

            # mean of power at the locations with observations
            m_all_grid_points = np.reshape(self.grid.t_coordinates, (1, 3, -1))
            mean_pnobs = np.zeros(shape=(num_grid_points, 1))

            for ind_num_grid_points in range(num_grid_points):
                mean_pnobs[ind_num_grid_points] = \
                    self.f_mean(ind_channel, np.reshape((m_all_grid_points[0, :, ind_num_grid_points]), (3, 1)))

        v_location = m_measurement_loc.T

        # First obtain the partitioned covariance matrix (theorem 10.2 of Kay, page 324)

        cov_nobs_nobs = self.f_shadowing_covariance(
            self.grid.get_distance_matrix())

        cov_obs_nobs = self.f_shadowing_covariance(
            self.grid.get_distance_to_grid_points(v_location)).reshape(
                1, num_grid_points)

        # Computing vector a_t and scalars b_t and lambda_t in the draft
        v_at = np.linalg.inv(cov_nobs_nobs) @ cov_obs_nobs.reshape(-1, 1)
        b_t = mean_pobs - (
            cov_obs_nobs @ np.linalg.inv(cov_nobs_nobs)) @ mean_pnobs
        lambda_t = self.f_shadowing_covariance(0) - (cov_obs_nobs @ (
            np.linalg.inv(cov_nobs_nobs) @ cov_obs_nobs.reshape(-1, 1)))

        if self.count_ind_channel < num_sources:
            # for initial at time t = 0
            self.prev_cov.append(cov_nobs_nobs)
            self.prev_mean.append(mean_pnobs)
            self.count_ind_channel += 1

        # MMSE estimator for power at each spatial (for here at every grid point)
        m_estimated_cov = np.linalg.inv(
            np.linalg.inv(self.prev_cov[ind_channel]) +
            (v_at * v_at.reshape(1, -1)) / lambda_t)

        v_estimated_mean_power = m_estimated_cov @ ((
            (v_measurements - b_t) / lambda_t) * v_at + np.linalg.inv(
                self.prev_cov[ind_channel]) @ self.prev_mean[ind_channel])

        # Store the current mean_power and covariance for next measurement
        self.prev_cov[ind_channel] = m_estimated_cov
        self.prev_mean[ind_channel] = v_estimated_mean_power

        # m_estimated_mean_power = np.reshape(
        #     v_estimated_mean_power,
        #     (self.grid.num_points_y, self.grid.num_points_x))
        v_estimated_pow_of_source = v_estimated_mean_power[:,0]
        # (i,j)-th entry contains the posterior variance of the power at the (i,j)-th grid point
        # m_variance = np.reshape(
        #     (m_estimated_cov.diagonal()),
        #     (self.grid.num_points_y, self.grid.num_points_x))

        v_variance = m_estimated_cov.diagonal()

        if (v_variance < 0).any():
            # set_trace()
            v_variance = np.copy(v_variance)
            v_variance[v_variance < 0] = 1e-6

        # m_uncertainty = np.zeros((self.grid.num_points_x, self.grid.num_points_y))

        return v_estimated_pow_of_source, v_variance


    #############################################################################
    def estimate_power_one_channel_old(self, ind_channel):
        num_grid_points = self.grid.t_coordinates.shape[
            1] * self.grid.t_coordinates.shape[2]

        v_measurements = self.m_all_measurements[ind_channel, :]
        num_measurements = v_measurements.shape[0]

        num_sources = len(self.m_all_measurements)

        if not self.f_mean:
            # average mean = 0
            mean_pobs = np.zeros(shape=(
                num_measurements,
                1))  # mean of power at the locations of the observations

            # mean of power at the locations without observations
            mean_pnobs = np.zeros(shape=(num_grid_points, 1))
        else:
            # mean of power at the locations with observations
            mean_pobs = self.f_mean(ind_channel, self.m_all_measurement_loc)

            # mean of power at the locations with observations
            m_all_grid_points = np.reshape(self.grid.t_coordinates, (1, 3, -1))
            mean_pnobs = np.zeros(shape=(num_grid_points, 1))

            for ind_num_grid_points in range(num_grid_points):
                mean_pnobs[ind_num_grid_points] = \
                    self.f_mean(ind_channel, np.reshape((m_all_grid_points[0, :, ind_num_grid_points]), (3, 1)))

        v_location = self.m_all_measurement_loc.T

        # First obtain the partitioned covariance matrix (theorem 10.2 of Kay, page 324)

        cov_nobs_nobs = self.f_shadowing_covariance(
            self.grid.get_distance_matrix())

        cov_obs_nobs = self.f_shadowing_covariance(
            self.grid.get_distance_to_grid_points(v_location)).reshape(
                1, num_grid_points)

        # Computing vector a_t and scalars b_t and lambda_t in the draft
        v_at = np.linalg.inv(cov_nobs_nobs) @ cov_obs_nobs.reshape(-1, 1)
        b_t = mean_pobs - (
            cov_obs_nobs @ np.linalg.inv(cov_nobs_nobs)) @ mean_pnobs
        lambda_t = self.f_shadowing_covariance(0) - (cov_obs_nobs @ (
            np.linalg.inv(cov_nobs_nobs) @ cov_obs_nobs.reshape(-1, 1)))

        if self.count_ind_channel < num_sources:
            # for initial at time t = 0
            self.prev_cov.append(cov_nobs_nobs)
            self.prev_mean.append(mean_pnobs)
            self.count_ind_channel += 1

        # MMSE estimator for power at each spatial (for here at every grid point)
        m_estimated_cov = np.linalg.inv(
            np.linalg.inv(self.prev_cov[ind_channel]) +
            (v_at * v_at.reshape(1, -1)) / lambda_t)

        v_estimated_mean_power = m_estimated_cov @ ((
            (v_measurements - b_t) / lambda_t) * v_at + np.linalg.inv(
                self.prev_cov[ind_channel]) @ self.prev_mean[ind_channel])

        # Store the current mean_power and covariance for next measurement
        self.prev_cov[ind_channel] = m_estimated_cov
        self.prev_mean[ind_channel] = v_estimated_mean_power

        m_estimated_mean_power = np.reshape(
            v_estimated_mean_power,
            (self.grid.num_points_y, self.grid.num_points_x))

        # (i,j)-th entry contains the posterior variance of the power at the (i,j)-th grid point
        m_variance = np.reshape(
            (m_estimated_cov.diagonal()),
            (self.grid.num_points_y, self.grid.num_points_x))

        if (m_variance < 0).any():
            # set_trace()
            m_variance = np.copy(m_variance)
            m_variance[m_variance < 0] = 1e-6

        # m_uncertainty = np.zeros((self.grid.num_points_x, self.grid.num_points_y))

        return m_estimated_mean_power, m_variance

    def store_measurement_old(self, v_measurement_loc, v_measurement):
        """Args:
            `v_measurement` : number source- length vector whose i-th
            entry denotes the received power at the i-th
            v_measurement_loc location transmitted by i-th power
            source.
           Returns:
           -`m_all_measurement_loc`: measurement location whose
           j-th column represents (x, y, z) coordinate of measurement location
           "m_all_measurements" :  measurements whose j-th
           entry denotes the received power at j-th m_all_measurement_loc
           transmitted by i-th power source.
        """
        num_sources = len(v_measurement)

        self.m_all_measurements = np.reshape(v_measurement, (num_sources, 1))
        self.m_all_measurement_loc = np.reshape(v_measurement_loc, (3, 1))