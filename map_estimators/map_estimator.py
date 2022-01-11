from utilities import np, empty_array
import abc
from scipy.stats import norm
from IPython.core.debugger import set_trace
from utilities import watt_to_dbm, dbm_to_watt


class MapEstimator():
    """
    `metric`: indicates the estimated metric:
      - "power": metric = received power
      - "service": metric = posterior probability of received power >= `self.min_service_power`.
      -`f_shadowing_covariance` is a function of a distance argument that returns
            the covariance between the map values at two points separated by
            that distance.
    """
    name_on_figs = "Unnamed"

    freq_basis = None
    estimation_fun_type = ''  # 'g2g', 's2s'

    def __init__(
            self,
            grid=None,
            f_shadowing_covariance=None,
            min_service_power=None, # Minimum rx power to consider that there service at a point
    ):

        self.min_service_power = min_service_power
        self.grid = grid
        self._m_all_measurements = None
        self._m_all_measurement_loc = None
        self.f_shadowing_covariance = f_shadowing_covariance

    # def estimate(self, m_measurement_loc, m_measurement, m_test_loc=None, m_meta_data=None):
    # # def estimate(self, measurement_locs=None, measurements=None, test_loc=None, building_meta_data=None):
    #     """Args:
    #
    #         if `measurement_locs` is None: # input in grid form
    #             - `measurements` : num_sources x self.grid.num_points_y x self.grid.num_points_x.
    #             - `building_meta_data`: self.grid.num_points_y x self.grid.num_points_x
    #         else:                         # input in standard form
    #             - `measurement_locs` : num_measurements x 3 matrix with the
    #                3D locations of the measurements.
    #             - `measurements` : num_measurements x num_sources matrix
    #                with the measurements at each channel.
    #             - `building_meta_data`: num_points x 3 matrix with the 3D locations of the buildings
    #
    #         `m_test_loc`: can be None or a num_test_loc x 3 matrix with the
    #         locations where the map estimate will be evaluated.
    #
    #        Returns:
    #
    #        `d_map_estimate`: dictionary whose fields are:
    #
    #        - "t_power_map_estimate" :  tensor whose (i,j,k)-th entry is the
    #        estimated power of the i-th channel at grid point (j,k).
    #        - "t_power_map_norm_variance" : Contains the variance a
    #        posteriori normalized to the interval (0,1).
    #        - "t_service_map_estimate": [only if self.min_service_power
    #        is not None]. Each entry contains the posterior probability
    #        that the power is above `self.min_service_power`.
    #        - "t_service_map_entropy": [only if self.min_service_power
    #        is not None] Entropy of the service indicator.
    #
    #        If `test_loc` is None, the shape of these items is num_sources
    #        x self.grid.num_points_y x self.grid.num_points_x.
    #
    #        If `test_loc` is not None, the shape of these items is
    #        num_test_loc x num_sources.
    #
    #     """
    #     # Ny x Nx buffer matrix for a building meta data
    #     if self.m_meta_data is None:
    #         self.m_meta_data = m_meta_data
    #
    #     num_sources = m_measurement.shape[1]
    #     # get the measurements
    #     for loc, meas in zip(m_measurement_loc, m_measurement):
    #         self.store_measurement_old(loc, meas)
    #
    #     # Now estimate metric per channel
    #     t_power_map_est = empty_array(
    #         (num_sources, self.grid.num_points_y, self.grid.num_points_x))
    #     t_power_map_var = empty_array(
    #         (num_sources, self.grid.num_points_y, self.grid.num_points_x))
    #     if self.min_service_power is not None:
    #         t_service_map_est = empty_array(
    #             (num_sources, self.grid.num_points_y, self.grid.num_points_x))
    #         t_service_map_ent = empty_array(
    #             (num_sources, self.grid.num_points_y, self.grid.num_points_x))
    #     for ind_src in range(num_sources):
    #         t_power_map_est[ind_src, :, :], t_power_map_var[ind_src, :, :] = \
    #             self.estimate_power_one_channel(ind_src)
    #         if self.min_service_power is not None:
    #             t_service_map_est[ind_src, :, :], t_service_map_ent[ind_src, :, :] = \
    #                 self.estimate_service_one_gaussian_channel(t_power_map_est[ind_src, :, :], t_power_map_var[ind_src, :, :])
    #             # self.estimate_service_one_gaussian_channel(self._m_all_measurements[ind_src, :], ind_src)
    #
    #     # m_entropy = np.mean(t_power_map_var, axis=0)
    #
    #     d_map_estimate = {
    #         "t_power_map_estimate":
    #         t_power_map_est,
    #         "t_power_map_norm_variance":
    #         t_power_map_var / self.f_shadowing_covariance(0)  # ensure in (0,1)
    #     }
    #     if self.min_service_power:
    #         d_map_estimate["t_service_map_estimate"] = t_service_map_est
    #         d_map_estimate["t_service_map_entropy"] = t_service_map_ent
    #
    #     return d_map_estimate

    def estimate(self, measurement_locs=None, measurements=None, building_meta_data=None, test_loc=None):
        """
        This method first stores `measurement_locs` and `measurements` in a buffer together with
        previous measurement_locs and measurements provided through this method or through method
        store_measurement since the instantiation of the object or the last call to the reset method.

        Then the estimation method is implemented by the subclass is invoked over the stored
        measurements and measurement locations.

        Args:

            if `measurements` is in the form of
                num_sources x self.grid.num_points_y x self.grid.num_points_x: # input in grid form
                - `building_meta_data`: self.grid.num_points_y x self.grid.num_points_x
                - `measurement_locs`: self.grid.num_points_y x self.grid.num_points_x is a sampling mask
            else:                         # input in standard form
                - `measurements` : num_measurements x num_sources matrix
                   with the measurements at each channel.
                - `measurement_locs` : num_measurements x 3 matrix with the
                   3D locations of the measurements.
                - `building_meta_data`: num_points x 3 matrix with the 3D locations of the buildings

            `m_test_loc`: can be None or a num_test_loc x 3 matrix with the
            locations where the map estimate will be evaluated.

        Returns:
           `d_map_estimate`: dictionary whose fields are:

           - "t_power_map_estimate" :  tensor whose (i,j,k)-th entry is the
           estimated power of the i-th channel at grid point (j,k).
           - "t_power_map_norm_variance" : Contains the variance a
           posteriori normalized to the interval (0,1).
           - "t_service_map_estimate": [only if self.min_service_power
           is not None]. Each entry contains the posterior probability
           that the power is above `self.min_service_power`.
           - "t_service_map_entropy": [only if self.min_service_power
           is not None] Entropy of the service indicator.

           If `test_loc` is None, the shape of these items is num_sources
           x self.grid.num_points_y x self.grid.num_points_x.

           If `test_loc` is not None, the shape of these items is
           num_test_loc x num_sources.

            """

        # 1. store the data
        measurement_locs, measurements = self.store_measurement(measurement_locs,
                                                                measurements)

        # 2. invoke _estimate_no_storage on the stored data
        d_map_estimate = self._estimate_no_storage(measurement_locs, measurements,
                                                   building_meta_data, test_loc)
        return d_map_estimate

    def store_measurement(self, measurement_loc, measurements):
        """
            These method stores the provided measurement and measurement locations
            together with previous measurements and measurement locations.
        Args:
            if measurements.ndim is 3 then
                -`measurement`: is in the form of
                    num_sources x self.grid.num_points_y x self.grid.num_points_x
                -`measurement_loc`: is a mask in the form of
                    self.grid.num_points_y x self.grid.num_points_x

            else:
                -`measurement`: is in the form of num_measurements x num_sources
                -`measurement_loc`: is a mask in the form of num_measurements x 3
        Returns:
            if measurements.ndim is 3 then
                -`measurement`: is in the form of
                    num_sources x self.grid.num_points_y x self.grid.num_points_x
                -`measurement_loc`: is a mask in the form of
                    self.grid.num_points_y x self.grid.num_points_x

            else:
                -`measurement`: is in the form of num_measurements x num_sources
                -`measurement_loc`: is a mask in the form of num_measurements x 3

        """
        b_input_gf = (measurements.ndim == 3)
        if b_input_gf:
            # input data are in grid format
            if self._m_all_measurements is None:
                self._m_all_measurements = measurements
                self._m_all_measurement_loc = measurement_loc
            else:
                self._m_all_measurement_loc += measurement_loc
                self._m_all_measurements += measurements

            measurement_loc = np.minimum(self._m_all_measurement_loc, 1)
            # average the measurements at each grid point
            measurements = self._m_all_measurements / np.maximum(self._m_all_measurement_loc, 1)

        else:
            # input data are in standard format
            if self._m_all_measurements is None:
                self._m_all_measurements = measurements
                self._m_all_measurement_loc = measurement_loc
            else:
                self._m_all_measurements = np.vstack((self._m_all_measurements,
                                                      measurements))
                self._m_all_measurement_loc = np.vstack((self._m_all_measurement_loc,
                                                         measurement_loc))
            measurement_loc = self._m_all_measurement_loc
            measurements = self._m_all_measurements

        return measurement_loc, measurements

    def reset(self):
        """ Clear buffers to start estimating again."""
        self._m_all_measurements = None
        # self.m_all_mask = None
        self._m_all_measurement_loc = None
        # self.m_building_meta_data = None
        # self._m_building_meta_data_grid = None

    ########################### Utilities for the subclasses ###############################

    def estimate_metric_per_channel(self, measurement_loc,
                                    measurements, building_meta_data, test_loc,
                                    f_power_est=None, f_service_est=None):
        """
        Concatenates the estimates returned by `f_power_est` and `f_service_est`
        by invoking these functions per channel.
        Args:
             if `measurements` is in the form of
                num_sources x self.grid.num_points_y x self.grid.num_points_x: # input in grid form
                - `building_meta_data`: self.grid.num_points_y x self.grid.num_points_x
                - `measurement_locs`: self.grid.num_points_y x self.grid.num_points_x is a sampling mask
            else:                         # input in standard form
                - `measurements` : num_sources x num_measurements matrix
                   with the measurements at each channel.
                - `measurement_locs` : 3 x num_measurements matrix with the
                   3D locations of the measurements.
                - `building_meta_data`: 3 x num_points matrix with the 3D locations of the buildings

            -`f_power_est`: is a method provided by the subclass to estimate the power map per channel.
                    The input parameters of this method are:
                    Args:
                    if `estimation_func_type` is 's2s':
                        -`ind_channel`: an integer value corresponding to the channel(or source)
                        - `measurements` : num_sources x num_measurements matrix
                        with the measurements at each channel.
                        - `m_measurement_loc`: 3 x num_measurements matrix with the
                        3D locations of the measurements.
                        - `m_test_loc`: num_test_loc x 3 matrix with the
                        locations where the map estimate will be evaluated.
                    elif `estimation_func_type` is 'g2g':
                        - `measurements`: num_sources x self.grid.num_points_y x self.grid.num_points_x
                        - `building_meta_data`: self.grid.num_points_y x self.grid.num_points_x whose
                        value at each grid point is 1 if the grid point is inside a building
                        - `measurement_locs`: self.grid.num_points_y x self.grid.num_points_x, is
                            a sampling mask whose value is 1 at the grid point where
                            the samples were taken.
                    Returns:
                        if `estimation_func_type` is 's2s':
                        - `v_estimated_pow_of_source`: num_test_loc length vector that represents
                            the estimated power at each test location for ind_channel
                        - `v_variance`: a num_test_loc length vector that provides the posterior
                            variance at each test location for ind_channel

                        elif `estimation_func_type` is 'g2g':
                            Two self.grid.num_points_y x self.grid.num_points_x matrices of
                            the estimated posterior mean and the posterior variance of ind_channel.

            -`f_service_est`: is a method provided by the subclass to estimate the service map per channel.
                    This method should take the input parameters:
                    Args:
                        if`estimation_func_type` is 's2s':
                            `mean_power`, `variance` of shape num_test_loc length vector

                        elif `estimation_func_type` is 'g2g':
                            self.grid.num_points_y x self.grid.num_points_x
                    Returns:
                        `service` and `entropy` of corresponding input shape.

        Returns:

            -`d_map_estimate`: dictionary whose fields are of the shape
                    num_sources x self.grid.num_points_y x self.grid.num_points_x
                    if estimation_fun_type is g2g, else the shape is num_test_loc x num_sources
        """

        # Now estimate metric per channel
        if self.estimation_fun_type == 's2s':
            num_sources = measurements.shape[0]
            num_test_loc = test_loc.shape[0]
            t_power_map_est = empty_array((num_test_loc, num_sources))
            t_power_map_var = empty_array((num_test_loc, num_sources))
            if self.min_service_power is not None:
                t_service_map_est = empty_array((num_test_loc, num_sources))
                t_service_map_ent = empty_array((num_test_loc, num_sources))
            for ind_src in range(num_sources):
                # t_power_map_est[:, ind_src], t_power_map_var[:, ind_src] = \
                #     f_power_est(ind_src, measurement_loc, measurements, test_loc)
                t_power_map_est[:, ind_src], t_power_map_var_buffer = \
                    f_power_est(ind_src, measurement_loc, measurements, test_loc)
                if t_power_map_var_buffer is None:
                    t_power_map_var = None
                else:
                    t_power_map_var[:, ind_src] = t_power_map_var_buffer

                if self.min_service_power is not None:
                    t_service_map_est[:, ind_src], t_service_map_ent[:, ind_src] = \
                        f_service_est(t_power_map_est[:, ind_src], t_power_map_var[:, ind_src])

                    # self.estimate_service_one_gaussian_channel(self._m_all_measurements[ind_src, :], ind_src)
            # t_power_map_est = t_power_map_est.T
            # t_power_map_var = t_power_map_var.T
            # if self.min_service_power is not None:
            #     t_service_map_est = t_service_map_est.T
            #     t_service_map_ent = t_service_map_ent.T

        elif self.estimation_fun_type == 'g2g':
            num_sources = measurements.shape[0]
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
                # t_power_map_est[ind_src, :, :], t_power_map_var[ind_src, :, :] = \
                #     f_power_est(ind_src, measurement_loc, measurements, building_meta_data)
                t_power_map_est[ind_src, :, :], t_power_map_var_buffer = \
                    f_power_est(ind_src, measurement_loc, measurements, building_meta_data)
                if t_power_map_var_buffer is None:
                    t_power_map_var = None
                else:
                    t_power_map_var[ind_src, :, :] = t_power_map_var_buffer

                if self.min_service_power is not None:
                    t_service_map_est[ind_src, :, :], t_service_map_ent[ind_src, :, :] = \
                        f_service_est(t_power_map_est[ind_src, :, :],
                                                          t_power_map_var[ind_src, :, :])

        else:
            raise NotImplementedError

        d_map_estimate = {
            "t_power_map_estimate":
                t_power_map_est
        }
        if self.f_shadowing_covariance is not None:
            d_map_estimate["t_power_map_norm_variance"] = \
                t_power_map_var #/ self.f_shadowing_covariance(0)  # ensure in (0,1)
        else:
            d_map_estimate["t_power_map_norm_variance"] = t_power_map_var

        if self.min_service_power:
            d_map_estimate["t_service_map_estimate"] = t_service_map_est
            d_map_estimate["t_service_map_entropy"] = t_service_map_ent

        return d_map_estimate

    def estimate_service_one_gaussian_channel(self, mean_power, variance):
        """
        Returns:

            if inputs are in the form num_point_y x num_points_x matrix
            -`service`: num_point_y x num_points_x matrix where the
                (i,j)-th entry is the probability that the power at grid point
                (i,j) is greater than `self.min_service_power`.
            -`entropy`: bernoulli entropy associated with (i, j)-th grid point.

            else:
                The shape of the `service` and `entropy` are of are
                a vector of length num_test_loc
        """
        def entropy_bernoulli(service):
            # Avoid log2(0):
            service_copy = np.copy(service)
            b_zero_entropy = np.logical_or((service_copy == 0),
                                            (service_copy == 1))
            service_copy[b_zero_entropy] = .5  # dummy value

            entropy = -(1 - service_copy) * np.log2(1 - service_copy) - (
                service_copy) * np.log2(service_copy)
            entropy[b_zero_entropy] = 0

            return entropy

        service = 1 - norm.cdf(self.min_service_power, mean_power,
                                 np.sqrt(variance))

        entropy = entropy_bernoulli(service)
        return service, entropy

    ########################### private methods of this class ###############################

    def _estimate_no_storage(self, measurement_loc=None, measurements=None, building_meta_data=None, test_loc=None):
        """Args:

            if `measurements` is in the form of
                num_sources x self.grid.num_points_y x self.grid.num_points_x: # input in grid form
                - `building_meta_data`: self.grid.num_points_y x self.grid.num_points_x
                - `measurement_locs`: self.grid.num_points_y x self.grid.num_points_x is a sampling mask
            else:                         # input in standard form
                - `measurements` : num_measurements x num_sources matrix
                   with the measurements at each channel.
                - `measurement_locs` : num_measurements x 3 matrix with the
                   3D locations of the measurements.
                - `building_meta_data`: num_points x 3 matrix with the 3D locations of the buildings

            `m_test_loc`: can be None or a num_test_loc x 3 matrix with the
            locations where the map estimate will be evaluated.

           Returns:

           `d_map_estimate`: dictionary whose fields are:

           - "t_power_map_estimate" :  tensor whose (i,j,k)-th entry is the
           estimated power of the i-th channel at grid point (j,k).
           - "t_power_map_norm_variance" : Contains the variance a
           posteriori normalized to the interval (0,1).
           - "t_service_map_estimate": [only if self.min_service_power
           is not None]. Each entry contains the posterior probability
           that the power is above `self.min_service_power`.
           - "t_service_map_entropy": [only if self.min_service_power
           is not None] Entropy of the service indicator.

           If `test_loc` is None, the shape of these items is num_sources
           x self.grid.num_points_y x self.grid.num_points_x.

           If `test_loc` is not None, the shape of these items is
           num_test_loc x num_sources.

        """

        # flag to check if measurements is in a grid form
        b_input_gf= (len(measurements.shape) == 3)
        b_output_gf = (test_loc is None)

        # check input
        if b_input_gf:
            assert measurement_loc.shape[0] == measurements.shape[1]
            assert measurement_loc.shape[1] == measurements.shape[2]
        else:
            assert measurement_loc.shape[1] == 3

        # input adaptation
        if b_input_gf and self.estimation_fun_type == 's2s':
            measurement_loc, measurements, building_meta_data = self._input_g2s(
                measurement_loc, measurements, building_meta_data)

        elif not b_input_gf and self.estimation_fun_type == 'g2g':
            measurement_loc, measurements, building_meta_data = self._input_s2g(
                measurement_loc, measurements, building_meta_data)

        # if b_output_gf then set all the grid points as test locations for s2s
        if b_output_gf:
            test_loc = self.grid.all_grid_points_in_matrix_form()

        # invoke the method from the subclass
        if self.estimation_fun_type == 's2s':
            d_map_estimate = self.estimate_s2s(measurement_loc, measurements,
                                               building_meta_data, test_loc)
        elif self.estimation_fun_type == 'g2g':
            d_map_estimate = self.estimate_g2g(measurement_loc, measurements, building_meta_data)
            # d_map_estimate_old = self.estimate_g2g_old(measurement_loc_old, measurements_old, building_meta_data_old)
            # print("Norm equals", np.linalg.norm(d_map_estimate["t_power_map_norm_variance"] -
            #                                     d_map_estimate_old["t_power_map_norm_variance"]))
        else:
            raise NotImplementedError

        # output adaptation
        if b_output_gf:
            if self.estimation_fun_type == 's2s':
                d_map_estimate = self._output_s2g(d_map_estimate)
                # d_map_estimate_old = self.estimate_s2s_old(measurement_locs, measurements,
                #                                            building_meta_data, test_loc)
                # print("Norm equals", np.linalg.norm(d_map_estimate["t_power_map_estimate"] -
                #                                     d_map_estimate_old["t_power_map_estimate"]))

                return d_map_estimate
            elif self.estimation_fun_type == 'g2g':
                return d_map_estimate
            else:
                raise NotImplementedError
        else:
            if self.estimation_fun_type == 's2s':
                return d_map_estimate
            # elif self.estimation_fun_type == 'g2g':
            #     d_map_estimate = self._output_g2s(d_map_estimate)
            else:
                # TODO: implement this, e.g. using a sampler from sampler.py
                raise NotImplementedError

    def _input_g2s(self, measurement_loc=None, measurements=None, building_meta_data=None):
        """
        Args:
            - `measurements`: num_sources x self.grid.num_points_y x self.grid.num_points_x
            - `building_meta_data`: self.grid.num_points_y x self.grid.num_points_x
            - `measurement_locs`: self.grid.num_points_y x self.grid.num_points_x, a mask

        Returns:
            - `measurements` : num_measurements x num_sources matrix
               with the measurements at each channel.
            - `measurement_locs` : num_measurements x 3 matrix with the
               3D locations of the measurements.
            - `building_meta_data`: num_points x 3 matrix with the 3D locations of the buildings
        """

        m_measurements = measurements[:, measurement_loc == 1]
        measurements = m_measurements.T

        measurement_loc = self.grid.convert_grid_meta_data_to_standard_form(
            m_meta_data=measurement_loc)

        if building_meta_data is not None:
            building_meta_data = self.grid.convert_grid_meta_data_to_standard_form(
                m_meta_data=building_meta_data)

        return measurement_loc, measurements, building_meta_data

    def _input_s2g(self, measurement_loc=None, measurements=None, building_meta_data=None):
        """
        Args:
            - `measurements` : num_measurements x num_sources matrix
               with the measurements at each channel.
            - `measurement_locs` : num_measurements x 3 matrix with the
               3D locations of the measurements.
            - `building_meta_data`: num_points x 3 matrix with the 3D locations of the buildings

        Returns:
            _`t_all_measurements_grid`: a tensor of shape
                num_sources x num_gird_points_y x num_grid_points_x
            -`m_mask`: num_gird_points_y x num_grid_points_x binary mask whose
                entry is 1 at the grid point where measurement is taken,
            -`building_meta_data_grid`: num_gird_points_y x num_grid_points_x binary mask whose
                entry is 1 at the grid point that is inside a building,
                0 otherwise.
        """
        m_all_measurements = measurements.T
        m_all_measurements_loc = measurement_loc.T
        num_sources = m_all_measurements.shape[0]

        t_all_measurements_grid = np.zeros(
            (num_sources, self.grid.num_points_y, self.grid.num_points_x))
        m_mask = np.zeros(
            (self.grid.num_points_y, self.grid.num_points_x))

        m_all_measurements_loc_trans = m_all_measurements_loc.T

        m_all_measurements_col_index = 0  # to iterate through column of measurements

        # buffer counter to count repeated measurement in the grid point
        m_counter = np.zeros(np.shape(m_mask))

        for v_measurement_loc in m_all_measurements_loc_trans:
            # Find the nearest indices of the grid point closet to v_measurement_loc
            v_meas_loc_inds = self.grid.nearest_gridpoint_inds(v_measurement_loc)

            # replace the value of nearest grid point indices with measured value
            # for ind_sources in range(num_sources):
            #     t_all_measurements_grid[ind_sources, v_measurement_loc_inds[0],
            #                             v_measurement_loc_inds[1]] = v_measurement[ind_sources]

            # Add the previous measurements to the current measurement
            # at the (j,k)-th grid point
            t_all_measurements_grid[:,
            v_meas_loc_inds[0], v_meas_loc_inds[1]] += m_all_measurements[:,
                                                       m_all_measurements_col_index]

            # increment counters to store repeated measurements at the (j, k)-th grid point
            m_counter[v_meas_loc_inds[0], v_meas_loc_inds[1]] += 1

            # set the value of mask to 1 at the measured grid point indices
            m_mask[v_meas_loc_inds[0],
                   v_meas_loc_inds[1]] = 1

            m_all_measurements_col_index += 1

        # Average the measurements
        t_all_measurements_grid = np.divide(t_all_measurements_grid,
                                            m_counter,
                                            where=m_counter != 0,
                                            out=np.zeros(np.shape(
                                                t_all_measurements_grid)))

        building_meta_data_grid = np.zeros((self.grid.num_points_y,
                                      self.grid.num_points_x))
        if building_meta_data is not None:
            for v_building_meta_data in building_meta_data:
                v_meta_data_inds = self.grid.nearest_gridpoint_inds(v_building_meta_data)
                building_meta_data_grid[v_meta_data_inds[0],
                                                v_meta_data_inds[1]] = 1

        return m_mask, t_all_measurements_grid, building_meta_data_grid

    def _output_s2g(self, d_map_estimate=None):
        """
        Args:
            -`d_map_estimate`: whose fields are in the form
                num_test_loc x num_sources

        Returns:
            - `d_map_estimate`: whose fields are in the form
                    num_sources x self.grid.num_points_y x self.grid.num_points_x
            """
        num_sources = d_map_estimate["t_power_map_estimate"].shape[1]
        d_map_estimate["t_power_map_estimate"] = np.reshape(d_map_estimate["t_power_map_estimate"].T,
                                                            (num_sources, self.grid.num_points_y,
                                                             self.grid.num_points_x))
        if d_map_estimate["t_power_map_norm_variance"] is not None:
            d_map_estimate["t_power_map_norm_variance"] = np.reshape(d_map_estimate["t_power_map_norm_variance"].T,
                                                                (num_sources, self.grid.num_points_y,
                                                                 self.grid.num_points_x))
        if self.min_service_power:
            d_map_estimate["t_service_map_estimate"] = np.reshape(d_map_estimate["t_service_map_estimate"].T,
                                                                     (num_sources, self.grid.num_points_y,
                                                                      self.grid.num_points_x))
            d_map_estimate["t_service_map_entropy"] = np.reshape(d_map_estimate["t_service_map_entropy"].T,
                                                                     (num_sources, self.grid.num_points_y,
                                                                      self.grid.num_points_x))
        return d_map_estimate

    ########################### Old methods ################################################
    def estimate_at_loc(self,
                        m_measurement_loc,
                        m_measurements,
                        m_test_loc,
                        **kwargs):
        """Args:

            `m_measurement_loc`: num_measurements x num_dims_grid matrix

            `m_measurements`: num_measurements x num_channels matrix

            `m_test_loc`: num_test_loc x num_channels matrix

        Returns:

            dictionary with keys:

                `power_est`: num_test_loc x num_channels matrix with
                the power estimates at the test locations.

        """

        if self.freq_basis is None:
            power_est = np.array([
                self._estimate_power_map_one_channel_at_loc(
                    m_measurement_loc, m_measurements[:, ind_ch], m_test_loc,
                    **kwargs)[:, 0] for ind_ch in range(m_measurements.shape[1])
            ]).T
            d_est = {"power_est": power_est}
        else:
            raise NotImplementedError

        return d_est

    @abc.abstractmethod
    def _estimate_power_map_one_channel_at_loc(self, m_measurement_loc,
                                               v_measurements, m_test_loc,
                                               **kwargs):
        """Args:

            `m_measurement_loc`: num_meas x num_dims matrix

            `v_measurements`: length-num_meas vector

            `m_test_loc`: num_test_loc x num_dims matrix with the
            locations where the map estimate will be evaluated.

          Returns:

            length num_meas vector with the power estimates at
            locations `m_test_loc`.

        """
        pass




