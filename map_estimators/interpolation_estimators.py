import scipy
import cvxpy as cpy
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import norm
from map_estimators.map_estimator import MapEstimator
from IPython.core.debugger import set_trace
from utilities import empty_array


class MultikernelEstimator(MapEstimator):
    """
           Arguments:
               kernel_type : can be laplacian,  Gaussian, or other types  kernels ( Note: only laplacian and Gaussian
               kernels are implemented here, see kernel_function)
               max_kernel_width : maximum parameter of the chosen kernels
               num_kernels : number of  kernels
               reg_par: regularization parameter in kernel ridge regression
           """

    name_on_figs = "Multikernel"
    estimation_fun_type = 's2s'

    def __init__(self,
                 kernel_type="laplacian",
                 max_kernel_width=None,
                 num_kernels=15,
                 reg_par=1e-2,
                 **kwargs):
        super(MultikernelEstimator, self).__init__(**kwargs)
        self.kernel_type = kernel_type
        self.max_kernel_width = max_kernel_width
        self.num_kernels = num_kernels
        self.reg_par = reg_par

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
           - "t_power_map_norm_variance" : None
           - "t_service_map_estimate": None
           - "t_service_map_entropy": None
        """
        measurements = measurements.T
        measurement_loc = measurement_loc.T

        d_map_estimate = self.estimate_metric_per_channel(measurement_loc,
                                                          measurements, building_meta_data,
                                                          test_loc,
                                                          f_power_est=self.estimate_power_one_channel)
        return d_map_estimate

    def estimate_power_one_channel(self, ind_channel, measurement_loc=None,
                                   measurements=None, test_loc= None):
        """
        Args:
            - `measurements` : num_sources x num_measurements matrix
                   with the measurements at each channel.
            - `m_measurement_loc`: 3 x num_measurements matrix with the
                   3D locations of the measurements.
            - `m_test_loc`: num_test_loc x 3 matrix with the
                    locations where the map estimate will be evaluated.
        Returns:
            - length num_meas vector with the power estimates at
                locations `test_loc`.
        """
        measurements = measurements.T
        measurement_loc = measurement_loc.T

        # m_test_loc = self.grid.all_grid_points_in_matrix_form()
        v_power_map_est_one_ch = self._estimate_power_map_one_channel_at_loc(
            m_measurement_loc=measurement_loc,
            v_measurements=measurements[:, ind_channel],
            m_test_loc=test_loc)

        return v_power_map_est_one_ch[:, 0], None

    def _estimate_power_map_one_channel_at_loc(self, m_measurement_loc,
                                              v_measurements, m_test_loc, **kwargs):

        """Args:
            `m_measurement_loc`: num_meas x num_dims matrix

            `v_measurements`: length-num_meas vector
            `m_test_loc`: num_test_loc x num_dims matrix with the
            locations where the map estimate will be evaluated.
          Returns:
            length num_meas vector with the power estimates at
            locations `m_test_loc`.
        """

        if len(v_measurements) == 1:
            return v_measurements * np.ones((m_test_loc.shape[0], 1))

        num_mea_loc = m_measurement_loc.shape[0]
        # check if the kernel parameter max_kernel_width is None, if yes, use the this default value

        if self.max_kernel_width is None:
            m_distances = euclidean_distances(m_measurement_loc, m_measurement_loc)
            self.max_kernel_width = 10 * np.mean(m_distances)

        # v_kernel_widths = np.random.uniform(size=self.num_kernels) * self.max_kernel_width
        if self.kernel_type == "laplacian":
            v_kernel_widths = self.max_kernel_width * np.linspace(0.1, 1, self.num_kernels)
        else:
            v_kernel_widths = self.max_kernel_width * np.linspace(0.005, 1, self.num_kernels)

        # obtain the kernel coefficients
        t_kernel_matrices = np.zeros((num_mea_loc, num_mea_loc, self.num_kernels))
        for ind_loc in range(num_mea_loc):
            v_point_1 = m_measurement_loc[ind_loc, :]
            m_row_inputs_to_kernel_func = np.tile(v_point_1, (num_mea_loc, 1))
            for ind_kernel in range(self.num_kernels):
                t_kernel_matrices[ind_loc, :, ind_kernel] = kernel_function(m_row_inputs_to_kernel_func, m_measurement_loc,
                                                                      v_kernel_widths[ind_kernel],
                                                                      kernel_type=self.kernel_type)

        m_all_kernel_matrices = np.reshape(t_kernel_matrices, (num_mea_loc, num_mea_loc * self.num_kernels),
                                         order='F')
        t_chol_matrices = np.zeros(t_kernel_matrices.shape)
        for ind_kernel in range(self.num_kernels):
            t_chol_matrices[:, :, ind_kernel] = np.linalg.cholesky(
                t_kernel_matrices[:, :, ind_kernel] + 1e-7 * np.eye(
                    num_mea_loc))  # or 1e-7 worked better with laplacian

        v_kernel_coeffs = cpy.Variable(self.num_kernels * num_mea_loc)
        m_all_chol_matrices = np.reshape(t_chol_matrices, (num_mea_loc, num_mea_loc * self.num_kernels), order='F')
        gamma = cpy.Parameter(nonneg=True)

        objective = cpy.Minimize(0.5 * cpy.sum_squares((m_all_kernel_matrices @ v_kernel_coeffs) - v_measurements) +
                             gamma * (cpy.norm2(cpy.sum_squares(m_all_chol_matrices @ v_kernel_coeffs))))
        p = cpy.Problem(objective)
        gamma_value = self.reg_par
        gamma.value = gamma_value
        result = p.solve(solver='ECOS')
        v_kernel_coeffs = v_kernel_coeffs.value
        # print(v_kernel_coeffs)
        
        # Estimate using obtained kernel coefficients
        num_test_loc = m_test_loc.shape[0]
        v_power_map_est_one_ch = empty_array((num_test_loc, 1))
        for ind_loc in range(num_test_loc):
            v_query_point = m_test_loc[ind_loc, :]
            m_query_point_rep = np.tile(v_query_point, (num_mea_loc, 1))
            v_kernel_vecs = np.zeros((num_mea_loc, self.num_kernels))
            for ind_kernel in range(self.num_kernels):
                v_kernel_vecs[:, ind_kernel] = kernel_function(m_query_point_rep, m_measurement_loc,
                                                               v_kernel_widths[ind_kernel],
                                                               kernel_type=self.kernel_type)
                v_power_map_est_one_ch[ind_loc, 0] = (v_kernel_vecs.flatten('F').dot(v_kernel_coeffs))

        return v_power_map_est_one_ch


class KernelRidgeRegressionEstimator(MapEstimator):

    """
        Arguments:
            kernel_type : can be laplacian,  Gaussian, or other types  kernels ( Note: only laplacian and Gaussian
            kernels are implemented here, see kernel_function)
            kernel_width : parameter of the chosen kernel
            reg_par: regularization parameter in kernel ridge regression
        """
    name_on_figs = "KernelRidgeRegression"
    estimation_fun_type = 's2s'

    def __init__(self,
                 kernel_type="gaussian",
                 kernel_width=None,
                 reg_par=1e-7,
                 **kwargs):
        super(KernelRidgeRegressionEstimator, self).__init__(**kwargs)
        self.kernel_type = kernel_type
        self.kernel_width = kernel_width
        self.reg_par = reg_par
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
           - "t_power_map_norm_variance" : None
           - "t_service_map_estimate": None
           - "t_service_map_entropy": None
        """
        measurements = measurements.T
        measurement_loc = measurement_loc.T

        d_map_estimate = self.estimate_metric_per_channel(measurement_loc,
                                                          measurements, building_meta_data,
                                                          test_loc,
                                                          f_power_est=self.estimate_power_one_channel)
        return d_map_estimate

    def estimate_power_one_channel(self, ind_channel, measurement_loc=None,
                                   measurements=None, test_loc= None):
        """
        Args:
            - `measurements` : num_sources x num_measurements matrix
                   with the measurements at each channel.
            - `m_measurement_loc`: 3 x num_measurements matrix with the
                   3D locations of the measurements.
            - `m_test_loc`: num_test_loc x 3 matrix with the
                    locations where the map estimate will be evaluated.
        Returns:
            - length num_meas vector with the power estimates at
                locations `test_loc`.
        """
        measurements = measurements.T
        measurement_loc = measurement_loc.T
        # m_test_loc = self.grid.all_grid_points_in_matrix_form()
        v_power_map_est_one_ch = self._estimate_power_map_one_channel_at_loc(
            m_measurement_loc=measurement_loc,
            v_measurements=measurements[:, ind_channel],
            m_test_loc=test_loc)

        return v_power_map_est_one_ch[:, 0], None

    def _estimate_power_map_one_channel_at_loc(self, m_measurement_loc,
                                            v_measurements, m_test_loc, **kwargs):
        """
        Args:
            -`m_measurement_loc`: num_meas x num_dims matrix

            -`v_measurements`: length-num_meas vector
            -`m_test_loc`: num_test_loc x num_dims matrix with the
                locations where the map estimate will be evaluated.
        Returns:
            - length num_meas vector with the power estimates at
                locations `m_test_loc`.
        """

        if len(v_measurements)==1:
            return v_measurements * np.ones((m_test_loc.shape[0], 1))

        num_mea_loc = m_measurement_loc.shape[0]

        # check if the kernel parameter kernel_width is None, if yes, use the this default value
        if self.kernel_width is None:
            m_distances = euclidean_distances(m_measurement_loc, m_measurement_loc)
            self.kernel_width = 1 * np.mean(m_distances)
            print(self.kernel_width)
        # obtain the kernel coefficients
        m_kernel_matrix = np.zeros((num_mea_loc, num_mea_loc))
        for ind_loc in range(num_mea_loc):
            v_point_1 = m_measurement_loc[ind_loc, :]
            m_row_inputs_to_kernel_func = np.tile(v_point_1, (num_mea_loc, 1))
            m_kernel_matrix[ind_loc] = kernel_function(m_row_inputs_to_kernel_func, m_measurement_loc,
                                                   kernel_width=self.kernel_width, kernel_type=self.kernel_type)
        v_kernel_coeff = np.linalg.inv(m_kernel_matrix + num_mea_loc * self.reg_par * np.eye(num_mea_loc)) \
            .dot(v_measurements)

        # Estimate using obtained kernel coefficients
        num_test_loc = m_test_loc.shape[0]
        v_power_map_est_one_ch = empty_array((num_test_loc, 1))
        for ind_loc in range(num_test_loc):
            v_query_point = m_test_loc[ind_loc, :]
            m_query_point_rep = np.tile(v_query_point, (num_mea_loc, 1))
            v_kernel_vec = kernel_function(m_query_point_rep, m_measurement_loc,
                                         kernel_width=self.kernel_width, kernel_type=self.kernel_type)
            v_power_map_est_one_ch[ind_loc, 0] = v_kernel_vec.dot(v_kernel_coeff)
        return v_power_map_est_one_ch


class KNNEstimator(MapEstimator):
    """
           Arguments:
               num_neighbors : number of neighbors
           """

    name_on_figs = "KNN"
    estimation_fun_type = 's2s'

    def __init__(self,
                 # f_shadowing_covariance,
                 num_neighbors=5,
                 **kwargs):
        super(KNNEstimator, self).__init__(**kwargs)
        self.num_neighbors = num_neighbors
        # self.f_shadowing_covariance = f_shadowing_covariance
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
           - "t_power_map_norm_variance" : None
           - "t_service_map_estimate": None
           - "t_service_map_entropy": None
        """
        measurements = measurements.T
        measurement_loc = measurement_loc.T

        d_map_estimate = self.estimate_metric_per_channel(measurement_loc,
                                                          measurements, building_meta_data,
                                                          test_loc,
                                                          f_power_est=self.estimate_power_one_channel)
        return d_map_estimate

    def estimate_power_one_channel(self, ind_channel, measurement_loc=None,
                                   measurements=None, test_loc= None):
        """
        Args:
            - `measurements` : num_sources x num_measurements matrix
                   with the measurements at each channel.
            - `m_measurement_loc`: 3 x num_measurements matrix with the
                   3D locations of the measurements.
            - `m_test_loc`: num_test_loc x 3 matrix with the
                    locations where the map estimate will be evaluated.
        Returns:
            - length num_meas vector with the power estimates at
                locations `test_loc`.
        """
        measurements = measurements.T
        measurement_loc = measurement_loc.T
        # m_test_loc = self.grid.all_grid_points_in_matrix_form()
        v_power_map_est_one_ch = self._estimate_power_map_one_channel_at_loc(
            m_measurement_loc=measurement_loc,
            v_measurements=measurements[:, ind_channel],
            m_test_loc=test_loc)

        return v_power_map_est_one_ch[:, 0], None

    def _estimate_power_map_one_channel_at_loc(self, m_measurement_loc,
                                            v_measurements, m_test_loc, **kwargs):
        """Args:
            `m_measurement_loc`: num_meas x num_dims matrix

            `v_measurements`: length-num_meas vector
            `m_test_loc`: num_test_loc x num_dims matrix with the
            locations where the map estimate will be evaluated.
          Returns:
            length num_meas vector with the power estimates at
            locations `m_test_loc`.
        """
        num_mea_loc = m_measurement_loc.shape[0]
        num_test_loc = m_test_loc.shape[0]
        v_power_map_est_one_ch = empty_array((num_test_loc, 1))
        for ind_loc in range(num_test_loc):
            v_current_point = m_test_loc[ind_loc, :]
            m_current_point_rep = np.tile(v_current_point, (num_mea_loc, 1))

            v_dist_to_all_meas = np.linalg.norm(np.subtract(m_current_point_rep, m_measurement_loc), axis=1)
            v_neighbor_meas = v_measurements[np.argsort(v_dist_to_all_meas)[0:self.num_neighbors]]
            v_power_map_est_one_ch[ind_loc, 0] = np.mean(v_neighbor_meas)

        return v_power_map_est_one_ch



def kernel_function(x1, x2, kernel_width, kernel_type):
    if kernel_type == "laplacian":
        output = np.exp(- np.linalg.norm(np.subtract(x1, x2), axis=1) / kernel_width)
    elif kernel_type == "gaussian":
        output = np.exp(- np.linalg.norm(np.subtract(x1, x2), axis=1) ** 2 / (kernel_width ** 2))
    else:
        raise NotImplementedError
    return output