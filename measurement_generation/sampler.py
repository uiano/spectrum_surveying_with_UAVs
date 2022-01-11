
from utilities import np, empty_array
import scipy


class Sampler():

    def __init__(self, grid, noise_std=0):
        self.grid = grid
        self.noise_std = noise_std

    def sample_map(self, t_map, v_sample_location):
        return NotImplementedError
    

class InterpolationSampler(Sampler):
    def __init__(self, *args, interpolation_method="linear", **kwargs):

        self.interpolation_method = interpolation_method
        super().__init__(*args, **kwargs)

    def sample_map_multiple_loc(self, t_map, m_sample_locations):
        """m_sample_location: num_points x 3 matrix with the 3D locations of
        the sampling points.

        Returns:

        num_points x num_channels matrix with the measurements.

        """
        l_samples = []
        for v_location in m_sample_locations:
            l_samples.append(self.sample_map(t_map, v_location))

        return np.array(l_samples)

    def sampled_map_multiple_grid_loc(self, t_true_map, num_measurements=1, m_building_metadata=None):
        """
        Args:
            `m_sample_locations`: num_points x 3 matrix with 3D locations of
                the sampling points.
            `m_building_metadata`: Ny x Nx matrix whose (i, j)-th entry is 1
            if the grid point is inside buildings
        Returns:
            `t_sampled_map` - a sampled map of shape
            num_sources x grid.num_point_y x grid.num_point_x whose [i, j, k] entry is
            the received power (or measured power) from i-th source
            at the grid location indexed with [j,k], elsewhere 0.

            `t_sample_mask` - a binary mask of shape
            1 x grid.num_point_y x grid.num_point_x whose [0,j,k] entry is 1
            if the measurement is taken at the grid location indexed with [j,k],
            else 0.
        """
        t_sample_mask = np.zeros(
            (1, t_true_map.shape[1], t_true_map.shape[2]))

        v_indices_to_sampled_from = self.grid.random_grid_points_inds_outside_buildings(num_points=num_measurements,
                                                            m_building_metadata=m_building_metadata)
        v_mask = t_sample_mask.flatten()
        # set the mask value to 1 at the sampled grid point indices
        v_mask[v_indices_to_sampled_from] = 1
        t_sample_mask = v_mask.reshape(t_sample_mask.shape)

        # Sampled map
        t_sampled_map = t_true_map * t_sample_mask

        return t_sampled_map, t_sample_mask
        
    def sample_map(self, t_map, v_sample_location):
        """Args:
           `t_map` is num_sources x self.grid.num_points_y x
        self.grid.num_points_x tensor with the power at each channel
        and location.
        Returns:
           length- num_sources vector where the i-th entry is the
           power measured at point `v_sample_location`. It is computed
           by interpolating `t_map[i,:,:]` and adding a zero-mean
           Gaussian random variable with std self.noise_std
           FUT: improve noise model.
           """
        # find the nearest grid point to the sample point
        # then find the power from t_map at that grid
        num_sources = t_map.shape[0]

        # OLD
        # v_sample_power = np.full(shape=(num_sources, ),
        #                          fill_value=None,
        #                          dtype=float)
        # for ind_source in range(num_sources):
        #     v_sample_power[ind_source] = t_map[ind_source, min_dist_index[0], min_dist_index[1]] + \
        #         np.random.normal(loc=0, scale=self.noise_std)

        if self.interpolation_method == "avg_nearest":
            l_four = self.grid.four_nearest_gridpoint_indices(
                v_sample_location)
            m_four_values = empty_array((num_sources, 4))
            for ind_source in range(num_sources):
                for ind_point in range(4):
                    m_four_values[ind_source, ind_point] = \
                        t_map[ind_source, l_four[ind_point][0], l_four[ind_point][1]]

            v_sample_power_interpolated = np.mean(m_four_values, axis=1) + \
                                          np.random.normal(size=(num_sources,), loc=0, scale=self.noise_std)

        elif self.interpolation_method == "linear":
            l_three, l_coef = self.grid.point_as_convex_combination(
                v_sample_location)
            v_sample_power_interpolated = empty_array((num_sources,))
            for ind_source in range(num_sources):
                l_three_values = [
                    t_map[ind_source, point_inds[0], point_inds[1]]
                    for point_inds in l_three
                ]
                v_sample_power_interpolated[ind_source] = np.dot(
                    l_coef, l_three_values)

        elif self.interpolation_method == "splines":

            x = self.grid.t_coordinates[0, 0, :]
            y = self.grid.t_coordinates[1, ::-1, 0]
            v_sample_power_interpolated = empty_array((num_sources,))
            for ind_source in range(num_sources):
                z = t_map[ind_source, ::-1, :].T
                interpolator = scipy.interpolate.RectBivariateSpline(x, y, z)
                v_sample_power_interpolated[ind_source] = interpolator.ev(
                    v_sample_location[0], v_sample_location[1])

        elif self.interpolation_method == "in_grid":
            """v_sample_location is a grid point """
            v_sample_power_interpolated = empty_array((num_sources,))

            min_dist_index = self.grid.nearest_gridpoint_inds(v_sample_location)

            for ind_source in range(num_sources):
                v_sample_power_interpolated[ind_source] = t_map[ind_source, min_dist_index[0], min_dist_index[1]] + \
                    np.random.normal(loc=0, scale=self.noise_std)

            # print(v_sample_power)
            # print(v_sample_power_interpolated)
            # print(m_four_values)

        return v_sample_power_interpolated

################### old method ########################

    def sampled_map_multiple_grid_loc_old(self, t_true_map, m_sample_locations):
        """
        Args:
            `m_sample_locations`: num_points x 3 matrix with 3D locations of
                the sampling points.
        Returns:
            `t_all_measurements_grid` - a sampled map of shape
            num_sources x grid.num_point_y x grid.num_point_x whose [i, j, k] entry is
            the received power (or measured power) from i-th source
            at the grid location indexed with [j,k], elsewhere 0.

            `t_sample_mask` - a binary mask of shape
            1 x grid.num_point_y x grid.num_point_x whose [0,j,k] entry is 1
            if the measurement is taken at the grid location indexed with [j,k],
            else 0.
        """
        t_all_measurements_grid = np.zeros(np.shape(t_true_map))
        t_sample_mask = np.zeros(
            (1, t_true_map.shape[1], t_true_map.shape[2]))

        # num_samples = np.random.randint(0, 100)
        # num_samples = num_measurements

        " collect/sample measurements in grid"
        # m_sample_locations = grid.random_points_in_the_grid(num_points=num_samples)

        num_sources = t_true_map.shape[0]

        for v_sample_location in m_sample_locations:
            v_measurement = self.sample_map(t_map=t_true_map,
                                            v_sample_location=v_sample_location)
            'get the grid point indices'
            v_measurement_loc_inds = self.grid.nearest_gridpoint_inds(v_sample_location)

            for ind_sources in range(num_sources):
                t_all_measurements_grid[ind_sources, v_measurement_loc_inds[0],
                                        v_measurement_loc_inds[1]] = v_measurement[ind_sources]
            t_sample_mask[0,
                          v_measurement_loc_inds[0],
                          v_measurement_loc_inds[1]] = 1

        return t_all_measurements_grid, t_sample_mask