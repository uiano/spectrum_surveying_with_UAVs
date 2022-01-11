
from IPython.core.debugger import set_trace
import pickle
from utilities import project_to_interval, mat_argmin, mat_argmax, \
    natural_to_dB, dB_to_natural
import numpy as np


class Grid():
    pass


class RectangularGrid(Grid):
    """
        The origin is on the bottom-left entry of the grid.
    """
    def __init__(
            self,
            # area_side_length=None,
            gridpoint_spacing=None,  # Distance between adjacent grid points
            num_points_x=None,
            num_points_y=None,
            height=None):
        # Input check: check that mandatory arguments were provided
        assert gridpoint_spacing
        assert num_points_x
        assert num_points_y
        assert height, "Argument `height` must be provided"

        self.num_points_x = num_points_x
        self.num_points_y = num_points_y
        self.gridpoint_spacing = gridpoint_spacing

        # create a grid
        v_x_coords = np.arange(0, self.num_points_x) * self.gridpoint_spacing
        v_y_coords = np.arange(self.num_points_y - 1, -1,
                               step=-1) * self.gridpoint_spacing

        x_coords, y_coords = np.meshgrid(v_x_coords, v_y_coords, indexing='xy')
        z_coords = height * np.ones((self.num_points_y, self.num_points_x))

        self.t_coordinates = np.array([x_coords, y_coords, z_coords])

        self._m_all_distances = None

        # self.m_all_distances = self.get_distance_matrix()

    #  def all_coordinates(self):
    #     # returns a  two-row array of grid i.e rearrange corresponding x an y coordinates
    #     return m_grid_cordinates = np.array([self.x_coords.ravel(), self.y_coords.ravel()])

    # m_all_distances is computed only if the object access this property.
    @property
    def m_all_distances(self):
        if self._m_all_distances is None:
            self._m_all_distances = self.get_distance_matrix()

        return self._m_all_distances

    def four_nearest_gridpoint_indices(self, v_point):
        """returns a list with the indices of the 4 nearest grid points to
            `v_point`. Two or four of these points may coincide.
        """
        row_before, row_after, col_before, col_after = self.nearest_rows_and_cols(
            v_point)

        l_points = [(row_before, col_before), (row_before, col_after),
                    (row_after, col_after), (row_after, col_before)]

        return l_points

    def nearest_rows_and_cols(self, v_point):
        col_before = np.floor(v_point[0] / self.gridpoint_spacing)
        col_after = np.ceil(v_point[0] / self.gridpoint_spacing)

        row_before = self.num_points_y - 1 - np.ceil(
            v_point[1] / self.gridpoint_spacing)
        row_after = self.num_points_y - 1 - np.floor(
            v_point[1] / self.gridpoint_spacing)

        # Ensure within limits
        col_before = int(project_to_interval(col_before, 0, self.num_points_x))
        col_after = int(project_to_interval(col_after, 0, self.num_points_x))
        row_before = int(project_to_interval(row_before, 0, self.num_points_y))
        row_after = int(project_to_interval(row_after, 0, self.num_points_y))

        return row_before, row_after, col_before, col_after

    def point_as_convex_combination(self, v_point):
        """This function may be used for interpolation.
        Returns:
        `l_point_inds`: list with at most 3 tuples of two indices,
        each corresponding to one of the grid points that are
        nearest to v_point.
        'l_coef': list of coefficients adding up to 1 such that the
        x and y coordinates (1st two entries) of \sum_ind l_coef[ind]
        l_point_inds[ind] equal the x and y coordinates of v_point.
        """
        def point_inds_to_coefficients(l_point_inds, v_point):
            # Find convex combination coefficients -> system of equations
            m_points = np.array(
                [self.indices_to_point(tp_inds) for tp_inds in l_point_inds])
            m_A = np.vstack((m_points[:, 0:2].T, np.ones(
                (1, len(l_point_inds)))))
            v_b = np.array([v_point[0], v_point[1], 1])

            if np.linalg.matrix_rank(m_A) < len(l_point_inds):
                set_trace()

            # v_coefficients = np.linalg.solve(m_A, v_b)
            v_coefficients = np.linalg.pinv(
                m_A) @ v_b  # Use pinv since there may be repeated rows

            # avoid numerical issues
            eps = 1e-6
            v_coefficients[np.logical_and(v_coefficients < 0,
                                          v_coefficients > -eps)] = 0

            if np.logical_or(
                    v_coefficients > 1, v_coefficients <
                    0).any() or np.abs(np.sum(v_coefficients) - 1) > 1e-3:
                print(v_coefficients)
                set_trace()
            return v_coefficients

        # debug_code
        # v_point = [51, 137,0]

        row_before, row_after, col_before, col_after = self.nearest_rows_and_cols(
            v_point)

        # Boundary cases
        if row_before == row_after:
            if col_before == col_after:
                return [(row_before, col_before)], [1]
            else:
                # The point lies in a segment between two points
                l_point_inds = [(row_before, col_before),
                                (row_before, col_after)]
        else:
            if col_before == col_after:
                l_point_inds = [(row_before, col_before),
                                (row_after, col_before)]
            else:
                # General case
                l_point_inds = [(row_after, col_before),
                                (row_before, col_after)]

                # Out of the other two points, find the one that is the closest
                ind_pt_1 = (row_before, col_before)
                ind_pt_2 = (row_after, col_after)
                d1 = np.linalg.norm(self.indices_to_point(ind_pt_1) - v_point)
                d2 = np.linalg.norm(self.indices_to_point(ind_pt_2) - v_point)
                if d1 < d2:
                    l_point_inds.append(ind_pt_1)
                else:
                    l_point_inds.append(ind_pt_2)

        v_coefficients = point_inds_to_coefficients(l_point_inds, v_point)

        return l_point_inds, list(v_coefficients)

    def get_distance_to_grid_points(self, v_point):
        """ returns an self.num_points_x x self.num_points_y whose
        (i,j)-th entry is the distance from the (i,j)-th grid point to
        v_point.
           """

        #
        # to get the distance between grid point and transmitter
        t_repeated_point = np.repeat(np.repeat(np.reshape(v_point, (3, 1, 1)),
                                               self.num_points_y,
                                               axis=1),
                                     self.num_points_x,
                                     axis=2)
        return np.linalg.norm(self.t_coordinates - t_repeated_point,
                              ord=2,
                              axis=0)

    def get_distance_matrix(self):
        """Returns an (self.num_points_x * self.num_points_y) x
        (self.num_points_x * self.num_points_y) matrix with the
        distance between each pair of grid points.
        """
        m_all_points = np.reshape(self.t_coordinates, (1, 3, -1))
        tcol_all_points = np.transpose(m_all_points, axes=(1, 2, 0))

        tcol_all_points_repeated = np.repeat(tcol_all_points,
                                             self.num_points_x *
                                             self.num_points_y,
                                             axis=2)

        m_all_distances = np.linalg.norm(
            tcol_all_points_repeated -
            np.transpose(tcol_all_points_repeated, axes=(0, 2, 1)),
            ord=2,
            axis=0)

        return m_all_distances

    def get_distance_matrix_test_loc(self, test_loc):
        """
        Args:
            - `test_loc` : a num_test_loc x 3 matrix matrix of test locations

        Returns:
            - `m_all_distances` :a num_test_loc x num_test_loc matrix with the
            distance between each pair of test_loc.
        """
        m_all_points = test_loc.T
        m_all_points = m_all_points[None, :, :]
        tcol_all_points = np.transpose(m_all_points, axes=(1, 2, 0))

        tcol_all_points_repeated = np.repeat(tcol_all_points,
                                             test_loc.shape[0],
                                             axis=2)

        m_all_distances = np.linalg.norm(
            tcol_all_points_repeated -
            np.transpose(tcol_all_points_repeated, axes=(0, 2, 1)),
            ord=2,
            axis=0)

        return m_all_distances

    def nearest_gridpoint_inds(self, point):
        """Returns a tuple with the indices of the grid point closest to `point`"""

        distance_to_grid_points = self.get_distance_to_grid_points(point)
        return mat_argmin(distance_to_grid_points)

    def indices_to_point(self, inds):
        """Returns a length-3 vector with the coordinates of the point with 2D
indices `inds`."""

        return self.t_coordinates[:, inds[0], inds[1]]

    def random_point_in_the_area(self):
        # The implementation may be improved
        x_coord = np.random.uniform((min(self.t_coordinates[0, 0])),
                                    (max(self.t_coordinates[0, 0])))
        y_coord = np.random.uniform((min(self.t_coordinates[1, :, 0])),
                                    (max(self.t_coordinates[1, :, 0])))
        z_coord = self.t_coordinates[2, 0, 0]

        return np.array([x_coord, y_coord, z_coord])

    def random_points_in_the_area(self, num_points=1):
        """Returns a num_points x 3 matrix. Each row is a random
        point drawn independently from an independent distribution.

        FUT: use this function in random_point_in_the_area
        """

        x_coord = np.random.uniform(low=self.min_x(),
                                    high=self.max_x(),
                                    size=(num_points, ))
        y_coord = np.random.uniform(low=self.min_y(),
                                    high=self.max_y(),
                                    size=(num_points, ))
        z_coord = self.z_value() * np.ones((num_points, ))

        return np.array((x_coord, y_coord, z_coord)).T

    def random_point_in_the_grid(self):

        x_coord = np.random.choice(self.t_coordinates[0, 0])
        y_coord = np.random.choice(self.t_coordinates[1, :, 0])
        z_coord = self.t_coordinates[2, 0, 0]

        return np.array([x_coord, y_coord, z_coord])

    def random_points_in_the_grid(self, num_points=1):
        """Returns a length grid points"""

        m_points = np.full((3, num_points), fill_value=None, dtype=float)
        for ind_point in range(num_points):
            m_points[:, ind_point] = self.random_point_in_the_grid()

        return m_points.T

    # def random_point_in_the_grid_outside_building(self, m_building_metadata=None):
    #     """Returns a grid point outside the building in the [x, y, z] form"""
    #     if m_building_metadata is None:
    #         return self.random_point_in_the_grid()
    #
    #     l_indices = np.where(m_building_metadata == 0)
    #     ind = np.random.randint(0, int(len(l_indices[0])))
    #
    #     random_grid_indices_outside_building = (l_indices[0][ind], l_indices[1][ind])
    #     return np.array(self.t_coordinates[:, random_grid_indices_outside_building[0],
    #                     random_grid_indices_outside_building[1]])

    def random_points_in_the_grid_outside_buildings(self, num_points=1, m_building_metadata=None):
        """Returns a num_sources x 3 matrix containing
        grid point outside the building in the 3D coordinate form"""
        if m_building_metadata is None:
            return self.random_points_in_the_grid(num_points)

        # m_points = np.full((num_points, 3), fill_value=None, dtype=float)
        # for ind_point in range(num_points):
        #
        #     l_indices = np.where(m_building_metadata == 0)
        #     ind = np.random.randint(0, int(len(l_indices[0])))
        #
        #     random_grid_indices_outside_building = (l_indices[0][ind], l_indices[1][ind])
        #     m_points[ind_point, :] = np.array(self.t_coordinates[:, random_grid_indices_outside_building[0],
        #                 random_grid_indices_outside_building[1]])
        # return m_points
        else:
            v_indices_to_sample_from = self.random_grid_points_inds_outside_buildings(num_points=num_points,
                                                           m_building_metadata=m_building_metadata)
            m_grid_points = self.t_coordinates[:,
                   np.unravel_index(v_indices_to_sample_from, self.t_coordinates[0].shape)[0],
                   np.unravel_index(v_indices_to_sample_from, self.t_coordinates[0].shape)[1]]
            m_grid_points = m_grid_points.T
            return m_grid_points

    def random_grid_points_inds_outside_buildings(self, num_points=1, m_building_metadata=None):
        """
        Args:
            `m_building_metadata`: Ny x Nx matrix whose (i,j)-th entry is 1
            if the (i, j)-th grid point is inside a building.

        Returns:
            `v_indices_to_sampled_from`: a vector of length num_points containing
            point-wise indices of grid points outside building.
        """

        v_meta_data = m_building_metadata.flatten()
        # indices of grid points that do not include building
        relevant_ind = np.where(v_meta_data == 0)[0]

        if num_points > len(relevant_ind):
            num_points = len(relevant_ind)

        v_indices_to_sampled_from = np.random.choice(relevant_ind,
                                                   size=num_points,
                                                   replace=False)

        return v_indices_to_sampled_from

    def all_grid_points_in_matrix_form(self):
        """ Returns a matrix of size total_number_of_grid_points x 3 matrix
            that contains all the 3D coordinates in the grid"""
        l_grid_loc = []
        for ind_row in range(self.t_coordinates.shape[1]):
            for ind_col in range(self.t_coordinates.shape[2]):
                l_grid_loc.append(self.t_coordinates[:, ind_row, ind_col])
        return np.array(l_grid_loc)

    def convert_grid_meta_data_to_standard_form(self,
                                                m_meta_data):
        """Convert grid point indices whose entry is 1 to  3D locations.
        Args:
             - `m_meta_data`: a num_points_y x num_points_x matrix
                    whose (i,j)-th entry is 1 represents a building location or
                    sample locations depending which matrix is passed in the argument.

        Returns:
            m_grid_coordinates = num_grid_points_inside x 3 matrix
            whose i-th entry is a 3D location where (i, j)-th entry of m_meta_data
            is 1."""

        m_grid_coordinates = None

        l_meta_data_inds = np.where(m_meta_data == 1)
        for ind_meta_data_inds in zip(l_meta_data_inds[0],
                                           l_meta_data_inds[1]):

            v_meta_data_coord = self.indices_to_point(ind_meta_data_inds)

            if m_grid_coordinates is None:
                m_grid_coordinates = v_meta_data_coord[None, :]
            else:
                m_grid_coordinates = np.concatenate((m_grid_coordinates,
                                                            v_meta_data_coord[None, :]),
                                                           axis=0)

        return m_grid_coordinates

    def convert_measurements_to_grid_form(self, m_all_measurements_loc,
                                          m_all_measurements):
        """
        Args:
            -`m_all_measurements`: num_sources x num_measurements
            buffered measurements matrix whose (i, j)-th
           entry denotes the received power at j-th m_all_measurement_loc
           transmitted by i-th power source.

           -`m_all_measurement_loc`: 3 x num_measurements
           buffered measurement locations matrix whose
           j-th column represents (x, y, z) coordinate of measurement location

        Returns:
            _`t_all_measurements_grid`: a tensor of shape
            num_sources x num_gird_points_y x num_grid_points_x
            -`t_mask_with_meta_data`: 1 x num_gird_points_y x num_grid_points_x binary mask whose
            entry is 1 at the grid point where measurement is taken,
            -1 at the grid point that is inside a building,
            0 otherwise.

        """
        num_sources = m_all_measurements.shape[0]

        t_all_measurements_grid = np.zeros(
            (num_sources, self.num_points_y, self.num_points_x))

        t_mask = np.zeros(
            (1, self.num_points_y, self.num_points_x))

        m_all_measurements_loc_trans = m_all_measurements_loc.T

        m_all_measurements_col_index = 0  # to iterate through column of measurements

        # buffer counter to count repeated measurement in the grid point
        m_counter = np.zeros(np.shape(t_mask))

        for v_measurement_loc in m_all_measurements_loc_trans:

            # Find the nearest indices of the grid point closet to v_measurement_loc
            v_meas_loc_inds = self.nearest_gridpoint_inds(v_measurement_loc)

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
            m_counter[0, v_meas_loc_inds[0], v_meas_loc_inds[1]] += 1

            # set the value of mask to 1 at the measured grid point indices
            t_mask[0, v_meas_loc_inds[0],
                   v_meas_loc_inds[1]] = 1

            m_all_measurements_col_index += 1

        # Average the measurements
        t_all_measurements_grid = np.divide(t_all_measurements_grid,
                                           m_counter,
                                           where=m_counter != 0,
                                           out=np.zeros(np.shape(
                                               t_all_measurements_grid)))

        # mask whose (j, k) entry is 1 if sampled taken, 0 if not taken,
        # and -1 if it is inside the building
        # t_mask_with_meta_data = t_mask - self.m_building_meta_data_grid[None, :, :]

        t_mask_with_meta_data = t_mask #- self.m_building_meta_data_grid[None, :, :]
        return t_all_measurements_grid, t_mask_with_meta_data

    def min_x(self):
        return self.t_coordinates[0, 0, 0]

    def max_x(self):
        return self.t_coordinates[0, 0, -1]

    def min_y(self):
        return self.t_coordinates[1, -1, 0]

    def max_y(self):
        return self.t_coordinates[1, 0, 0]

    def z_value(self):
        return self.t_coordinates[2, 0, 0]

    def num_points_x(self):
        return self.t_coordinates.shape[2]

    def num_points_y(self):
        return self.t_coordinates.shape[1]

    @property
    def dim(self):
        """Dimension of the grid."""
        return 3


class MapGenerator():

    def __init__(self, combine_channels=False):

        self.combine_channels = combine_channels

    def generate_map(self):

        """If self.combine_channels == False:

        returns a num_sources x grid.num_points_x x grid.num_points_y
        tensor, where the entry [i,j,k] is the power rx from source i
        at grid point (j,k).

        `m_meta_data`, a Ny x Nx matrix whose [i,j] element is 1 if
        the (i, j)th grid point is inside the building.

        Else:

        returns a 1 x grid.num_points_x x grid.num_points_y tensor,
        where the entry [0,j,k] is the aggregate power rx from all
        sources at grid point (j,k).

        `m_meta_data`, a Ny x Nx matrix whose [i,j] element is 1 if
        the (i, j)th grid point is inside the building.
        """

        source_maps, m_meta_data = self._generate_map()
        
        if self.combine_channels:
            return natural_to_dB(np.sum(dB_to_natural(source_maps), axis=0)[None, :, :]), m_meta_data
        else:
            return source_maps, m_meta_data

########################
# grid = RectangularGrid(
#             gridpoint_spacing=3,
#             num_points_x=32,  # 20,  #12
#             num_points_y=32,  # 18,  #10
#             height=20)
#
# m_metadata= np.zeros(shape=(32, 32))
# a = grid.random_point_in_the_grid_outside_building(num_points=5, m_building_metadata=m_metadata)
# print(a, a.shape)
