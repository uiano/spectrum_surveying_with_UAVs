from IPython.core.debugger import set_trace
import dill
import tensorflow as tf
import numpy as np
import gsim

from map_generators.map_generator import Grid
from map_generators.map_generator import MapGenerator
from measurement_generation.sampler import Sampler


class MeasurementDataset():
    """Generation and storage of data sets of measurements of radio maps.

    Each dataset comprises a collection of measurement records. A
    measurement record is a collection of measurements of a certain
    radio map.

    The purpose of this class is to save time when runing simulations. 

    Data is stored internally in self._data as lists. A
    tf.data.Dataset is produced when the user requests that data if
    `tf_dataset` is True. The tf.data.Dataset is not used to store the
    data internally because it cannot be dilled or pickled. Storing it
    is complicated, and the approach in
    https://www.tensorflow.org/api_docs/python/tf/data/experimental/TFRecordWriter
    does not work because the data set comprises tuples.

    """
    def __init__(self, grid, map_generator, sampler, tf_dataset=True):

        assert isinstance(grid, Grid)
        assert isinstance(map_generator, MapGenerator)
        if sampler is not None:
            assert isinstance(sampler, Sampler)

        self.grid = grid
        self.map_generator = map_generator
        self.sampler = sampler
        self.tf_dataset = tf_dataset

        self._data = dict()
        # self.data["train"] = []
        # self.data["test"] = []

    @classmethod
    def generate(cls,
                 grid,
                 map_generator,
                 num_measurements_per_map,
                 num_maps_train,
                 num_maps_test,
                 sampler=None,
                 num_blocks_per_map=20,
                 tf_dataset=True):
        """Args:

            num_blocks_per_map: number of blocks of
                `num_measurements_per_map` samples that are drawn from
                each generated map. This is useful since the operation
                that consumes most time is the generation of the map.

            tf_dataset: if True, the data is stored in
                tf.data.Datasets. Else, it is stored in lists.

        Returns a dictionary with two keys:

        "train": length-num_maps_train*num_blocks_per_map list or
        Dataset of measurement blocks.

        "test": length-num_maps_test*num_blocks_per_map list or
        Dataset of measurement blocks.

        Each measurement block is a tuple (t_input, t_target).

        """

        md = cls(grid, map_generator, sampler, tf_dataset)

        print(f"Generating {num_maps_train} training maps...")
        md._data["train"] = cls._gen_data(
            grid=grid,
            map_generator=map_generator,
            sampler=sampler,
            num_measurements=num_measurements_per_map,
            num_maps=num_maps_train,
            num_blocks_per_map=num_blocks_per_map)

        print(f"Generating {num_maps_test} testing maps...")
        md._data["test"] = cls._gen_data(
            grid=grid,
            map_generator=map_generator,
            sampler=sampler,
            num_measurements=num_measurements_per_map,
            num_maps=num_maps_test,
            num_blocks_per_map=num_blocks_per_map)

        return md

    def get_data(self, train_or_test):
        """ Args:

            `train_or_test` can be "train" or "test". """
        def adapt(data):
            if self.tf_dataset:
                t_data = ([x for x, _ in data], [y for _, y in data])
                return tf.data.Dataset.from_tensor_slices(t_data)
            else:
                return data

        return adapt(self._data[train_or_test])

    def save(self, filename):

        assert isinstance(filename, str)

        # d_objs = {
        #     "grid": grid,
        #     "map_generator": map_generator,
        #     "sampler": sampler,
        #     "l_measurement_data": l_measurement_data
        # }
        
        dill.dump(self, open(filename, "wb"))
        print(f"MeasurementDataset saved to file '{filename}'.")

    def load(filename):

        obj = dill.load(open(filename, "rb"))

        obj.filename = filename

        # backwards compatibility
        if hasattr(obj, "data"):
            obj.tf_dataset = True
            obj._data = obj.data

        return obj

    def _gen_data(grid,
                  map_generator,
                  sampler,
                  num_measurements,
                  num_maps,
                  num_blocks_per_map=1):
        """List of tuples (m_locations, m_measurements). Each tuple contains
        measurements collected in a different environment. The rows of
        each matrix correspond to a measurement.

        num_blocks_per_map: number of blocks of num_measurements
        samples that are drawn from each generated map. This is useful
        since the operation that consumes most time is the generation
        of the map.

        """
        if sampler is None:
            raise NotImplementedError("Provide sampler in generate method()")

        l_env = []
        for ind_maps in range(num_maps):

            if num_maps > 50 and ind_maps and ind_maps % 50 == 0:
                print(f"{ind_maps} maps generated")

            map, _ = map_generator.generate_map()

            for ind_blocks_per_map in range(num_blocks_per_map):
                m_locations = grid.random_points_in_the_area(
                    num_points=num_measurements)
                m_measurements = sampler.sample_map_multiple_loc(
                    map, m_locations)

                l_env.append((m_locations, m_measurements))

        return l_env


class GridMeasurementDataset(MeasurementDataset):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @classmethod
    def generate(cls,
                 grid,
                 map_generator,
                 num_measurements_per_map,
                 num_maps_train,
                 num_maps_test,
                 sampler=None,
                 num_blocks_per_map=20,
                 tf_dataset=True,
                 route_planner=None,
                 ):
        """Args:

            num_blocks_per_map: number of blocks of
                `num_measurements_per_map` samples that are drawn from
                each generated map. This is useful since the operation
                that consumes most time is the generation of the map.

            tf_dataset: if True, the data is stored in
                tf.data.Datasets. Else, it is stored in lists.

        Returns a dictionary with two keys:

        "train": length-num_maps_train*num_blocks_per_map list or
        Dataset of measurement blocks.

        "test": length-num_maps_test*num_blocks_per_map list or
        Dataset of measurement blocks.

        Each measurement block is a tuple (t_input, t_target).

        """

        md = cls(grid, map_generator, sampler, tf_dataset)

        print(f"Generating {num_maps_train} training maps...")
        md._data["train"] = cls._gen_data(
            grid=grid,
            map_generator=map_generator,
            sampler=sampler,
            num_measurements=num_measurements_per_map,
            num_maps=num_maps_train,
            num_blocks_per_map=num_blocks_per_map,
            route_planner=route_planner,
            )

        print(f"Generating {num_maps_test} testing maps...")
        md._data["test"] = cls._gen_data(
            grid=grid,
            map_generator=map_generator,
            sampler=sampler,
            num_measurements=num_measurements_per_map,
            num_maps=num_maps_test,
            num_blocks_per_map=num_blocks_per_map,
            route_planner=route_planner,
            )

        return md

    def get_data(self, train_or_test):
        """ Args:

            `train_or_test` can be "train" or "test". """

        def adapt(data):
            if self.tf_dataset:
                l_x_data, l_y_data = ([x for x, _ in data], [y for _, y in data])
                return tf.data.Dataset.from_tensor_slices((l_x_data, l_y_data))
            else:
                return data

        return adapt(self._data[train_or_test])

    def _gen_data(grid,
                  map_generator,
                  num_measurements,
                  num_maps,
                  sampler,
                  num_blocks_per_map=1,
                  route_planner=None,
                  **kwargs):

        """
        Args:
            num_measurements: it can be an integer or tuple of integer (min, max).
            Single integer denotes the number measurements to be taken in each generated map.
            If tuple, then number of measurements is between min and max.

            num_blocks_per_map: number of blocks of num_measurements
            samples that are drawn from each generated map. This is useful
            since the operation that consumes most time is the generation
            of the map.

        Returns:

            l_env: List of tuples (t_sampled_map, t_true_map).The`t_sampled_map` is a
            (num_sources + 1) x num_point_y x num_point_x tensor and
            `t_true_map` is a num_sources x num_point_y x num_point_x tensor




        """

        l_env = []
        for ind_maps in range(num_maps):

            if num_maps > 50 and ind_maps and ind_maps % 50 == 0:
                print(f"{ind_maps} maps generated")

            "True Map"
            map, m_meta_data = map_generator.generate_map()

            # total_grid_points = map.shape[1] * map.shape[2]

            for ind_blocks_per_map in range(num_blocks_per_map):

                if type(num_measurements) == tuple:
                    num_measurements_random = np.random.randint(num_measurements[0],
                                                         num_measurements[1] + 1)
                else:
                    num_measurements_random = num_measurements

                if route_planner is not None:
                    assert route_planner
                    assert sampler

                    m_all_measurements = None
                    m_all_measurement_loc = None # num_measurements x 3
                    num_sources = map.shape[0]
                    # store measurement locations
                    for ind_measurements in range(num_measurements_random):
                        v_measurement_location = route_planner.next_measurement_location(map, m_meta_data)
                        # v_measurement = sampler.sample_map(map, v_measurement_location)

                        if m_all_measurement_loc is None:
                            # m_all_measurement_loc = np.reshape(v_measurement_location, (3, 1))
                            m_all_measurement_loc = np.reshape(v_measurement_location, (1, 3))

                        else:
                            m_all_measurement_loc = np.vstack(
                                (m_all_measurement_loc,
                                 np.reshape(v_measurement_location, (1, 3))))

                    t_sampled_map, t_mask = sampler.sampled_map_multiple_grid_loc_old(t_true_map=map,
                                                                                 m_sample_locations=m_all_measurement_loc)
                    t_mask = t_mask - m_meta_data[None, ...]
                    sampled_map_with_mask = np.concatenate((t_sampled_map, t_mask), axis=0)

                else:
                    assert sampler

                    # " collect/sample measurements on the grid"
                    # m_sample_locations = grid.random_point_in_the_grid_outside_building(
                    #     num_points=num_measurements, m_building_metadata=m_meta_data)

                    # sample measurements on the grid using the sampler
                    t_sampled_map, t_mask = sampler.sampled_map_multiple_grid_loc(
                        t_true_map=map, num_measurements=num_measurements,
                        m_building_metadata=m_meta_data)

                    t_mask = t_mask - m_meta_data[None, ...]

                    sampled_map_with_mask = np.concatenate((t_sampled_map, t_mask), axis=0)

                t_true_map = map
                t_sampled_map = sampled_map_with_mask
                l_env.append((t_sampled_map.astype('float32'),
                              t_true_map.astype('float32')))

        return l_env


class GridMeasurementPosteriorDataset(GridMeasurementDataset):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @classmethod
    def generate(cls,
                 grid,
                 map_generator,
                 num_measurements_per_map,
                 num_maps_train,
                 num_maps_test,
                 sampler=None,
                 num_blocks_per_map=20,
                 tf_dataset=True,
                 estimator=None):
        """Args:

            num_blocks_per_map: number of blocks of
                `num_measurements_per_map` samples that are drawn from
                each generated map. This is useful since the operation
                that consumes most time is the generation of the map.

            tf_dataset: if True, the data is stored in
                tf.data.Datasets. Else, it is stored in lists.

        Returns a dictionary with two keys:

        "train": length-num_maps_train*num_blocks_per_map list or
        Dataset of measurement blocks.

        "test": length-num_maps_test*num_blocks_per_map list or
        Dataset of measurement blocks.

        Each measurement block is a tuple (t_input, t_target).

        """

        md = cls(grid, map_generator, sampler, tf_dataset)

        print(f"Generating {num_maps_train} training maps...")
        md._data["train"] = cls._gen_data(
            grid=grid,
            map_generator=map_generator,
            sampler=sampler,
            num_measurements=num_measurements_per_map,
            num_maps=num_maps_train,
            num_blocks_per_map=num_blocks_per_map,
            estimator=estimator)

        print(f"Generating {num_maps_test} testing maps...")
        md._data["test"] = cls._gen_data(
            grid=grid,
            map_generator=map_generator,
            sampler=sampler,
            num_measurements=num_measurements_per_map,
            num_maps=num_maps_test,
            num_blocks_per_map=num_blocks_per_map,
            estimator=estimator)

        return md

    def get_data(self, train_or_test):
        """ Args:

            `train_or_test` can be "train" or "test". """

        def adapt(data):
            if self.tf_dataset:
                l_x_data, l_y_data = ([x for x, _ in data], [y for _, y in data])
                return tf.data.Dataset.from_tensor_slices((l_x_data, l_y_data))
            else:
                return data

        return adapt(self._data[train_or_test])


    def _gen_data(grid,
                  map_generator,
                  num_measurements,
                  num_maps,
                  num_blocks_per_map=1,
                  estimator=None,
                  **kwargs):

        """
        Args:
            num_measurements: it can be an integer or tuple of integer (min, max).
            Single integer denotes the number measurements to be taken in each generated map.
            If tuple, then number of measurements is between min and max.

            num_blocks_per_map: number of blocks of num_measurements
            samples that are drawn from each generated map. This is useful
            since the operation that consumes most time is the generation
            of the map.

        Returns:

            l_env: List of tuples (t_sampled_map, t_true_posterior_target).The`t_sampled_map` is a
            (num_sources + 1) x num_point_y x num_point_x tensor and
            `t_true_posterior_target` is a 2*num_sources x num_point_y x num_point_x tensor where
                first num_sources is a posterior mean and remaining num_sources is a posterior standard
                deviation.




        """

        assert estimator is not None
        l_env = []
        for ind_maps in range(num_maps):

            if num_maps > 50 and ind_maps and ind_maps % 50 == 0:
                print(f"{ind_maps} maps generated")

            "True Map"
            map, m_meta_data = map_generator.generate_map()

            total_grid_points = map.shape[1] * map.shape[2]

            for ind_blocks_per_map in range(num_blocks_per_map):

                if type(num_measurements) == tuple:
                    num_measurements_random = np.random.randint(num_measurements[0],
                                                                num_measurements[1] + 1)
                else:
                    num_measurements_random = num_measurements
                # sample in the grid with sample t_mask filter
                t_sampled_map, t_mask = GridMeasurementPosteriorDataset.get_sample_map_from_sample_mask(t_true_map=map,
                                                                        total_grid_points=total_grid_points,
                                                                        num_measurements=num_measurements_random,
                                                                        m_meta_data=m_meta_data)

                # " collect/sample measurements in the grid"
                # m_sample_locations = grid.random_points_in_the_grid(num_points=num_measurements)
                # #
                # # # sample measurements in the grid using the sampler
                # t_sampled_map, t_mask = sampler.sampled_map_multiple_grid_loc(
                #     t_true_map=map,
                #     m_sample_locations=m_sample_locations)

                sampled_map_with_mask = np.concatenate((t_sampled_map, t_mask), axis=0)
                estimator.reset()
                t_mask_withou_building_data = np.where(t_mask == -1, 0, t_mask)

                d_map_estimate = estimator.estimate(measurement_locs=t_mask_withou_building_data[0],
                                                    measurements=t_sampled_map,
                                                    building_meta_data=m_meta_data)
                posterior_mean = d_map_estimate["t_power_map_estimate"]
                posterior_std = np.sqrt(d_map_estimate["t_power_map_norm_variance"] \
                                * estimator.f_shadowing_covariance(0))

                t_true_posterior_target = np.concatenate((posterior_mean, posterior_std),
                                                         axis=0)

                t_sampled_map = sampled_map_with_mask
                l_env.append((t_sampled_map.astype('float32'),
                              t_true_posterior_target.astype('float32')))

        return l_env

    def get_sample_map_from_sample_mask(t_true_map,
                                        total_grid_points,
                                        num_measurements,
                                        m_meta_data):

        m_mask = np.zeros((t_true_map.shape[1],
                           t_true_map.shape[2]))

        v_mask = m_mask.flatten('C')
        v_meta_data = m_meta_data.flatten('C')

        # indices of grid points that do not include building
        relevant_ind = np.where(v_meta_data == 0)[0]

        if num_measurements > len(relevant_ind):
            num_measurements = len(relevant_ind)

        indices_to_sampled_from = np.random.choice(relevant_ind,
                                                   size=num_measurements,
                                                   replace=False)

        # set the mask value to 1 at the sampled grid point indices
        v_mask[indices_to_sampled_from] = 1

        m_mask = v_mask.reshape(m_mask.shape,
                                order='C')

        t_mask = m_mask[None, :, :]

        # Sampled map
        t_sampled_map = t_true_map * t_mask

        # mask whose (j, k) entry is 1 if sampled taken, 0 if not taken,
        # and -1 if it is inside the building

        t_mask_with_meta_data = t_mask - m_meta_data[None, :, :]

        return t_sampled_map, t_mask_with_meta_data


################################old methods #######################################
        def get_sample_map_from_sample_mask(t_true_map,
                                            total_grid_points,
                                            num_measurements,
                                            m_meta_data):

            m_mask = np.zeros((t_true_map.shape[1],
                               t_true_map.shape[2]))

            v_mask = m_mask.flatten('C')
            v_meta_data = m_meta_data.flatten('C')

            # indices of grid points that do not include building
            relevant_ind = np.where(v_meta_data == 0)[0]

            if num_measurements > len(relevant_ind):
                num_measurements = len(relevant_ind)

            indices_to_sampled_from = np.random.choice(relevant_ind,
                                                       size=num_measurements,
                                                       replace=False)

            # set the mask value to 1 at the sampled grid point indices
            v_mask[indices_to_sampled_from] = 1

            m_mask = v_mask.reshape(m_mask.shape,
                                    order='C')

            # # generate num_measurements of 1 and remaining 0
            # array_1_or_0 = np.array([1] * num_measurements +
            #                         [0] * (total_grid_points - num_measurements))
            #
            # # shuffle the 1's and 0's
            # np.random.shuffle(array_1_or_0)
            # m_mask = array_1_or_0.reshape(t_true_map.shape[1],
            #                               t_true_map.shape[2])

            t_mask = m_mask[None, :, :]

            # Sampled map
            t_sampled_map = map * t_mask

            # mask whose (j, k) entry is 1 if sampled taken, 0 if not taken,
            # and -1 if it is inside the building

            t_mask_with_meta_data = t_mask - m_meta_data[None, :, :]

            return t_sampled_map, t_mask_with_meta_data

        def convert_measurements_to_grid_form(m_all_measurements_loc, m_all_measurements, m_meta_data):
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
                (num_sources, grid.num_points_y, grid.num_points_x))

            t_mask = np.zeros(
                (1, grid.num_points_y, grid.num_points_x))

            m_all_measurements_loc_trans = m_all_measurements_loc.T

            m_all_measurements_col_index = 0  # to iterate through column of measurements

            # buffer counter to count repeated measurement in the grid point
            m_counter = np.zeros(np.shape(t_mask))

            for v_measurement_loc in m_all_measurements_loc_trans:
                # Find the nearest indices of the grid point closet to v_measurement_loc
                v_meas_loc_inds = grid.nearest_gridpoint_inds(v_measurement_loc)

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
            t_mask_with_meta_data = t_mask - m_meta_data[None, :, :]

            return t_all_measurements_grid, t_mask_with_meta_data
