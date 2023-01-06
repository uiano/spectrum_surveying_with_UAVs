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

