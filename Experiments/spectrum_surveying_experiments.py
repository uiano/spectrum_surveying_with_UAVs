import gsim
from gsim.gfigure import GFigure
import tensorflow as tf
from map_generators.map_generator import RectangularGrid
from map_generators.correlated_shadowing_generator import CorrelatedShadowingGenerator
from measurement_generation.sampler import InterpolationSampler
from measurement_generation.measurement_dataset import MeasurementDataset, \
    GridMeasurementDataset, GridMeasurementPosteriorDataset
from map_estimators.kriging_estimator import BatchKrigingEstimator, OnlineKrigingEstimator
from map_estimators.interpolation_estimators import KNNEstimator, \
    KernelRidgeRegressionEstimator, MultikernelEstimator
from map_estimators.neural_network_estimator import NeuralNetworkEstimator, \
    FullyConvolutionalNeuralNetwork, ConvolutionalVAE
from route_planners.route_planner import RandomPlanner, \
    MinimumCostPlanner, UniformRandomSamplePlanner, \
    SquareSpiralGridPlanner,\
    GridPlanner, IndependentUniformPlanner
from simulators.simulate_surveying_map import simulate_surveying_map, simulate_surveying_montecarlo, \
    plot_metrics, compare_maps, plot_map
from IPython.core.debugger import set_trace
from tensorflow.keras.layers import Dense, Flatten, Conv2D, \
    MaxPool1D, Input, GlobalMaxPooling1D, Conv2DTranspose, Reshape, Conv1D
from map_estimators.fully_conv_nn_arch import ForkConvNnArch14, ForkConvNnArchExample, \
    ForkConvNnArch16, FullyConvNnArch13, ForkConvNnArch17, \
    ForkConvNn, ForkConvNnArch18, ForkConvNnArchExample2, SkipConvNnArch20
from tensorflow.keras import Model
import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import dill as pickle
import seaborn as sns
import scipy
from tensorflow import keras
from map_generators.insite_map_generator import InsiteMapGenerator
from util.communications import db_to_dbm
from utilities import dbm_to_watt, watt_to_dbm, empty_array
from matplotlib.lines import Line2D
tfd = tfp.distributions

# tf.config.experimental_run_functions_eagerly(False)
datafolder = "train_data/datasets/"
trainedfolder = "train_data/trained/"
weightfolder = "./saved_weights/"
lossfolder = "./train_data/"

data_file_string = './train_data/mydata_with_mask_04.pkl'
# weights_file_string = './saved_weights/ckpt_100_corr'
#loss_file_string = './train_data/loss_100_corr.pkl'

print('tensorflow version: ', tf.__version__)

class ExperimentSet(gsim.AbstractExperimentSet):

    """
    Experiment for Gudmundson Dataset Generation.

    """

    def experiment_2001(args):

        # 0. Grid
        grid = RectangularGrid(
            gridpoint_spacing=3,
            num_points_x=32,  # 20,  #12
            num_points_y=32,  # 18,  #10
            height=20)

        # 1. Map generator
        num_sources = 2
        shadowing_std = np.sqrt(10)
        shadowing_correlation_dist = 50 # 13.5  # values as in the autoencoder paper
        # source_power = 30 * np.ones(shape=(num_sources, ))
        source_power = np.tile([10, 10], (num_sources, 1))
        f_shadowing_covariance = CorrelatedShadowingGenerator.gudmundson_correlation(
            shadowing_std=shadowing_std,  # 1
            shadowing_correlation_dist=shadowing_correlation_dist  # 40, #15
        )
        map_generator = CorrelatedShadowingGenerator(
            grid=grid,
            source_height=20,
            source_power=source_power,
            frequency=2.4e9,
            f_shadowing_covariance=f_shadowing_covariance,
            combine_channels=True)

        # 2. Sampling
        sampler = InterpolationSampler(
            grid,
            interpolation_method="in_grid",
        )

        def generate_dataset(num_maps_train=100,
                             num_maps_test=25,
                             num_measurements_per_map=(1, 100),
                             num_blocks_per_map=50):

            md = GridMeasurementDataset.generate(
                grid=grid,
                map_generator=map_generator,
                sampler=sampler,
                num_measurements_per_map=num_measurements_per_map,
                num_maps_train=num_maps_train,
                num_maps_test=num_maps_test,
                num_blocks_per_map=num_blocks_per_map)

            filename = (
                    datafolder + f"grid_{grid.num_points_y}_x_{grid.num_points_x}"
                                 f"-shadowing_dist_{shadowing_correlation_dist}"
                                 f"-gudmundson-{num_sources}_combined_sources"
                                 f"-{num_maps_train}_training_maps-{num_measurements_per_map}_measurements"
                                 f"-{num_blocks_per_map}_blocks_per_map"
                                 f"-shadowing_std_{shadowing_std:.2f}.dill")

            md.save(filename)

        generate_dataset(num_maps_train=5,
                         num_maps_test=1,
                         num_measurements_per_map=100,
                         num_blocks_per_map=1)
        # generate_dataset(num_maps_train=5000,
        #                  num_maps_test=1000,
        #                  num_measurements_per_map=(1, 20),
        #                  num_blocks_per_map=300)
        # generate_dataset(num_maps_train=5000,
        #                  num_maps_test=1000,
        #                  num_measurements_per_map=(20, 50),
        #                  num_blocks_per_map=200)
        # generate_dataset(num_maps_train=5000,
        #                  num_maps_test=1000,
        #                  num_measurements_per_map=(50, 100),
        #                  num_blocks_per_map=100)
        # generate_dataset(num_maps_train=5000,
        #                  num_maps_test=1000,
        #                  num_measurements_per_map=(100, 200),
        #                  num_blocks_per_map=50)

        return

    """ MinimumCostPlanner experiment for the 
    gudmundson dataset to plot the map 
    estimate and the uncertainty metric along 
    with the trajectory."""
    def experiment_20051(args):
        ''' region of interest is square with l =100, speed of EM (C), Freq, Po, antenna height
            'grid_length' is length of square region of interest
            ' grid_size is the spacing of grid in ROI
            'v_source_power' is the transmitted power in watt
        '''

        # 0. Grid
        grid = RectangularGrid(
            gridpoint_spacing=3,
            num_points_x=32,  # 20,  #12
            num_points_y=32,  # 18,  #10
            height=20)

        # 1. Map generator
        num_sources = 2
        shadowing_std = np.sqrt(10)
        shadowing_correlation_dist = 50 # 13.5  # values as in the autoencoder paper
        # source_power = 30 * np.ones(shape=(num_sources, ))

        source_height = 20
        m_source_locations = grid.random_points_in_the_area(num_sources)
        m_source_locations = np.array([[40.32, 70.34, 0], [60.23, 30.12, 0]])
        m_source_locations[:, 2] = source_height
        v_source_power = 10 * np.ones(shape=(num_sources,))

        source_power = np.tile([10, 10], (num_sources, 1))
        f_shadowing_covariance = CorrelatedShadowingGenerator.gudmundson_correlation(
            shadowing_std=shadowing_std,  # 1
            shadowing_correlation_dist=shadowing_correlation_dist  # 40, #15
        )

        map_generator = CorrelatedShadowingGenerator(
            grid=grid,
            # source_height=20,
            # source_power=source_power,
            v_source_power=v_source_power,
            m_source_locations=m_source_locations.T,
            frequency=2.4e9,
            f_shadowing_covariance=f_shadowing_covariance,
            combine_channels=True)

        # 2. Sampling
        initial_location = grid.indices_to_point((0, 0))
        dist_between_measurements = 5
        dist_between_waypoints = 30
        l_route_planners = [
            GridPlanner(grid=grid,
                        initial_location=initial_location,
                        dist_between_measurements=dist_between_measurements),

            SquareSpiralGridPlanner(grid=grid,
                                    initial_location=initial_location,
                                    dist_between_measurements=dist_between_measurements),
            IndependentUniformPlanner(grid=grid,
                                      initial_location=initial_location,
                                      dist_between_measurements=dist_between_measurements),
            RandomPlanner(grid=grid,
                          initial_location=initial_location,
                          dist_between_measurements=dist_between_measurements,
                          dist_between_waypoints=dist_between_waypoints),
            IndependentUniformPlanner(
                grid=grid,
                initial_location=initial_location,
                dist_between_measurements=dist_between_measurements),
            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               dist_between_measurements=dist_between_measurements,
                               num_measurement_to_update_destination=10,
                              ),
            UniformRandomSamplePlanner(grid=grid,
                                            initial_location=initial_location,
                                            dist_between_measurements=dist_between_measurements)
        ]

        # 3. Sampler
        sampler = InterpolationSampler(grid,
                                       interpolation_method='in_grid')

        # 4. Estimator

        min_service_power = -30
        model = SkipConvNnArch20()
        nn_estimator = NeuralNetworkEstimator(grid=grid,
                                              f_shadowing_covariance=f_shadowing_covariance,
                                              min_service_power=min_service_power,
                                              estimator=model)

        # weights_file_string = (weightfolder + "(1, 100)_meas_nn_arch_18_alpha=0_epochs=3_samp_scale=0.97_out_same=True")
        # weights_file_string = (weightfolder + "(1, 100)_meas_nn_arch_18_alpha=1_epochs=6_samp_scale=0.76_out_same=False")
        weights_file_string = (weightfolder +
                               "nn_arch_20_(1, 10)_meas_[4]_epochs__samp_scale=0.99_out_same_True_[1]_alpha.ckpt")
        weights_file_string = (weightfolder +
                               "(1, 100)_meas_nn_arch_20_alpha=1_epochs=3_samp_scale=0.973_out_same=True")
        nn_estimator.estimator.load_weights(weights_file_string)

        estimators = [BatchKrigingEstimator(
            grid,
            min_service_power=min_service_power,
            f_shadowing_covariance=f_shadowing_covariance,
            f_mean=map_generator.mean_at_channel_and_location),

            OnlineKrigingEstimator(
                grid,
                min_service_power=min_service_power,
                f_shadowing_covariance=f_shadowing_covariance,
                f_mean=map_generator.mean_at_channel_and_location),

            nn_estimator,

            KNNEstimator(grid=grid,
                         f_shadowing_covariance=f_shadowing_covariance),
            KernelRidgeRegressionEstimator(
                grid=grid,
                f_shadowing_covariance=f_shadowing_covariance),

            # MultikernelEstimator(grid=grid,
            #                      f_shadowing_covariance=f_shadowing_covariance)
        ]

        # Simulation

        def experiment_1a():
            """Plot maps after every measurement."""
            global b_do_not_plot_power_maps
            b_do_not_plot_power_maps = False  # True
            route_planner = l_route_planners[5]
            route_planner.debug_level = 0
            map, m_building_data = map_generator.generate_map()
            # for i in range(10):
            #     m_source_locations = grid.random_points_in_the_area(num_sources)
            #     m_source_locations[:, 2] = source_height
            #     map_generator.m_source_locations = m_source_locations.T
            #     map, m_building_data = map_generator.generate_map()
            #     print(map_generator.m_source_locations)


            d_map_estimate, m_all_measurement_loc, d_metrics = \
                simulate_surveying_map(map, num_measurements=70, min_service_power=min_service_power,
                                       grid=grid, route_planner=route_planner, sampler=sampler,
                                       estimator=estimators[2], num_measurements_to_plot=10,
                                       m_meta_data=m_building_data,
                                       num_frozen_uncertainty=20)

            # ld_metrics = [(route_planner, d_metrics)]
            # return map_generator.plot_metrics(ld_metrics)

        experiment_1a()

    """This experiment generates the Rosslyn dataset."""
    def experiment_3001(args):

        # np.random.seed(0)
        # tf.random.set_seed(1234)

        # 0. Grid
        grid = RectangularGrid(
            gridpoint_spacing=3,
            num_points_x=32,  # 20,  #12
            num_points_y=32,  # 18,  #10
            height=20)

        # 1. Map generator
        num_sources = 2
        shadowing_std = np.sqrt(10)
        l_file_num = np.arange(41, 43)
        shadowing_correlation_dist = 50  # 13.5  # values as in the autoencoder paper
        # source_power = 30 * np.ones(shape=(num_sources, ))
        source_power = np.tile([10, 10], (num_sources, 1))
        f_shadowing_covariance = CorrelatedShadowingGenerator.gudmundson_correlation(
            shadowing_std=shadowing_std,  # 1
            shadowing_correlation_dist=shadowing_correlation_dist  # 40, #15
        )
        map_generator = InsiteMapGenerator(grid=grid,
                                           l_file_num=l_file_num,
                                           inter_grid_points_dist_factor=1)

        # 2. Sampling
        sampler = InterpolationSampler(
            grid,
            interpolation_method="in_grid",
        )

        def generate_dataset(num_maps_train=100,
                             num_maps_test=25,
                             num_measurements_per_map=(1, 100),
                             num_blocks_per_map=50):
            md = GridMeasurementDataset.generate(
                grid=grid,
                map_generator=map_generator,
                # sampler=sampler,
                num_measurements_per_map=num_measurements_per_map,
                num_maps_train=num_maps_train,
                num_maps_test=num_maps_test,
                num_blocks_per_map=num_blocks_per_map)

            filename = (
                    datafolder + f"grid_{grid.num_points_y}_x_{grid.num_points_x}"
                                 f"-Wireless_Insite-{num_sources}_combined_sources"
                                 f"-files_between_{l_file_num[0]}_{l_file_num[-1]}"
                                 f"-{num_maps_train}_training_maps-{num_measurements_per_map}_measurements"
                                 f"-{num_blocks_per_map}_blocks_per_map.dill")

            md.save(filename)

        generate_dataset(num_maps_train=100,
                         num_maps_test=10,
                         num_measurements_per_map=100,
                         num_blocks_per_map=1)
        # generate_dataset(num_maps_train=1,
        #                  num_maps_test=100,
        #                  num_measurements_per_map=(20, 30),
        #                  num_blocks_per_map=1)
        # generate_dataset(num_maps_train=50000,
        #                  num_maps_test=10000,
        #                  num_measurements_per_map=(50, 80),
        #                  num_blocks_per_map=20)
        # generate_dataset(num_maps_train=50000,
        #                  num_maps_test=10000,
        #                  num_measurements_per_map=(50, 100),
        #                  num_blocks_per_map=20)
        #
        # generate_dataset(num_maps_train=50000,
        #                  num_maps_test=10000,
        #                  num_measurements_per_map=(20, 30),
        #                  num_blocks_per_map=40)
        # generate_dataset(num_maps_train=50000,
        #                  num_maps_test=10000,
        #                  num_measurements_per_map=(10, 20),
        #                  num_blocks_per_map=50)
        # generate_dataset(num_maps_train=50000,
        #                  num_maps_test=10000,
        #                  num_measurements_per_map=(1, 10),
        #                  num_blocks_per_map=50)

        # generate_dataset(num_maps_train=100000,
        #                  num_maps_test=20000,
        #                  num_measurements_per_map=200,#(1, 20),
        #                  num_blocks_per_map=5)
        #
        # generate_dataset(num_maps_train=50000,
        #                  num_maps_test=10000,
        #                  num_measurements_per_map=100, #(1, 100),
        #                  num_blocks_per_map=10)
        # generate_dataset(num_maps_train=50000,
        #                  num_maps_test=10000,
        #                  num_measurements_per_map=50,#(1, 20),
        #                  num_blocks_per_map=20)
        #
        # generate_dataset(num_maps_train=50000,
        #                  num_maps_test=10000,
        #                  num_measurements_per_map=20, #(1, 100),
        #                  num_blocks_per_map=40)

        return

    """ 
       MinimumCostPlanner experiment for the 
    Rosslyn dataset to plot the map 
    estimate and the uncertainty metric along 
    with the trajectory.
       """
    def experiment_3004(args):
        # 0. Grid
        grid = RectangularGrid(
            gridpoint_spacing=3,
            num_points_x=32,  # 20,  #12
            num_points_y=32,  # 18,  #10
            height=20)

        # 1. Map generator
        num_sources = 2
        shadowing_std = np.sqrt(10)
        shadowing_correlation_dist = 50 # 13.5  # values as in the autoencoder paper
        # source_power = 30 * np.ones(shape=(num_sources, ))

        source_height = 20
        m_source_locations = grid.random_points_in_the_area(num_sources)
        m_source_locations[:, 2] = source_height
        v_source_power = 10 * np.ones(shape=(num_sources,))

        source_power = np.tile([10, 10], (num_sources, 1))
        f_shadowing_covariance = CorrelatedShadowingGenerator.gudmundson_correlation(
            shadowing_std=shadowing_std,  # 1
            shadowing_correlation_dist=shadowing_correlation_dist  # 40, #15
        )

        map_generator = InsiteMapGenerator(grid=grid,
                                           l_file_num=[41, 42],
                                           inter_grid_points_dist_factor=1,
                                           filter_map=True)

        # 2. Sampling
        # initial_location = grid.indices_to_point((0, 0))
        initial_location = grid.random_points_in_the_grid_outside_buildings()[0]
        dist_between_measurements = 5
        dist_between_waypoints = 50
        l_route_planners = [
            GridPlanner(grid=grid,
                        initial_location=initial_location,
                        dist_between_measurements=dist_between_measurements),

            SquareSpiralGridPlanner(grid=grid,
                                    initial_location=initial_location,
                                    dist_between_measurements=dist_between_measurements),
            RandomPlanner(grid=grid,
                          initial_location=initial_location,
                          dist_between_measurements=dist_between_measurements,
                          dist_between_waypoints=dist_between_waypoints),
            IndependentUniformPlanner(
                grid=grid,
                initial_location=initial_location,
                dist_between_measurements=dist_between_measurements),
            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               # metric="service_entropy",
                               initial_location=initial_location,
                               dist_between_measurements=dist_between_measurements),
            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               dist_between_measurements=dist_between_measurements,
                               num_measurement_to_update_destination=7),
            UniformRandomSamplePlanner(grid=grid,
                                       initial_location=initial_location,
                                       dist_between_measurements=dist_between_measurements)
        ]

        # 3. Sampler
        sampler = InterpolationSampler(grid,
                                       interpolation_method='in_grid')

        # 4. Estimator
        # model = FullyConvolutionalNeuralNetwork()
        # model.load_weights(weights_file_string)
        nn_arch_id = '18'
        min_service_power = -80
        model = ForkConvNn.get_fork_conv_nn_arch(nn_arch_id=nn_arch_id)
        nn_estimator = NeuralNetworkEstimator(grid=grid,
                                              f_shadowing_covariance=f_shadowing_covariance,
                                              min_service_power=min_service_power,
                                              nn_arch_id=nn_arch_id,
                                              estimator=model
                                              )

        weights_file_string = (weightfolder + "insite_nn_arch_18_[4, 3, 7]_epochs__samp_scale=0.95_out_same_True_[0.5, 0, 1]_alpha.ckpt")
        # weights_file_string = (weightfolder + "insite_nn_arch_18_[3]_epochs__samp_scale=0.93_out_same_False_[0.5]_alpha.ckpt")

        nn_estimator.estimator.load_weights(weights_file_string)

        estimators = [BatchKrigingEstimator(
            grid,
            min_service_power=min_service_power,
            f_shadowing_covariance=f_shadowing_covariance,
            f_mean=None),  # map_generator.mean_at_channel_and_location),

            nn_estimator]

        # Simulation

        def experiment_1a():
            """Plot maps after every measurement."""
            global b_do_not_plot_power_maps
            b_do_not_plot_power_maps = False  # True
            route_planner = l_route_planners[5]
            route_planner.debug_level = 0
            map, m_building_data = map_generator.generate_map()

            d_map_estimate, m_all_measurement_loc, d_metrics = \
                simulate_surveying_map(map, num_measurements=150, min_service_power=min_service_power,
                                       grid=grid, route_planner=route_planner, sampler=sampler,
                                       estimator=estimators[1], num_measurements_to_plot=5, m_meta_data=m_building_data)
        experiment_1a()

        # F = ExperimentSet.compare_estimators_montecarlo(num_mc_iterations=2,  # 40,
        #                                                 num_measurements=10,  # 300,
        #                                                 min_service_power=min_service_power,
        #                                                 map_generator=map_generator,
        #                                                 l_route_planners=l_route_planners[5],
        #                                                 sampler=sampler,
        #                                                 estimators=estimators,
        #                                                 grid=grid)
        # return F

    """ MinimumCostPlanner experiment for the 
    Rosslyn dataset to plot the map 
    estimate and the uncertainty metric along 
    with the trajectory.
           """
    def experiment_30041(args):
        # 0. Grid
        grid = RectangularGrid(
            gridpoint_spacing=3,
            num_points_x=32,  # 20,  #12
            num_points_y=32,  # 18,  #10
            height=20)

        # 1. Map generator
        num_sources = 2
        shadowing_std = np.sqrt(10)
        shadowing_correlation_dist = 50 # 13.5  # values as in the autoencoder paper
        # source_power = 30 * np.ones(shape=(num_sources, ))

        source_height = 20
        m_source_locations = grid.random_points_in_the_area(num_sources)
        m_source_locations[:, 2] = source_height
        v_source_power = 10 * np.ones(shape=(num_sources,))

        source_power = np.tile([10, 10], (num_sources, 1))
        f_shadowing_covariance = CorrelatedShadowingGenerator.gudmundson_correlation(
            shadowing_std=shadowing_std,  # 1
            shadowing_correlation_dist=shadowing_correlation_dist  # 40, #15
        )

        map_generator = InsiteMapGenerator(grid=grid,
                                           l_file_num=[50, 51],
                                           inter_grid_points_dist_factor=1,
                                           filter_map=True)

        # 2. Sampling
        initial_location = grid.indices_to_point((0, 0))
        # initial_location = grid.random_points_in_the_grid_outside_buildings()[0]
        dist_between_measurements = 5
        dist_between_waypoints = 50
        l_route_planners = [
            GridPlanner(grid=grid,
                        initial_location=initial_location,
                        dist_between_measurements=dist_between_measurements
                        ),

            SquareSpiralGridPlanner(grid=grid,
                                    initial_location=initial_location,
                                    dist_between_measurements=dist_between_measurements
                                    ),

            IndependentUniformPlanner(
                grid=grid,
                initial_location=initial_location,
                # dist_between_measurements=dist_between_measurements
            ),
            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               dist_between_measurements=dist_between_measurements,
                               num_measurement_to_update_destination=10),
            UniformRandomSamplePlanner(grid=grid,
                                       initial_location=initial_location,
                                       dist_between_measurements=dist_between_measurements)
        ]

        # 3. Sampler
        sampler = InterpolationSampler(grid,
                                       interpolation_method='in_grid')

        # 4. Estimator
        # model = FullyConvolutionalNeuralNetwork()
        # model.load_weights(weights_file_string)
        nn_arch_id = '20'
        min_service_power = -80
        model = SkipConvNnArch20()
        nn_estimator = NeuralNetworkEstimator(grid=grid,
                                              f_shadowing_covariance=f_shadowing_covariance,
                                              min_service_power=min_service_power,
                                              nn_arch_id=nn_arch_id,
                                              estimator=model
                                              )

        # weights_file_string = (weightfolder +
        #                        "insite_nn_arch_20_[4]_epochs__samp_scale=0.96_out_same_True_[1]_alpha.ckpt")
        weights_file_string = (weightfolder +
                               "insite_nn_arch_20_[2]_epochs__samp_scale=0.9523_out_same_True_[1]_alpha.ckpt")

        nn_estimator.estimator.load_weights(weights_file_string)

        estimators = [BatchKrigingEstimator(
            grid,
            min_service_power=min_service_power,
            f_shadowing_covariance=f_shadowing_covariance,
            f_mean=None),  # map_generator.mean_at_channel_and_location),

            nn_estimator]

        # Simulation

        def experiment_1a():
            """Plot maps after every measurement."""
            global b_do_not_plot_power_maps
            b_do_not_plot_power_maps = False  # True
            route_planner = l_route_planners[3]
            route_planner.debug_level = 0
            map, m_building_data = map_generator.generate_map()

            d_map_estimate, m_all_measurement_loc, d_metrics = \
                simulate_surveying_map(map, num_measurements=90, min_service_power=min_service_power,
                                       grid=grid, route_planner=route_planner, sampler=sampler,
                                       estimator=estimators[1], num_measurements_to_plot=2, m_meta_data=m_building_data,
                                       num_frozen_uncertainty=15)
        experiment_1a()


    """Monte Carlo simulation to plot the RMSE vs 
    num_measurements for all the estimators on the 
    Gudmundson dataset. The samples are collected 
    uniformly at random grid points using a 
    UniformRandomSample Planner."""
    def experiment_5001(args):

        # 0. Grid
        grid = RectangularGrid(
            gridpoint_spacing=3,
            num_points_x=32,  # 20,  #12
            num_points_y=32,  # 18,  #10
            height=20)

        # 1. Map generator
        num_sources = 2
        shadowing_std = np.sqrt(10)
        shadowing_correlation_dist = 50
        source_height = 20
        m_source_locations = grid.random_points_in_the_area(num_sources)
        m_source_locations[:, 2] = source_height
        v_source_power = 10 * np.ones(shape=(num_sources,))
        # `m_source_locations` is updated inside a monte carlo simulator
        source_power = np.tile([10, 10], (num_sources, 1))
        f_shadowing_covariance = CorrelatedShadowingGenerator.gudmundson_correlation(
            shadowing_std=shadowing_std,  # 1
            shadowing_correlation_dist=shadowing_correlation_dist  # 40, #15
        )
        map_generator = CorrelatedShadowingGenerator(
            grid=grid,
            source_height=20,
            source_power=source_power,
            # v_source_power=v_source_power,
            # m_source_locations=m_source_locations.T,
            frequency=2.4e9,
            f_shadowing_covariance=f_shadowing_covariance,
            combine_channels=True)

        # 2. Sampler
        sampler = InterpolationSampler(
            grid,
            # interpolation_method="avg_nearest",
            interpolation_method="in_grid",
        )

        # 3. Sampling
        initial_location = grid.indices_to_point((0, 0))
        dist_between_measurements = 30
        dist_between_waypoints = 50
        l_route_planners = [
            GridPlanner(grid=grid,
                        initial_location=initial_location,
                        dist_between_measurements=dist_between_measurements),

            SquareSpiralGridPlanner(grid=grid,
                                    initial_location=initial_location,
                                    dist_between_measurements=dist_between_measurements),
            RandomPlanner(grid=grid,
                          initial_location=initial_location,
                          dist_between_measurements=dist_between_measurements,
                          dist_between_waypoints=dist_between_waypoints),
            IndependentUniformPlanner(
                grid=grid,
                initial_location=initial_location,
                dist_between_measurements=dist_between_measurements),
            MinimumCostPlanner(grid=grid,
                               metric="service_entropy",
                               initial_location=initial_location,
                               dist_between_measurements=dist_between_measurements),
            UniformRandomSamplePlanner(grid=grid,
                                       initial_location=initial_location,
                                       dist_between_measurements=dist_between_measurements)
        ]

        # 3. Estimator
        nn_arch_id = '18'
        # nn_estimator = NeuralNetworkEstimator(grid=grid,
        #                             f_shadowing_covariance=f_shadowing_covariance,
        #                             min_service_power=5,
        #                             nn_arch_id=nn_arch_id)
        model = ForkConvNn.get_fork_conv_nn_arch(nn_arch_id=nn_arch_id)

        nn_estimator = NeuralNetworkEstimator(grid=grid,
                                           f_shadowing_covariance=f_shadowing_covariance,
                                           min_service_power=5,
                                           nn_arch_id=nn_arch_id,
                                           estimator=model)
        # weights_file_string = (weightfolder + "ckpt_nn_architecture_" + nn_arch_id + "-(1, 10)_meas")
        weights_file_string = (
                    weightfolder + "nn_arch_18_(1, 100)_meas_[5, 3, 9]_epochs__samp_scale=0.95_out_same_True_[0.5, 0, 1]_alpha.ckpt")
        nn_estimator.estimator.load_weights(weights_file_string)

        estimators = [BatchKrigingEstimator(
                                grid,
                                min_service_power=None,
                                f_shadowing_covariance=f_shadowing_covariance,
                                f_mean=map_generator.mean_at_channel_and_location),
            nn_estimator,

            KNNEstimator(grid=grid,
                         f_shadowing_covariance=f_shadowing_covariance,
                         num_neighbors=5
                         ),
            KernelRidgeRegressionEstimator(
                grid=grid,
                f_shadowing_covariance=f_shadowing_covariance,
                reg_par=1e-14),
            MultikernelEstimator(grid=grid,
                                 f_shadowing_covariance=f_shadowing_covariance,
                                 num_kernels=10,
                                 reg_par=1e-9)  ]

        # estimators =[nn_estimator]
        # estimators = [KNNEstimator(grid=grid,
        #                  f_shadowing_covariance=f_shadowing_covariance,
        #                  num_neighbors=5
        #                  ),]
        F = ExperimentSet.compare_estimators_montecarlo(num_mc_iterations=2,  # 40,
                                                        num_measurements=10,  # 300,
                                                        min_service_power=None,
                                                        map_generator=map_generator,
                                                        l_route_planners=l_route_planners[5],
                                                        sampler=sampler,
                                                        estimators=estimators,
                                                        grid=grid,
                                                        )
        return F

    """Monte Carlo simulation to plot the RMSE vs 
    num_measurements for all the estimators on the 
    Gudmundson dataset. The samples are collected 
    uniformly at random grid points using a 
    UniformRandomSample Planner."""
    def experiment_500111(args):

        # 0. Grid
        grid = RectangularGrid(
            gridpoint_spacing=3,
            num_points_x=32,  # 20,  #12
            num_points_y=32,  # 18,  #10
            height=20)

        # 1. Map generator
        num_sources = 2
        shadowing_std = np.sqrt(10)
        shadowing_correlation_dist = 50
        source_height = 20
        m_source_locations = grid.random_points_in_the_area(num_sources)
        m_source_locations[:, 2] = source_height
        v_source_power = 10 * np.ones(shape=(num_sources,))
        # `m_source_locations` is updated inside a monte carlo simulator
        source_power = np.tile([10, 10], (num_sources, 1))
        f_shadowing_covariance = CorrelatedShadowingGenerator.gudmundson_correlation(
            shadowing_std=shadowing_std,  # 1
            shadowing_correlation_dist=shadowing_correlation_dist  # 40, #15
        )
        map_generator = CorrelatedShadowingGenerator(
            grid=grid,
            source_height=20,
            source_power=source_power,
            # v_source_power=v_source_power,
            # m_source_locations=m_source_locations.T,
            frequency=2.4e9,
            f_shadowing_covariance=f_shadowing_covariance,
            combine_channels=True)

        # 2. Sampler
        sampler = InterpolationSampler(
            grid,
            # interpolation_method="avg_nearest",
            interpolation_method="in_grid",
        )

        # 3. Sampling
        initial_location = grid.indices_to_point((0, 0))
        dist_between_measurements = 5
        dist_between_waypoints = 50
        l_route_planners = [
            GridPlanner(grid=grid,
                        initial_location=initial_location,
                        # dist_between_measurements=dist_between_measurements
                        ),

            SquareSpiralGridPlanner(grid=grid,
                                    initial_location=initial_location,
                                    # dist_between_measurements=dist_between_measurements
                                    ),

            IndependentUniformPlanner(
                grid=grid,
                initial_location=initial_location,
                # dist_between_measurements=dist_between_measurements
            ),
            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               dist_between_measurements=dist_between_measurements,
                               num_measurement_to_update_destination=10),
            UniformRandomSamplePlanner(grid=grid,
                                       initial_location=initial_location,
                                       dist_between_measurements=dist_between_measurements)
        ]

        # 3. Estimator
        nn_arch_id = '20'
        # nn_estimator = NeuralNetworkEstimator(grid=grid,
        #                             f_shadowing_covariance=f_shadowing_covariance,
        #                             min_service_power=5,
        #                             nn_arch_id=nn_arch_id)
        model = SkipConvNnArch20()

        nn_estimator = NeuralNetworkEstimator(grid=grid,
                                              f_shadowing_covariance=f_shadowing_covariance,
                                              min_service_power=5,
                                              nn_arch_id=nn_arch_id,
                                              estimator=model)
        # weights_file_string = (weightfolder + "ckpt_nn_architecture_" + nn_arch_id + "-(1, 10)_meas")
        weights_file_string = (
                weightfolder + "(1, 10)_meas_nn_arch_20_alpha=1_epochs=4_samp_scale=0.99_out_same=True")
        nn_estimator.estimator.load_weights(weights_file_string)

        estimators = [BatchKrigingEstimator(
            grid,
            min_service_power=None,
            f_shadowing_covariance=f_shadowing_covariance,
            f_mean=map_generator.mean_at_channel_and_location),

            nn_estimator,

            KNNEstimator(grid=grid,
                         f_shadowing_covariance=f_shadowing_covariance,
                         num_neighbors=5
                         ),
            # KernelRidgeRegressionEstimator(
            #     grid=grid,
            #     kernel_width=12,
            #     f_shadowing_covariance=f_shadowing_covariance,
            #     reg_par=1e-14),
            MultikernelEstimator(
                grid=grid,
                kernel_type="gaussian",
                num_kernels=20,
                reg_par=1e-7,
                max_kernel_width=150
            )
        ]

        # estimators =[nn_estimator]
        # estimators = [KNNEstimator(grid=grid,
        #                  f_shadowing_covariance=f_shadowing_covariance,
        #                  num_neighbors=5
        #                  ),]
        F = ExperimentSet.compare_estimators_montecarlo(num_mc_iterations=1000,  # 40,
                                                        num_measurements=110,  # 300,
                                                        min_service_power=None,
                                                        map_generator=map_generator,
                                                        l_route_planners=l_route_planners[4],
                                                        sampler=sampler,
                                                        estimators=estimators,
                                                        grid=grid,
                                                        )
        return F


    """Monte Carlo simulation to plot the RMSE vs 
    num_measurements for all the estimators on the 
    Gudmundson dataset. The samples are collected 
    uniformly at random grid points using a 
    UniformRandomSample Planner."""
    def experiment_50011101(args):

        # 0. Grid
        grid = RectangularGrid(
            gridpoint_spacing=3,
            num_points_x=32,  # 20,  #12
            num_points_y=32,  # 18,  #10
            height=20)

        # 1. Map generator
        num_sources = 2
        shadowing_std = np.sqrt(10)
        shadowing_correlation_dist = 50
        source_height = 20
        m_source_locations = grid.random_points_in_the_area(num_sources)
        m_source_locations[:, 2] = source_height
        v_source_power = 10 * np.ones(shape=(num_sources,))
        # `m_source_locations` is updated inside a monte carlo simulator
        source_power = np.tile([10, 10], (num_sources, 1))
        f_shadowing_covariance = CorrelatedShadowingGenerator.gudmundson_correlation(
            shadowing_std=shadowing_std,  # 1
            shadowing_correlation_dist=shadowing_correlation_dist  # 40, #15
        )
        map_generator = CorrelatedShadowingGenerator(
            grid=grid,
            source_height=20,
            source_power=source_power,
            # v_source_power=v_source_power,
            # m_source_locations=m_source_locations.T,
            frequency=2.4e9,
            f_shadowing_covariance=f_shadowing_covariance,
            combine_channels=True)

        # 2. Sampler
        sampler = InterpolationSampler(
            grid,
            # interpolation_method="avg_nearest",
            interpolation_method="in_grid",
        )

        # 3. Sampling
        initial_location = grid.indices_to_point((0, 0))
        dist_between_measurements = 5
        dist_between_waypoints = 50
        l_route_planners = [
            GridPlanner(grid=grid,
                        initial_location=initial_location,
                        # dist_between_measurements=dist_between_measurements
                        ),

            SquareSpiralGridPlanner(grid=grid,
                                    initial_location=initial_location,
                                    # dist_between_measurements=dist_between_measurements
                                    ),

            IndependentUniformPlanner(
                grid=grid,
                initial_location=initial_location,
                # dist_between_measurements=dist_between_measurements
            ),
            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               dist_between_measurements=dist_between_measurements,
                               num_measurement_to_update_destination=10),
            UniformRandomSamplePlanner(grid=grid,
                                       initial_location=initial_location,
                                       dist_between_measurements=dist_between_measurements)
        ]

        # 3. Estimator
        nn_arch_id = '20'
        # nn_estimator = NeuralNetworkEstimator(grid=grid,
        #                             f_shadowing_covariance=f_shadowing_covariance,
        #                             min_service_power=5,
        #                             nn_arch_id=nn_arch_id)
        model = SkipConvNnArch20()

        nn_estimator = NeuralNetworkEstimator(grid=grid,
                                              f_shadowing_covariance=f_shadowing_covariance,
                                              min_service_power=5,
                                              nn_arch_id=nn_arch_id,
                                              estimator=model)
        # weights_file_string = (weightfolder + "ckpt_nn_architecture_" + nn_arch_id + "-(1, 10)_meas")
        weights_file_string = (
                weightfolder + "(1, 10)_meas_nn_arch_20_alpha=1_epochs=4_samp_scale=0.99_out_same=True")
        nn_estimator.estimator.load_weights(weights_file_string)

        estimators = [BatchKrigingEstimator(
            grid,
            min_service_power=None,
            f_shadowing_covariance=f_shadowing_covariance,
            f_mean=map_generator.mean_at_channel_and_location, name_on_figs = 'BatchKriging'
        ),
            OnlineKrigingEstimator(
                grid,
                min_service_power=None,
                f_shadowing_covariance=f_shadowing_covariance,
                f_mean=map_generator.mean_at_channel_and_location,
            ),

            nn_estimator,

            KNNEstimator(grid=grid,
                         f_shadowing_covariance=f_shadowing_covariance,
                         num_neighbors=5
                         ),
            # KernelRidgeRegressionEstimator(
            #     grid=grid,
            #     kernel_width=12,
            #     f_shadowing_covariance=f_shadowing_covariance,
            #     reg_par=1e-14),
            MultikernelEstimator(
                grid=grid,
                kernel_type="gaussian",
                num_kernels=20,
                reg_par=1e-7,
                max_kernel_width=150
            )
        ]

        # estimators =[nn_estimator]
        # estimators = [KNNEstimator(grid=grid,
        #                  f_shadowing_covariance=f_shadowing_covariance,
        #                  num_neighbors=5
        #                  ),]
        F = ExperimentSet.compare_estimators_montecarlo(num_mc_iterations=1000,  # 40,
                                                        num_measurements=100,  # 300,
                                                        min_service_power=None,
                                                        map_generator=map_generator,
                                                        l_route_planners=l_route_planners[4],
                                                        sampler=sampler,
                                                        estimators=estimators,
                                                        grid=grid,
                                                        )
        return F


    """Monte Carlo simulation to plot the RMSE vs 
    num_measurements for different route planners
    using the nerual network on the Gudmundson dataset."""
    def experiment_500112(args):

        # 0. Grid
        grid = RectangularGrid(
            gridpoint_spacing=3,
            num_points_x=32,  # 20,  #12
            num_points_y=32,  # 18,  #10
            height=20)

        # 1. Map generator
        num_sources = 2
        shadowing_std = np.sqrt(10)
        shadowing_correlation_dist = 50
        source_height = 20
        m_source_locations = grid.random_points_in_the_area(num_sources)
        m_source_locations[:, 2] = source_height
        v_source_power = 10 * np.ones(shape=(num_sources,))
        # `m_source_locations` is updated inside a monte carlo simulator
        source_power = np.tile([10, 10], (num_sources, 1))
        f_shadowing_covariance = CorrelatedShadowingGenerator.gudmundson_correlation(
            shadowing_std=shadowing_std,  # 1
            shadowing_correlation_dist=shadowing_correlation_dist  # 40, #15
        )
        map_generator = CorrelatedShadowingGenerator(
            grid=grid,
            source_height=20,
            source_power=source_power,
            # v_source_power=v_source_power,
            # m_source_locations=m_source_locations.T,
            frequency=2.4e9,
            f_shadowing_covariance=f_shadowing_covariance,
            combine_channels=True)

        # 2. Sampler
        sampler = InterpolationSampler(
            grid,
            # interpolation_method="avg_nearest",
            interpolation_method="in_grid",
        )

        # 3. Sampling
        initial_location = grid.indices_to_point((0, 0))
        dist_between_measurements = 5
        dist_between_waypoints = 50
        l_route_planners = [
            # GridPlanner(grid=grid,
            #                          initial_location=initial_location,
            #                          dist_between_measurements=dist_between_measurements
            #                          ),

            # SquareSpiralGridPlanner(grid=grid,
            #                                      initial_location=initial_location,
            #                                      dist_between_measurements=dist_between_measurements
            #                                      ),

            IndependentUniformPlanner(
                grid=grid,
                initial_location=initial_location,
                dist_between_measurements=dist_between_measurements
            ),
            # MinimumCostPlanner(grid=grid,
            #                                 metric="power_variance",
            #                                 initial_location=initial_location,
                                            # dist_between_measurements=dist_between_measurements,
                                            # num_measurement_to_update_destination=15),
            # UniformRandomSamplePlanner(grid=grid,
            #                            initial_location=initial_location,
            #                            dist_between_measurements=dist_between_measurements)
        ]

        # 3. Estimator
        nn_arch_id = '20'
        # nn_estimator = NeuralNetworkEstimator(grid=grid,
        #                             f_shadowing_covariance=f_shadowing_covariance,
        #                             min_service_power=5,
        #                             nn_arch_id=nn_arch_id)
        model = SkipConvNnArch20()

        nn_estimator = NeuralNetworkEstimator(grid=grid,
                                              f_shadowing_covariance=f_shadowing_covariance,
                                              min_service_power=5,
                                              nn_arch_id=nn_arch_id,
                                              estimator=model)
        # weights_file_string = (weightfolder + "ckpt_nn_architecture_" + nn_arch_id + "-(1, 10)_meas")
        weights_file_string = (
                weightfolder + "(1, 10)_meas_nn_arch_20_alpha=1_epochs=4_samp_scale=0.99_out_same=True")
        nn_estimator.estimator.load_weights(weights_file_string)

        estimators = [
            BatchKrigingEstimator(
            grid,
            min_service_power=None,
            f_shadowing_covariance=f_shadowing_covariance,
            f_mean=map_generator.mean_at_channel_and_location),

            nn_estimator,

            KNNEstimator(grid=grid,
                         f_shadowing_covariance=f_shadowing_covariance,
                         num_neighbors=5
                         ),
            KernelRidgeRegressionEstimator(
                grid=grid,
                f_shadowing_covariance=f_shadowing_covariance,
                reg_par=1e-14),
            MultikernelEstimator(grid=grid,
                                 f_shadowing_covariance=f_shadowing_covariance,
                                 num_kernels=10,
                                 reg_par=1e-9)]

        # estimators =[nn_estimator]
        # estimators = [KNNEstimator(grid=grid,
        #                  f_shadowing_covariance=f_shadowing_covariance,
        #                  num_neighbors=5
        #                  ),]
        F = ExperimentSet.compare_route_planners_montecarlo(num_mc_iterations=5,  # 40,
                                                        num_measurements=30,  # 300,
                                                        min_service_power=None,
                                                        map_generator=map_generator,
                                                        l_route_planners=l_route_planners,
                                                        sampler=sampler,
                                                        estimator=estimators[1],
                                                        grid=grid,
                                                        )
        return F

    """Uncertainty vs measurements number with 
    different destination_update values, for diff. 
    decreasing functions using the neural network 
    estimator for the Gudmundson dataset."""
    def experiment_500117(args):

        # 0. Grid
        grid = RectangularGrid(
            gridpoint_spacing=3,
            num_points_x=32,  # 20,  #12
            num_points_y=32,  # 18,  #10
            height=20)

        # 1. Map generator
        num_sources = 2
        shadowing_std = np.sqrt(10)
        shadowing_correlation_dist = 50
        source_height = 20
        m_source_locations = grid.random_points_in_the_area(num_sources)
        m_source_locations[:, 2] = source_height
        v_source_power = 10 * np.ones(shape=(num_sources,))
        # `m_source_locations` is updated inside a monte carlo simulator
        source_power = np.tile([10, 10], (num_sources, 1))
        f_shadowing_covariance = CorrelatedShadowingGenerator.gudmundson_correlation(
            shadowing_std=shadowing_std,  # 1
            shadowing_correlation_dist=shadowing_correlation_dist  # 40, #15
        )
        map_generator = CorrelatedShadowingGenerator(
            grid=grid,
            source_height=20,
            source_power=source_power,
            # v_source_power=v_source_power,
            # m_source_locations=m_source_locations.T,
            frequency=2.4e9,
            f_shadowing_covariance=f_shadowing_covariance,
            combine_channels=True)

        # 2. Sampler
        sampler = InterpolationSampler(
            grid,
            # interpolation_method="avg_nearest",
            interpolation_method="in_grid",
        )

        # 3. Sampling
        initial_location = grid.indices_to_point((0, 0))
        dist_between_measurements = 5
        dist_between_waypoints = 50
        l_route_planners = [

            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               # dist_between_measurements=dist_between_measurements,
                               cost_func="Shortest",
                               cost_factor=0,
                               num_measurement_to_update_destination=12
                               ),

            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               # dist_between_measurements=dist_between_measurements,
                               cost_func="Shortest",
                               cost_factor=0,
                               num_measurement_to_update_destination=2
                               ),

            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               # dist_between_measurements=dist_between_measurements,
                               num_measurement_to_update_destination=12,
                               cost_factor=1,
                               cost_func="Reciprocal",
                               smoothing_constant=0.5
                               ),
            # MinimumCostPlanner(grid=grid,
            #                                 metric="power_variance",
            #                                 initial_location=initial_location,
            #                                 # dist_between_measurements=dist_between_measurements,
            #                                 cost_factor=1,
            #                                 num_measurement_to_update_destination=5,
            #                                 cost_func="Reciprocal",
            #                                 smoothing_constant=0.5),
            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               # dist_between_measurements=dist_between_measurements,
                               cost_factor=1,
                               num_measurement_to_update_destination=2,
                               cost_func="Reciprocal",
                               smoothing_constant=0.5),
            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               # dist_between_measurements=dist_between_measurements,
                               num_measurement_to_update_destination=12,
                               cost_factor=0.5,
                               cost_func="Exponential"),
            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               # dist_between_measurements=dist_between_measurements,
                               num_measurement_to_update_destination=2,
                               cost_factor=0.5,
                               cost_func="Exponential"),
            # MinimumCostPlanner(grid=grid,
            #                                 metric="power_variance",
            #                                 initial_location=initial_location,
            #                                 dist_between_measurements=dist_between_measurements,
            #                                 # num_measurement_to_update_destination=10,
            #                                 cost_func="Threshold",
            #                                 smoothing_constant=0.5)
        ]

        # 3. Estimator
        nn_arch_id = '20'
        # nn_estimator = NeuralNetworkEstimator(grid=grid,
        #                             f_shadowing_covariance=f_shadowing_covariance,
        #                             min_service_power=5,
        #                             nn_arch_id=nn_arch_id)
        model = SkipConvNnArch20()

        nn_estimator = NeuralNetworkEstimator(grid=grid,
                                              f_shadowing_covariance=f_shadowing_covariance,
                                              min_service_power=-25,
                                              nn_arch_id=nn_arch_id,
                                              estimator=model)
        # weights_file_string = (weightfolder + "ckpt_nn_architecture_" + nn_arch_id + "-(1, 10)_meas")
        weights_file_string = (
                weightfolder + "(1, 10)_meas_nn_arch_20_alpha=1_epochs=4_samp_scale=0.99_out_same=True")
        nn_estimator.estimator.load_weights(weights_file_string)

        estimators = [
            BatchKrigingEstimator(
            grid,
            min_service_power=-25,
            f_shadowing_covariance=f_shadowing_covariance,
            f_mean=map_generator.mean_at_channel_and_location,
            # f_mean = None,
            ),

            nn_estimator,
            ]

        def compare_route_planners_montecarlo(num_mc_iterations, num_measurements,
                                              min_service_power, map_generator,
                                              l_route_planners, sampler,
                                              estimator, grid):
            "Monte Carlo"

            ld_metrics = []
            for route_planner in l_route_planners:
                d_metrics = simulate_surveying_montecarlo(num_mc_iterations=num_mc_iterations,
                                                          map_generator=map_generator,
                                                          route_planner=route_planner, estimator=estimator,
                                                          num_measurements=num_measurements,
                                                          min_service_power=min_service_power, sampler=sampler,
                                                          grid=grid)
                ld_metrics.append((f"{route_planner.cost_func}, "
                                   f"dist_meas="
                                   f"{route_planner.num_measurement_to_update_destination}",
                                   d_metrics))

            F = plot_metrics(ld_metrics)

            return F

        F = compare_route_planners_montecarlo(num_mc_iterations=500,  # 40,
                                                        num_measurements=90,  # 300,
                                                        min_service_power=None,
                                                        map_generator=map_generator,
                                                        l_route_planners=l_route_planners,
                                                        sampler=sampler,
                                                        estimator=estimators[1],
                                                        grid=grid,
                                                        )
        return F


    """Uncertainty vs measurements number 
    for a reciprocal decreasing functions 
    with diff. cost_factor using the neural 
    network estimator for the Gudmundson dataset."""
    def experiment_500121(args):

        # 0. Grid
        grid = RectangularGrid(
            gridpoint_spacing=3,
            num_points_x=32,  # 20,  #12
            num_points_y=32,  # 18,  #10
            height=20)

        # 1. Map generator
        num_sources = 2
        shadowing_std = np.sqrt(10)
        shadowing_correlation_dist = 50
        source_height = 20
        m_source_locations = grid.random_points_in_the_area(num_sources)
        m_source_locations[:, 2] = source_height
        v_source_power = 10 * np.ones(shape=(num_sources,))
        # `m_source_locations` is updated inside a monte carlo simulator
        source_power = np.tile([10, 10], (num_sources, 1))
        f_shadowing_covariance = CorrelatedShadowingGenerator.gudmundson_correlation(
            shadowing_std=shadowing_std,  # 1
            shadowing_correlation_dist=shadowing_correlation_dist  # 40, #15
        )
        map_generator = CorrelatedShadowingGenerator(
            grid=grid,
            source_height=20,
            source_power=source_power,
            # v_source_power=v_source_power,
            # m_source_locations=m_source_locations.T,
            frequency=2.4e9,
            f_shadowing_covariance=f_shadowing_covariance,
            combine_channels=True)

        # 2. Sampler
        sampler = InterpolationSampler(
            grid,
            # interpolation_method="avg_nearest",
            interpolation_method="in_grid",
        )

        # 3. Sampling
        initial_location = grid.indices_to_point((0, 0))
        dist_between_measurements = 5
        dest_update_num = 7
        dist_between_waypoints = 50
        l_route_planners = [

            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               dist_between_measurements=dist_between_measurements,
                               num_measurement_to_update_destination=dest_update_num,
                               cost_factor=1,
                               cost_func="Reciprocal",
                               smoothing_constant=0.5
                               ),

            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               dist_between_measurements=dist_between_measurements,
                               num_measurement_to_update_destination=dest_update_num,
                               cost_factor=0.75,
                               cost_func="Reciprocal",
                               smoothing_constant=0.5
                               ),
            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               dist_between_measurements=dist_between_measurements,
                               num_measurement_to_update_destination=dest_update_num,
                               cost_factor=0.5,
                               cost_func="Reciprocal",
                               smoothing_constant=0.5
                               ),
            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               dist_between_measurements=dist_between_measurements,
                               num_measurement_to_update_destination=dest_update_num,
                               cost_factor=0.25,
                               cost_func="Reciprocal",
                               smoothing_constant=0.5
                               ),
            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               dist_between_measurements=dist_between_measurements,
                               num_measurement_to_update_destination=dest_update_num,
                               cost_factor=0.0,
                               cost_func="Reciprocal",
                               smoothing_constant=0.5
                               ),
        ]

        # 3. Estimator
        nn_arch_id = '20'
        # nn_estimator = NeuralNetworkEstimator(grid=grid,
        #                             f_shadowing_covariance=f_shadowing_covariance,
        #                             min_service_power=5,
        #                             nn_arch_id=nn_arch_id)
        model = SkipConvNnArch20()

        nn_estimator = NeuralNetworkEstimator(grid=grid,
                                              f_shadowing_covariance=f_shadowing_covariance,
                                              min_service_power=-25,
                                              nn_arch_id=nn_arch_id,
                                              estimator=model)
        # weights_file_string = (weightfolder + "ckpt_nn_architecture_" + nn_arch_id + "-(1, 10)_meas")
        weights_file_string = (
                weightfolder + "(1, 10)_meas_nn_arch_20_alpha=1_epochs=4_samp_scale=0.99_out_same=True")
        nn_estimator.estimator.load_weights(weights_file_string)

        estimators = [
            BatchKrigingEstimator(
            grid,
            min_service_power=-25,
            f_shadowing_covariance=f_shadowing_covariance,
            f_mean=map_generator.mean_at_channel_and_location,
            # f_mean = None,
            ),

            nn_estimator,
            ]

        def compare_route_planners_montecarlo(num_mc_iterations, num_measurements,
                                              min_service_power, map_generator,
                                              l_route_planners, sampler,
                                              estimator, grid):
            "Monte Carlo"

            ld_metrics = []
            for route_planner in l_route_planners:
                d_metrics = simulate_surveying_montecarlo(num_mc_iterations=num_mc_iterations,
                                                          map_generator=map_generator,
                                                          route_planner=route_planner, estimator=estimator,
                                                          num_measurements=num_measurements,
                                                          min_service_power=min_service_power, sampler=sampler,
                                                          grid=grid)
                ld_metrics.append((f"{route_planner.cost_func}, "
                                  f"dest_upd="
                                  f"{route_planner.num_measurement_to_update_destination}, "
                                  f"cost_fac={route_planner.cost_factor}",
                                  d_metrics))

            F = plot_metrics(ld_metrics)

            return F

        F = compare_route_planners_montecarlo(num_mc_iterations=1000,  # 40,
                                                        num_measurements=90,  # 300,
                                                        min_service_power=None,
                                                        map_generator=map_generator,
                                                        l_route_planners=l_route_planners,
                                                        sampler=sampler,
                                                        estimator=estimators[1],
                                                        grid=grid,
                                                        )
        return F


    """Monte Carlo simulation to plot the RMSE vs 
    num_measurements for all the estimators on the 
    Rosslyn dataset (Wireless Insite). The samples 
    are collected uniformly at random grid points 
    using a UniformRandomSample Planner."""
    def experiment_500220(args):

        # 0. Grid
        grid = RectangularGrid(
            gridpoint_spacing=3,
            num_points_x=32,  # 20,  #12
            num_points_y=32,  # 18,  #10
            height=20)

        # 1. Map generator
        num_sources = 2
        shadowing_std = np.sqrt(10)
        shadowing_correlation_dist = 50  # 13.5  # values as in the autoencoder paper

        f_shadowing_covariance = CorrelatedShadowingGenerator.gudmundson_correlation(
            shadowing_std=shadowing_std,  # 1
            shadowing_correlation_dist=shadowing_correlation_dist  # 40, #15
        )

        map_generator = InsiteMapGenerator(grid=grid,
                                           l_file_num=[41, 42],
                                           inter_grid_points_dist_factor=1,
                                           filter_map=True)

        # 2. Sampling
        initial_location = grid.indices_to_point((0, 0))
        dist_between_measurements = 8
        dist_between_waypoints = 50
        l_route_planners = [
            GridPlanner(grid=grid,
                        initial_location=initial_location,
                        dist_between_measurements=dist_between_measurements
                        ),

            SquareSpiralGridPlanner(grid=grid,
                                    initial_location=initial_location,
                                    dist_between_measurements=5,
                                    ),

            IndependentUniformPlanner(
                grid=grid,
                initial_location=initial_location,
                dist_between_measurements=dist_between_measurements
            ),
            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               dist_between_measurements=dist_between_measurements,
                               num_measurement_to_update_destination=8),
            # UniformRandomSamplePlanner(grid=grid,
            #                            initial_location=initial_location,
                                       # dist_between_measurements=dist_between_measurements
            # ),
        ]

        # 3. Sampler
        sampler = InterpolationSampler(grid,
                                       interpolation_method='in_grid')

        # 4. Estimator
        nn_arch_id = '20'
        model = SkipConvNnArch20()
        nn_estimator = NeuralNetworkEstimator(grid=grid,
                                              f_shadowing_covariance=f_shadowing_covariance,
                                              # min_service_power=5,
                                              # nn_arch_id=nn_arch_id,
                                              estimator=model)


        weights_file_string = (weightfolder + "Insite_(1, 100)_meas_nn_arch_20_alpha=1_epochs=4_samp_scale=0.96_out_same=True")

        nn_estimator.estimator.load_weights(weights_file_string)

        estimators = [
            # BatchKrigingEstimator(
            # grid,
            # min_service_power=None,
            # f_shadowing_covariance=f_shadowing_covariance,
            # f_mean=None),

            nn_estimator,

            # KNNEstimator(grid=grid,
            #              f_shadowing_covariance=f_shadowing_covariance
            #              ),
            # # KernelRidgeRegressionEstimator(
            # #     grid=grid,
            # #     f_shadowing_covariance=f_shadowing_covariance
            # #     ),
            # MultikernelEstimator(grid=grid,
            #                      f_shadowing_covariance=f_shadowing_covariance,
            #                      num_kernels=10,
            #                      reg_par=1e-15)
        ]
        # Simulation

        # def experiment_1a():
        #     """Plot maps after every measurement."""
        #     global b_do_not_plot_power_maps
        #     b_do_not_plot_power_maps = False  # True
        #     route_planner = l_route_planners[5]
        #     route_planner.debug_level = 0
        #     map, m_building_data = map_generator.generate_map()
        #
        #     d_map_estimate, m_all_measurement_loc, d_metrics = \
        #         simulate_surveying_map(map, num_measurements=250, min_service_power=min_service_power,
        #                                grid=grid, route_planner=route_planner, sampler=sampler,
        #                                estimator=estimators[1], num_measurements_to_plot=10, m_meta_data=m_building_data)
        # experiment_1a()

        F = ExperimentSet.compare_route_planners_montecarlo(num_mc_iterations=3000,  # 40,
                                                        num_measurements=90,  # 300,
                                                        min_service_power=None,
                                                        map_generator=map_generator,
                                                        l_route_planners=l_route_planners,
                                                        sampler=sampler,
                                                        estimator=estimators[0],
                                                        grid=grid)
        return F

    """Uncertainty vs measurements number for a 
    reciprocal decreasing function with diff. cost_factor 
    using the neural network estimator carried on the 
    Rosslyn dataset."""
    def experiment_5002310(args):

        # 0. Grid
        grid = RectangularGrid(
            gridpoint_spacing=3,
            num_points_x=32,  # 20,  #12
            num_points_y=32,  # 18,  #10
            height=20)

        # 1. Map generator
        num_sources = 2
        shadowing_std = np.sqrt(10)
        shadowing_correlation_dist = 50  # 13.5  # values as in the autoencoder paper

        f_shadowing_covariance = CorrelatedShadowingGenerator.gudmundson_correlation(
            shadowing_std=shadowing_std,  # 1
            shadowing_correlation_dist=shadowing_correlation_dist  # 40, #15
        )

        map_generator = InsiteMapGenerator(grid=grid,
                                           l_file_num=[41, 42],
                                           inter_grid_points_dist_factor=1,
                                           filter_map=True)

        # 2. Sampling
        initial_location = grid.indices_to_point((0, 0))
        dist_between_measurements = 4
        dest_update_num = 4
        dist_between_waypoints = 50
        l_route_planners = [

            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               dist_between_measurements=dist_between_measurements,
                               num_measurement_to_update_destination=dest_update_num,
                               cost_factor=1,
                               cost_func="Reciprocal",
                               smoothing_constant=0.5
                               ),

            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               dist_between_measurements=dist_between_measurements,
                               num_measurement_to_update_destination=dest_update_num,
                               cost_factor=0.9,
                               cost_func="Reciprocal",
                               smoothing_constant=0.5
                               ),
            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               dist_between_measurements=dist_between_measurements,
                               num_measurement_to_update_destination=dest_update_num,
                               cost_factor=0.8,
                               cost_func="Reciprocal",
                               smoothing_constant=0.5
                               ),
            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               dist_between_measurements=dist_between_measurements,
                               num_measurement_to_update_destination=dest_update_num,
                               cost_factor=0.7,
                               cost_func="Reciprocal",
                               smoothing_constant=0.5
                               ),
            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               dist_between_measurements=dist_between_measurements,
                               num_measurement_to_update_destination=dest_update_num,
                               cost_factor=0.6,
                               cost_func="Reciprocal",
                               smoothing_constant=0.5
                               ),
            MinimumCostPlanner(grid=grid,
                               metric="power_variance",
                               initial_location=initial_location,
                               dist_between_measurements=dist_between_measurements,
                               num_measurement_to_update_destination=dest_update_num,
                               cost_factor=0.0,
                               cost_func="Reciprocal",
                               smoothing_constant=0.5
                               ),

        ]

        # 3. Sampler
        sampler = InterpolationSampler(grid,
                                       interpolation_method='in_grid')

        # 4. Estimator
        nn_arch_id = '20'
        model = SkipConvNnArch20()
        nn_estimator = NeuralNetworkEstimator(grid=grid,
                                              f_shadowing_covariance=f_shadowing_covariance,
                                              # min_service_power=5,
                                              # nn_arch_id=nn_arch_id,
                                              estimator=model)

        weights_file_string = (
                    weightfolder + "Insite_(1, 100)_meas_nn_arch_20_alpha=1_epochs=4_samp_scale=0.96_out_same=True")

        nn_estimator.estimator.load_weights(weights_file_string)

        estimators = [
            # BatchKrigingEstimator(
            # grid,
            # min_service_power=None,
            # f_shadowing_covariance=f_shadowing_covariance,
            # f_mean=None),

            nn_estimator,

            # KNNEstimator(grid=grid,
            #              f_shadowing_covariance=f_shadowing_covariance
            #              ),
            # # KernelRidgeRegressionEstimator(
            # #     grid=grid,
            # #     f_shadowing_covariance=f_shadowing_covariance
            # #     ),
            # MultikernelEstimator(grid=grid,
            #                      f_shadowing_covariance=f_shadowing_covariance,
            #                      num_kernels=10,
            #                      reg_par=1e-15)
        ]
        # Simulation

        def compare_route_planners_montecarlo(num_mc_iterations, num_measurements,
                                              min_service_power, map_generator,
                                              l_route_planners, sampler,
                                              estimator, grid):
            "Monte Carlo"

            ld_metrics = []
            for route_planner in l_route_planners:
                d_metrics = simulate_surveying_montecarlo(num_mc_iterations=num_mc_iterations,
                                                          map_generator=map_generator,
                                                          route_planner=route_planner, estimator=estimator,
                                                          num_measurements=num_measurements,
                                                          min_service_power=min_service_power, sampler=sampler,
                                                          grid=grid)
                ld_metrics.append((f"{route_planner.cost_func}, "
                                  f"dest_upd="
                                  f"{route_planner.num_measurement_to_update_destination}, "
                                  f"cost_fac={route_planner.cost_factor}",
                                  d_metrics))

            F = plot_metrics(ld_metrics)

            return F
        F = compare_route_planners_montecarlo(num_mc_iterations=2000,  # 40,
                                                            num_measurements=90,  # 300,
                                                            min_service_power=None,
                                                            map_generator=map_generator,
                                                            l_route_planners=l_route_planners,
                                                            sampler=sampler,
                                                            estimator=estimators[0],
                                                            grid=grid)
        return F


    """AN experiment to plot the  histogram of 
    normalized power at unobserved locations
    for the Gudmundson Dataset where the map estimate 
    and the uncertainty metric is obtained via
    Neural network estimator."""
    def experiment_5004(args):

        # 0. Grid
        grid = RectangularGrid(
            gridpoint_spacing=3,
            num_points_x=32,  # 20,  #12
            num_points_y=32,  # 18,  #10
            height=20)

        # 1. Map generator
        num_sources = 2
        shadowing_std = np.sqrt(10)
        shadowing_correlation_dist = 50 # 13.5  # values as in the autoencoder paper
        # source_power = 30 * np.ones(shape=(num_sources, ))
        source_power = np.tile([10, 10], (num_sources, 1))
        f_shadowing_covariance = CorrelatedShadowingGenerator.gudmundson_correlation(
            shadowing_std=shadowing_std,  # 1
            shadowing_correlation_dist=shadowing_correlation_dist  # 40, #15
        )
        map_generator = CorrelatedShadowingGenerator(
            grid=grid,
            source_height=20,
            source_power=source_power,
            frequency=2.4e9,
            f_shadowing_covariance=f_shadowing_covariance,
            combine_channels=True)
        map, m_meta_data = map_generator.generate_map()

        # 2. Sampler
        sampler = InterpolationSampler(
            grid,
            # interpolation_method="avg_nearest",
            interpolation_method="in_grid",
        )

        # 3. Estimator
        nn_arch_id = '20'
        model = SkipConvNnArch20()
        estimator = NeuralNetworkEstimator(grid=grid,
                                    f_shadowing_covariance=f_shadowing_covariance,
                                    min_service_power=5,
                                    nn_arch_id=nn_arch_id,
                                    estimator=model)
        weights_file_string = (weightfolder + "nn_arch_20_(1, 10)_meas_[4, 4]_epochs__samp_scale=0.95_out_same_False_[0, 1]_alpha.ckpt")
        estimator.estimator.load_weights(weights_file_string)

        def obtain_normalized_histogram_at_unobserved_locations(num_measurements,
                                                              num_monte_carlo_for_histogram_plot):
            """This method returns a list of normalized power at unobserved locations"""
            # Histogram plot
            # num_measurements = 5
            l_estimated_hist = []
            # num_monte_carlo_for_histogram_plot = 20
            for num_monte in range(num_monte_carlo_for_histogram_plot):
                map, _ = map_generator.generate_map()

                # m_measurement_locations = grid.random_points_in_the_grid(num_points=num_measurements)
                # m_measurements = sampler.sample_map_multiple_loc(t_map=map,
                #                                                  m_sample_locations=m_measurement_locations

                total_grid_points = grid.num_points_y * grid.num_points_x
                indices_to_sampled_from = np.random.choice(np.arange(total_grid_points),
                                                           size=num_measurements,
                                                           replace=False)
                v_mask = np.zeros(shape=total_grid_points)
                # set the mask value to 1 at the sampled grid point indices
                v_mask[indices_to_sampled_from] = 1
                m_mask = v_mask.reshape((grid.num_points_y, grid.num_points_x))
                m_measurements = map * m_mask
                m_measurement_locations = m_mask

                estimator.reset()
                d_map_estimate = estimator.estimate(m_measurement_locations, m_measurements)

                # m_est_std = np.sqrt(d_map_estimate["t_power_map_norm_variance"] * f_shadowing_covariance(0))
                m_est_std = np.sqrt(d_map_estimate["t_power_map_norm_variance"])
                # normalize the predicted power at unobserved locations in the map ((True - est. mean)/est. std.).
                m_normalized_unobserved_power = np.where(m_measurement_locations ==0,
                                           (map - d_map_estimate["t_power_map_estimate"])/m_est_std,
                                           None)

                for row in m_normalized_unobserved_power[0]:
                    for row_entry in row:
                        if row_entry is not None:
                            l_estimated_hist.append(row_entry)
            return l_estimated_hist
        # len(l_estimated_hist)

        # n, bins, patches = plt.hist(np.asarray(l_estimated_hist), 100, density=True, facecolor='g', alpha=0.75)
        l_num_measurements = [50]
        for ind_l_num_measurements in l_num_measurements:
            l_estimated_hist = obtain_normalized_histogram_at_unobserved_locations(
                num_measurements=ind_l_num_measurements,
                num_monte_carlo_for_histogram_plot=20)
            sns.distplot(l_estimated_hist, hist=True, kde=False,)

        # standard gaussian parameters
        mean = 0
        standard_deviation = 1

        x_values = np.arange(-10, 10, 0.01)
        # y_values = scipy.stats.norm(mean, standard_deviation)
        # plt.plot(x_values, y_values.pdf(x_values))
        plt.legend([f"{l_num_measurements[0]} measurements.",
                    # f"{l_num_measurements[1]} measurements.",
                    # f"{l_num_measurements[2]} measurements.",
                    # "Std. Gaussian pdf"
                    ])
        plt.grid(True)
        plt.title(f"Histogram of normalized power at Unobs. locations")
        plt.savefig(f"./output/variational_autoencoder_experiments/"
                    f"experiment_5004.pdf")
        plt.show()
        # set_trace()

    """AN experiment to plot the  histogram of 
    normalized power at unobserved locations
    for the Rosslyn Dataset where the map estimate 
    and the uncertainty metric is obtained via
    Neural network estimator."""
    def experiment_5005(args):

        # 0. Grid
        grid = RectangularGrid(
            gridpoint_spacing=3,
            num_points_x=32,  # 20,  #12
            num_points_y=32,  # 18,  #10
            height=20)

        # 1. Map generator
        num_sources = 2
        shadowing_std = np.sqrt(10)
        shadowing_correlation_dist = 50  # 13.5  # values as in the autoencoder paper

        f_shadowing_covariance = CorrelatedShadowingGenerator.gudmundson_correlation(
            shadowing_std=shadowing_std,  # 1
            shadowing_correlation_dist=shadowing_correlation_dist  # 40, #15
        )

        map_generator = InsiteMapGenerator(grid=grid,
                                           l_file_num=[50, 51],
                                           inter_grid_points_dist_factor=1,
                                           filter_map=True)

        # 2. Sampler
        sampler = InterpolationSampler(
            grid,
            # interpolation_method="avg_nearest",
            interpolation_method="in_grid",
        )

        # 3. Estimator
        nn_arch_id = '20'
        model = SkipConvNnArch20()
        estimator = NeuralNetworkEstimator(grid=grid,
                                    f_shadowing_covariance=f_shadowing_covariance,
                                    min_service_power=5,
                                    nn_arch_id=nn_arch_id,
                                    estimator=model)
        weights_file_string = (weightfolder + "insite_nn_arch_20_[2]_epochs__samp_scale=0.9523_out_same_True_[1]_alpha.ckpt")

        weights_file_string = (weightfolder +
                               "insite_nn_arch_20_[11, 12]_epochs__samp_scale=0.85_out_same_False_[0, 1]_alpha.ckpt")
        estimator.estimator.load_weights(weights_file_string)


        def obtain_normalized_histogram_at_unobserved_locations(num_measurements,
                                                              num_monte_carlo_for_histogram_plot, est_mutiplier):
            """This method returns a list of normalized power at unobserved locations"""
            # Histogram plot
            # num_measurements = 5
            l_estimated_hist = []
            # num_monte_carlo_for_histogram_plot = 20

            for num_monte in range(num_monte_carlo_for_histogram_plot):
                map, m_building_meta_data = map_generator.generate_map()

                # m_measurement_locations = grid.random_points_in_the_grid(num_points=num_measurements)
                # m_measurements = sampler.sample_map_multiple_loc(t_map=map,
                #                                                  m_sample_locations=m_measurement_locations

                # total_grid_points = grid.num_points_y * grid.num_points_x
                # v_meta_data = m_building_meta_data.flatten()
                #
                # # indices of grid points that do not include building
                # relevant_ind = np.where(v_meta_data == 0)[0]
                # if num_measurements > len(relevant_ind):
                #     num_measurements = len(relevant_ind)
                #
                # indices_to_sampled_from = np.random.choice(relevant_ind,
                #                                            size=num_measurements,
                #                                            replace=False)
                # v_mask = np.zeros(shape=total_grid_points)
                # # set the mask value to 1 at the sampled grid point indices
                # v_mask[indices_to_sampled_from] = 1
                # m_mask = v_mask.reshape((grid.num_points_y, grid.num_points_x))
                # m_measurements = map * m_mask
                # m_measurement_locations = m_mask

                t_sampled_map, t_mask = sampler.sampled_map_multiple_grid_loc(t_true_map=map,
                                                                              num_measurements=num_measurements,
                                                                              m_building_metadata=m_building_meta_data)
                t_mask_with_building_metadata = t_mask - m_building_meta_data

                estimator.reset()
                d_map_estimate = estimator.estimate(measurement_locs=t_mask[0], measurements=t_sampled_map,
                                                    building_meta_data=m_building_meta_data)

                # m_mask_with_building_metadata = m_mask - m_building_meta_data
                m_mask_with_building_metadata = t_mask[0] - m_building_meta_data

                m_est_std = np.sqrt(d_map_estimate["t_power_map_norm_variance"] ) * est_mutiplier

                # normalize the predicted power at unobserved locations in the map ((True - est. mean)/est. std.).
                m_normalized_unobserved_power = np.where(m_mask_with_building_metadata == 0,
                                           (map - d_map_estimate["t_power_map_estimate"])/m_est_std,
                                           None)

                for row in m_normalized_unobserved_power[0]:
                    for row_entry in row:
                        if row_entry is not None:
                            l_estimated_hist.append(row_entry)
            return l_estimated_hist
        # len(l_estimated_hist)

        # n, bins, patches = plt.hist(np.asarray(l_estimated_hist), 100, density=True, facecolor='g', alpha=0.75)
        l_num_measurements = [30]
        est_mutiplier = [1]
        i=0
        for ind_l_num_measurements in l_num_measurements:

            l_estimated_hist = obtain_normalized_histogram_at_unobserved_locations(
                num_measurements=ind_l_num_measurements,
                num_monte_carlo_for_histogram_plot=20,
            est_mutiplier = est_mutiplier[i])
            i+=1
            sns.distplot(l_estimated_hist, hist=True, bins=100, kde=False, color='g')

        # standard gaussian parameters
        mean = 0
        standard_deviation = 1

        x_values = np.arange(-10, 10, 0.01)
        # y_values = scipy.stats.norm(mean, standard_deviation)
        # plt.plot(x_values, y_values.pdf(x_values))
        # plt.legend([f"{l_num_measurements[0]} measurements.",
                    # f"{l_num_measurements[1]} measurements.",
                    # f"{l_num_measurements[2]} measurements.",
                    # ])
        plt.grid(True)
        plt.title(f"Histogram of normalized power at Unobs. locations")
        plt.savefig(f"./output/variational_autoencoder_experiments/"
                    f"experiment_5005.pdf")
        plt.show()
        # set_trace()

    """Experiment to train the SkipConvNn with 
    Rosslyn data"""
    def experiment_9004(args):

        grid = RectangularGrid(
            gridpoint_spacing=3,
            num_points_x=32,  # 20,  #12
            num_points_y=32,  # 18,  #10
            height=20)

        # shadowing_std = np.sqrt(10)
        # shadowing_correlation_dist = 50

        def train_network(num_maps_train=None, num_blocks_per_map=20,
                          num_measurements_per_map=(50, 100),
                          l_file_num=np.arange(41, 43),
                          num_sources=2, loss_metric=None, b_posterior_target=False,
                          load_weights=False, l_alpha=[], l_epochs=[], l_learning_rate=[0.00001],
                          test_changes=False,
                          ):
            # filename = (
            #         datafolder + f"grid_{grid.num_points_y}_x_{grid.num_points_x}"
            #                      f"-shadowing_dist_{shadowing_correlation_dist}"
            #                      f"-gudmundson-{num_sources}_combined_sources"
            #                      f"-{num_maps_train}_training_maps-{num_measurements_per_map}_measurements"
            #                      f"-{num_blocks_per_map}_blocks_per_map"
            #                      f"-shadowing_std_{shadowing_std:.2f}-fixed_source_location.dill")
            filename = (
                    datafolder + f"grid_{grid.num_points_y}_x_{grid.num_points_x}"
                                 f"-Wireless_Insite-{num_sources}_combined_sources"
                                 f"-files_between_{l_file_num[0]}_{l_file_num[-1]}"
                                 f"-{num_maps_train}_training_maps-{num_measurements_per_map}_measurements"
                                 f"-{num_blocks_per_map}_blocks_per_map.dill")
            md = GridMeasurementDataset.load(filename)

            # Batch and shuffle the data set
            train_ds = md.get_data("train").shuffle(1000).batch(64)
            test_ds = md.get_data("test").batch(64)
            # train_ds = train_ds.take(15)
            # test_ds = test_ds.take(2)
            # # x_train = next(iter(train_ds.map(lambda x, y: x)))
            # x_test_batch = next(iter(test_ds.map(lambda x, y: x)))  # returns x_test out of test_ds
            # y_test_batch = next(iter(test_ds.map(lambda x, y: y)))  # returns y_test out of test_ds

            print("...........Data loading completed.......")
            # ---------- create a NN model ---------- #
            nn_arch_id = '20'
            sample_scaling = 0.7
            meas_output_same = False

            model = SkipConvNnArch20(sample_scaling=sample_scaling,
                                     meas_output_same=meas_output_same)

            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=weightfolder + f"nn_arch_id={nn_arch_id}_" + ".ckpt",
                save_weights_only=True,
                verbose=0,
                mode='min',
                save_best_only=True,
                # save_freq=10
            )
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/logs_9004", update_freq=1,
                                                                  histogram_freq=1,
                                                                  )
            dict_history = model.fit(train_dataset=train_ds, l_alpha=l_alpha, l_epochs=l_epochs,
                                     test_changes=test_changes, loss_metric='Custom',
                                     l_learning_rate=l_learning_rate, validation_dataset=test_ds,
                                     callbacks=cp_callback,
                                     nn_arch_id=nn_arch_id,
                                     save_weight_in_callback=True,
                                     tensorboard_callback=[],
                                     weightfolder=(weightfolder + f"Insite_{num_measurements_per_map}_meas_"),
                                     loss_folder=(lossfolder + f"Insite_{num_measurements_per_map}_meas_")
                                     )

            loss_file_string = (lossfolder + "insite_loss_nn_arch_" + nn_arch_id
                                + f"_{num_measurements_per_map}_meas"
                                  f"-{l_epochs}_epochs"
                                  f"-{l_alpha}_alpha.pkl"
                                  f"-{sample_scaling}_sam_scaling"
                                  f"-{meas_output_same}_out_same.pkl")

            model.save_weights(weightfolder + f"insite_nn_arch_{nn_arch_id}_{l_epochs}_epochs_"
                                              f"_samp_scale={sample_scaling}"
                                              f"_out_same_{meas_output_same}"
                                              f"_{l_alpha}_alpha.ckpt")

            outfile = open(loss_file_string, 'wb')
            pickle.dump(dict_history, outfile)
            outfile.close()

            # plot and save the figure.

            l_epoch_number = np.arange(len(dict_history['train_mean_rmse_loss']))
            G = GFigure(xaxis=l_epoch_number,
                        yaxis=dict_history[f"train_mean_rmse_loss"],
                        xlabel="Epochs",
                        ylabel="Loss",
                        title="Loss vs. Epochs",
                        legend="Train loss mean rmse",
                        styles="-D")
            G.add_curve(xaxis=l_epoch_number,
                        yaxis=dict_history[f"train_sigma_rmse_error"],
                        legend="Train loss sigma error",
                        styles="-o")
            G.add_curve(xaxis=l_epoch_number,
                        yaxis=dict_history[f"val_mean_rmse_loss"],
                        legend="Val loss mean ",
                        styles="-+")
            G.add_curve(xaxis=l_epoch_number,
                        yaxis=dict_history[f"val_sigma_rmse_error"],
                        legend="Val loss sigma error",
                        styles="-x")
            G.add_curve(xaxis=l_epoch_number,
                        yaxis=dict_history[f"val_loss"],
                        legend="Total val loss",
                        styles=":+")
            G.add_curve(xaxis=l_epoch_number,
                        yaxis=dict_history[f"train_loss"],
                        legend="Total train loss",
                        styles=":*")
            G.next_subplot(xlabel="Epochs", ylabel="Value of alpha")
            G.add_curve(xaxis=l_epoch_number,
                        yaxis=dict_history[f"alpha_vector"])
            return G

        G = train_network(num_maps_train=100, num_blocks_per_map=1,
                          num_measurements_per_map=100, l_epochs=[25,10,10],
                          loss_metric='Custom', l_alpha=[0.5,0,1],
                          l_learning_rate=[1e-6,1e-6,1e-6],
                          test_changes=False)
        return


    """Experiment to train SkipConvNN with 
    the Gudmundson data"""
    def experiment_9005(args):

        grid = RectangularGrid(
            gridpoint_spacing=3,
            num_points_x=32,  # 20,  #12
            num_points_y=32,  # 18,  #10
            height=20)

        shadowing_std = np.sqrt(10)
        shadowing_correlation_dist = 50

        def train_network(num_maps_train=None, num_blocks_per_map=20,
                          num_measurements_per_map=(50, 100),
                          num_sources=2, loss_metric=None, b_posterior_target=False,
                          load_weights=False, l_alpha=[], l_epochs=[], l_learning_rate=[0.00001],
                          test_changes=False,
                          ):
            filename = (
                    datafolder + f"grid_{grid.num_points_y}_x_{grid.num_points_x}"
                                 f"-shadowing_dist_{shadowing_correlation_dist}"
                                 f"-gudmundson-{num_sources}_combined_sources"
                                 f"-{num_maps_train}_training_maps-{num_measurements_per_map}_measurements"
                                 f"-{num_blocks_per_map}_blocks_per_map"
                                 f"-shadowing_std_{shadowing_std:.2f}.dill")
            md = GridMeasurementDataset.load(filename)

            # Batch and shuffle the data set
            train_ds = md.get_data("train").shuffle(1000).batch(64)
            test_ds = md.get_data("test").batch(64)

            print("...........Data loading completed.......")
            # ---------- create a NN model ---------- #
            nn_arch_id = '20'
            sample_scaling = 0.7
            meas_output_same = False

            model = SkipConvNnArch20(sample_scaling=sample_scaling,
                                     meas_output_same=meas_output_same)

            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=weightfolder + f"nn_arch_id={nn_arch_id}_" + ".ckpt",
                save_weights_only=True,
                verbose=0,
                mode='min',
                save_best_only=True,
                # save_freq=10
            )
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./log/logs_9005", update_freq=1,
                                                                  histogram_freq=1,
                                                                  )
            dict_history = model.fit(train_dataset=train_ds, l_alpha=l_alpha, l_epochs=l_epochs,
                                     test_changes=test_changes, loss_metric='Custom',
                                     l_learning_rate=l_learning_rate, validation_dataset=test_ds,
                                     callbacks=cp_callback,
                                     nn_arch_id=nn_arch_id,
                                     save_weight_in_callback=True,
                                     tensorboard_callback=tensorboard_callback,
                                     weightfolder=(weightfolder + f"{num_measurements_per_map}_meas_"),
                                     loss_folder=(lossfolder + f"{num_measurements_per_map}_meas_")
                                     )

            loss_file_string = (lossfolder + "loss_nn_arch_" + nn_arch_id
                                + f"_{num_measurements_per_map}_meas"
                                  f"-{l_epochs}_epochs"
                                  f"-{l_alpha}_alpha.pkl"
                                  f"-{sample_scaling}_sam_scaling"
                                  f"-{meas_output_same}_out_same.pkl")

            model.save_weights(weightfolder + f"nn_arch_{nn_arch_id}"
                                              f"_{num_measurements_per_map}_meas"
                                              f"_{l_epochs}_epochs_"
                                              f"_samp_scale={sample_scaling}"
                                              f"_out_same_{meas_output_same}"
                                              f"_{l_alpha}_alpha.ckpt")

            outfile = open(loss_file_string, 'wb')
            pickle.dump(dict_history, outfile)
            outfile.close()

            # plot and save the figure.

            l_epoch_number = np.arange(len(dict_history['train_mean_rmse_loss']))
            G = GFigure(xaxis=l_epoch_number,
                        yaxis=dict_history[f"train_mean_rmse_loss"],
                        xlabel="Epochs",
                        ylabel="Loss",
                        title="Loss vs. Epochs",
                        legend="Train loss mean rmse",
                        styles="-D")
            G.add_curve(xaxis=l_epoch_number,
                        yaxis=dict_history[f"train_sigma_rmse_error"],
                        legend="Train loss sigma error",
                        styles="-o")
            G.add_curve(xaxis=l_epoch_number,
                        yaxis=dict_history[f"val_mean_rmse_loss"],
                        legend="Val loss mean ",
                        styles="-+")
            G.add_curve(xaxis=l_epoch_number,
                        yaxis=dict_history[f"val_sigma_rmse_error"],
                        legend="Val loss sigma error",
                        styles="-x")
            G.add_curve(xaxis=l_epoch_number,
                        yaxis=dict_history[f"val_loss"],
                        legend="Total val loss",
                        styles=":+")
            G.add_curve(xaxis=l_epoch_number,
                        yaxis=dict_history[f"train_loss"],
                        legend="Total train loss",
                        styles=":*")
            G.next_subplot(xlabel="Epochs", ylabel="Value of alpha")
            G.add_curve(xaxis=l_epoch_number,
                        yaxis=dict_history[f"alpha_vector"])
            return G

        G = train_network(num_maps_train=20000, num_blocks_per_map=5,
                          num_measurements_per_map=100, l_epochs=[30, 30],
                          loss_metric='Custom', l_alpha=[0, 1],
                          l_learning_rate=[1e-5, 1e-5],
                          test_changes=False)
        return G


    """An experiment to plot the histogram of 
    the received true power value
    for the Remcom Dataset """
    def experiment_11004(args):

        # 0. Grid
        grid = RectangularGrid(
            gridpoint_spacing=3,
            num_points_x=32,  # 20,  #12
            num_points_y=32,  # 18,  #10
            height=20)

        # 1. Map generator
        num_sources = 2
        shadowing_std = np.sqrt(10)
        shadowing_correlation_dist = 50  # 13.5  # values as in the autoencoder paper

        f_shadowing_covariance = CorrelatedShadowingGenerator.gudmundson_correlation(
            shadowing_std=shadowing_std,  # 1
            shadowing_correlation_dist=shadowing_correlation_dist  # 40, #15
        )

        map_generator = InsiteMapGenerator(grid=grid,
                                           l_file_num=[41, 42],
                                           inter_grid_points_dist_factor=1,
                                           filter_map=True)

        # 2. Sampler
        sampler = InterpolationSampler(
            grid,
            # interpolation_method="avg_nearest",
            interpolation_method="in_grid",
        )

        # 3. Estimator
        nn_arch_id = '20'
        model = SkipConvNnArch20()
        estimator = NeuralNetworkEstimator(grid=grid,
                                    f_shadowing_covariance=f_shadowing_covariance,
                                    min_service_power=5,
                                    nn_arch_id=nn_arch_id,
                                    estimator=model)
        weights_file_string = (weightfolder + "insite_nn_arch_20_[4, 4]_epochs__samp_scale=0.99_out_same_False_[0, 1]_alpha.ckpt")
        estimator.estimator.load_weights(weights_file_string)

        def obtain_histogram_of_map(num_monte_carlo_for_histogram_plot):
            l_estimated_hist = []
            # num_monte_carlo_for_histogram_plot = 20
            for num_monte in range(num_monte_carlo_for_histogram_plot):
                map, m_building_meta_data = map_generator.generate_map()

               # power values outside buildings
                m_normalized_unobserved_power = np.where(m_building_meta_data == 0,
                                                         map,
                                                         None)

                for row in m_normalized_unobserved_power[0]:
                    for row_entry in row:
                        if row_entry is not None:
                            l_estimated_hist.append(row_entry)

            return l_estimated_hist

        l_estimated_hist = obtain_histogram_of_map(
            num_monte_carlo_for_histogram_plot=30)

        sns.distplot(l_estimated_hist, hist=True)

        plt.grid(True)
        plt.title(f"Histogram of true maps")
        plt.xlabel("Power [dBm]")
        plt.savefig(f"./output/variational_autoencoder_experiments/"
                    f"experiment_11005.pdf")
        plt.show()
        # set_trace()

    def plot_loss(dict_history):
        # filename_str = ('./train_data/' + filename)

        # infile = open(filename_str, 'rb')
        # dict_history = pickle.load(infile)

        l_epoch_number = np.arange(len(dict_history['train_mean_rmse_loss']))
        G = GFigure(xaxis=l_epoch_number,
                    yaxis=dict_history[f"train_mean_rmse_loss"],
                    xlabel="Epochs",
                    ylabel="Loss",
                    title="Loss vs. Epochs",
                    legend="Train loss mean mse",
                    styles="b-")
        G.add_curve(xaxis=l_epoch_number,
                    yaxis=dict_history[f"val_mean_rmse_loss"],
                    legend="Val loss mean ",
                    styles="b--")
        G.add_curve(xaxis=l_epoch_number,
                    yaxis=dict_history[f"train_sigma_rmse_error"],
                    legend="Train loss sigma error",
                    styles="g-")
        G.add_curve(xaxis=l_epoch_number,
                    yaxis=dict_history[f"val_sigma_rmse_error"],
                    legend="Val loss sigma error",
                    styles="g--")
        G.add_curve(xaxis=l_epoch_number,
                    yaxis=dict_history[f"train_loss"],
                    legend="Total train loss",
                    styles="r-")
        G.add_curve(xaxis=l_epoch_number,
                    yaxis=dict_history[f"val_loss"],
                    legend="Total val loss",
                    styles="r--")

        G.next_subplot(xlabel="Epochs", ylabel="Value of alpha")
        G.add_curve(xaxis=l_epoch_number,
                    yaxis=dict_history[f"alpha_vector"])
        return G

    def compare_route_planners_montecarlo(num_mc_iterations, num_measurements,
                                          min_service_power, map_generator,
                                          l_route_planners, sampler,
                                          estimator, grid):
        "Monte Carlo"

        ld_metrics = []
        for route_planner in l_route_planners:
            d_metrics = simulate_surveying_montecarlo(num_mc_iterations=num_mc_iterations, map_generator=map_generator,
                                                      route_planner=route_planner, estimator=estimator,
                                                      num_measurements=num_measurements,
                                                      min_service_power=min_service_power, sampler=sampler,
                                                      grid=grid)
            ld_metrics.append((route_planner.name_on_figs, d_metrics))

        F = map_generator.plot_metrics(ld_metrics)

        return F


    def compare_estimators_montecarlo(num_mc_iterations, num_measurements,
                                      min_service_power, map_generator,
                                      l_route_planners, sampler,
                                      estimators,grid):
        "Monte Carlo"

        ld_metrics = []
        for estimator in estimators:
            d_metrics = simulate_surveying_montecarlo(num_mc_iterations=num_mc_iterations, map_generator=map_generator,
                                                      route_planner=l_route_planners, estimator=estimator,
                                                      parallel_proc=False, num_measurements=num_measurements,
                                                      min_service_power=min_service_power, sampler=sampler, grid=grid,
                                                      )
            ld_metrics.append((estimator.name_on_figs, d_metrics))

        #F = map_generator.plot_metrics(ld_metrics)
        F = plot_metrics(ld_metrics)

        return F

def plot_map(t_map, l_m_measurement_loc,
             m_axes_row, vmin=None,
             vmax=None, str_title_base="",
             interp='nearest',
             ind_maps_in_row=1, grid=None,
             m_building_meta_data=None,
             destination_point=None):
    if m_building_meta_data is not None:
        t_map = np.where(m_building_meta_data == 1, np.nan, t_map)

    im = m_axes_row.imshow(
        t_map,
        interpolation=interp,
        cmap='jet',
        # origin='lower',
        extent=[grid.t_coordinates[0, -1, 0] - grid.gridpoint_spacing / 2,
                grid.t_coordinates[0, -1, -1] + grid.gridpoint_spacing / 2,
                grid.t_coordinates[1, -1, 0] - grid.gridpoint_spacing / 2,
                grid.t_coordinates[1, 0, 0] + grid.gridpoint_spacing / 2],
        vmax=vmax,
        vmin=vmin)

    for m_measurement_loc, str_legend in l_m_measurement_loc:
        m_axes_row.plot(m_measurement_loc[0, :],
                        m_measurement_loc[1, :],
                        # '+',
                        # color="red",
                        label=str_legend,
                        lw=4,
                        ls='-',
                        marker="+",
                        markersize=8,
                        visible=True
                        )
        # Last position
        m_axes_row.plot(m_measurement_loc[0, -1],
                        m_measurement_loc[1, -1],
                        's',
                        color="white")
        # l_ax.append(a_x)
    # m_axes_row.legend([l_ax[0][0],l_ax[1][0]], ["ac", "bc"])
    m_axes_row.legend()
    # destination point

    if destination_point is not None:
        m_axes_row.plot(destination_point[0],
                        destination_point[1],
                        's',
                        color="magenta")

    m_axes_row.set_xlabel('x [m]')
    if ind_maps_in_row == 0:
        m_axes_row.set_ylabel('y [m]')

    return im


