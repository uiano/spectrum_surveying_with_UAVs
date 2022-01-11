from datetime import datetime
import abc
from IPython.core.debugger import set_trace
import numpy as np
from utilities import natural_to_dB, dbm_to_watt, watt_to_dbm
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from gsim.gfigure import GFigure
import logging
log = logging.getLogger("correlated_shadowing_generator")
log.setLevel(logging.DEBUG)
from utilities import natural_to_dB, empty_array
from map_generators.map_generator import MapGenerator

b_do_not_plot_power_maps = True


class CorrelatedShadowingGenerator(MapGenerator):

    def __init__(
            self,
            grid=None,  # Object of class RectangularGrid
            m_source_locations=None,  # 3 x num_sources matrix
            source_height=None,
            source_power=None,  # length-num_sources vector, in dBm
            frequency=None,
            # shadowing_std=0,
            # shadowing_correlation_dist=None,
            f_shadowing_covariance=None,
            antenna_gain_rec=0,
            antenna_gain_tr=0,
            v_source_power=None, # Deprecated
            **kwargs
    ):
        """Args:
        [brief explanation if needed]
        - `m_source_locations` is an 3 x num_sources array where the
          (0,j)-th entry is the x-coordinate of the j-th source, (1,j)-th
          entry is the y-coordinate of the j-th soure, etc. Alternatively, 
          one can specify `source_height` and the sources will be placed 
          uniformly at random in each realization.
        - `source_power` is an num_sources-length vector with the power of
          each source in dBm or an num_sources x 2 matrix whose (i,0)-th 
          entry is the min power and the (i,1)-th entry is the max power. 
          For each realization, the power is chosen uniformly at random 
          in this interval.
        - `f_shadowing_covariance`: function of distance. If None, no shadowing
          is generated.
        Returns:
        - `m_power_map` : num_grid_pts_per_dimension x
          num_grid_pts_per_dimension np.array whose (i,j)-th entry is the
          power at the (i,j)-th grid point.
        """
        assert grid is not None

        super().__init__(**kwargs)
        
        self.grid = grid

        if v_source_power is not None:
            source_power = v_source_power
            log.warning("Property `v_source_power` is deprecated."
                        " Please use `source_power` instead.")
        if m_source_locations is not None:
            assert len(source_power) == m_source_locations.shape[1]
            assert source_height is None
            self._mode = "deterministic" # the source location is deterministic
        else:
            assert source_height
            self._mode = "random" # the source location is randomly selected for each map.

        # if self.source_power.ndim == 2:
        #     assert self.source_power.shape[1] == 2
        if source_power.ndim == 2:
            assert source_power.shape[1] == 2
            self._source_power_interval = source_power
            self.source_power = empty_array((source_power.shape[0],))

        else:
            self._source_power_interval = None
            self.source_power = np.array(source_power)

        # self.m_source_locations = m_source_locations
        self._m_source_locations = m_source_locations
        self.source_height = source_height
        # self.shadowing_std = shadowing_std
        # self.shadowing_correlation_dist = shadowing_correlation_dist
        self.f_shadowing_covariance = f_shadowing_covariance


        emw_speed_c = 3e8
        self.f_propagation_gain = \
            lambda distance: CorrelatedShadowingGenerator.f_free_space_gain(
                distance,
                antenna_gain_tr=antenna_gain_tr,
                antenna_gain_rec=antenna_gain_rec,
                wavelength=emw_speed_c / frequency)

    @property
    def m_source_locations(self):
        return self._m_source_locations

    @m_source_locations.setter
    def m_source_locations(self, value):

        raise BufferError("Be careful before modifying source location.")

    def mean_at_channel_and_location(self, ind_channel, v_location):
        """
        Returns tx power plus propagation gain at point `v_location` and channel `ind_channel`

        """

        if self.m_source_locations is None:
            raise NotImplementedError("Before enabling this functionality, make sure "
                                          "that the existing code will not break")

        if self.source_power.ndim > 1:
            raise NotImplementedError(
                "The Tx. power is a random variable. "
                "Before allowing this ensure that existing code will not fail.")
        if self.combine_channels:
            "combine the mean at location over channel if the combined map is generated"
            total_mean_at_location_watt = 0
            for ind_source in range(len(self.source_power)):
                total_mean_at_location_watt += dbm_to_watt(
                    self.source_power[ind_source] + self.f_propagation_gain(np.linalg.norm(
                        v_location - np.expand_dims(self.m_source_locations[:, ind_source], 1), ord=2, axis=0)))
            return watt_to_dbm(total_mean_at_location_watt)
        else:
            return self.source_power[ind_channel] + \
                   self.f_propagation_gain(np.linalg.norm(
                       v_location - np.expand_dims(self.m_source_locations[:, ind_channel], 1), ord=2, axis=0))

    def _generate_map(self):
        """ 
        Used by MapGenerator.generate_map()

        returns a num_sources x grid.num_points_x x grid.num_points_y
        tensor, where the entry [i,j,k] is the power rx from source i
        at grid point (j,k).
        """

        num_sources = len(self.source_power)

        if self._mode == "deterministic":
            pass
            # m_source_locations = self._m_source_locations
        # if self.m_source_locations is not None:
        #     m_source_locations = self.m_source_locations
        elif self._mode == "random":
            assert self.source_height
            self._m_source_locations = self.grid.random_points_in_the_area(num_sources).T
            self._m_source_locations[2, :] = self.source_height

            # self._m_source_locations = m_source_locations
        else:
            raise NotImplementedError

        t_power_map = np.full(
            (num_sources, self.grid.num_points_y, self.grid.num_points_x),
            fill_value=None,
            dtype=float)

        for ind_source in range(num_sources):
            
            # if self.source_power.ndim < 2:
            #     power = self.source_power[ind_source]
            #
            # else:
            #     # Choose uniformly at random within the provided interval
            #     min_power = self.source_power[ind_source][0]
            #     max_power = self.source_power[ind_source][1]
            #     power = min_power + \
            #         (max_power-min_power)*np.random.random()
            if self._source_power_interval is None:
                power = self.source_power[ind_source]

            else:
                # Choose uniformly at random within the provided interval
                min_power = self._source_power_interval[ind_source][0]
                max_power = self._source_power_interval[ind_source][1]
                power = min_power + \
                        (max_power - min_power) * np.random.random()

                self.source_power[ind_source] = power

            m_base_power = self.base_power(
                source_power=power,
                v_source_location=self._m_source_locations[:, ind_source])
            t_power_map[
                ind_source, :, :] = m_base_power + self.fast_fading_realization(
                )

            if self.f_shadowing_covariance:
                t_power_map[ind_source, :, :] -= self.shadowing_realization()

        return t_power_map, np.zeros(shape=(self.grid.num_points_y, self.grid.num_points_x))

    def f_free_space_gain(distance, antenna_gain_tr, antenna_gain_rec,
                          wavelength):
        # return antenna_gain_tr + antenna_gain_rec - natural_to_dB(
        #     (distance**2) * wavelength**2)

        return antenna_gain_tr + antenna_gain_rec - natural_to_dB(
            distance ** 2) + natural_to_dB(wavelength ** 2)

    def base_power(self, source_power, v_source_location):
        """
        returns a matrix of the size of the grid with the tx + free space loss
        """

        return source_power + self.f_propagation_gain(
            self.grid.get_distance_to_grid_points(v_source_location))

    def gudmundson_correlation(shadowing_std, shadowing_correlation_dist):
        """
        - `shadowing_correlation_dist`: distance at which shadowing
          has correlation coefficient 1/2.
        - `shadowing_std` [dB]: if 0, no shadowing is generated
        """
        return lambda m_distance: (shadowing_std**2) * (0.5**(
            m_distance / shadowing_correlation_dist))

    def gaussian_correlation(stdev=None):
        return lambda m_distance: np.exp(-m_distance**2 / (2 * stdev**2))

    def shadowing_realization(self):

        # m_covariance = (self.shadowing_std**2) * (0.5**(
        #     self.grid.m_all_distances / self.shadowing_correlation_dist))

        m_covariance = self.f_shadowing_covariance(self.grid.m_all_distances)

        v_shadowing = np.random.multivariate_normal(mean=np.zeros(
            (self.grid.num_points_x * self.grid.num_points_y, )),
                                                    cov=m_covariance)

        return np.reshape(v_shadowing,
                          (self.grid.num_points_y, self.grid.num_points_x))

    def fast_fading_realization(self):

        return np.zeros((self.grid.num_points_y, self.grid.num_points_x))

    # Plotting maps --> move to a simulator class
    def compare_maps(
        self,
        t_true_map,  # True power map
        d_map_estimate,  # dictionary with the estimates
        min_service_power=None,
        # t_map_service=None,
        m_measurement_loc=None,
        # title="True Map vs Estimated Map",
        # vmax=None,
        # vmin=None,
        # fig=None,
        # l_axes=None
    ):
        """- when `min_service_power` is not provided, this function plots a 3
        x num_sources subplot with the true and power estimates.
        - when `min_service_power` is provided, it plots an additional
        figure with 3 rows. First row for the true service map, second for the
        estimated service map, third for the uncertainty.
        """
        if not b_do_not_plot_power_maps:
            self.compare_power_maps(t_true_map, d_map_estimate,
                                    m_measurement_loc)

        if min_service_power is not None:
            # self.compare_service_maps(t_true_map, d_map_estimate,
            #                           m_measurement_loc, min_service_power)
            self.compare_power_and_service_maps_mlsp(t_true_map,
                                                     d_map_estimate,
                                                     m_measurement_loc,
                                                     min_service_power)

    def compare_power_maps(self,
                           t_true_map,
                           d_map_estimate,
                           m_measurement_loc,
                           draw_lines=True):

        num_maps_in_row = t_true_map.shape[0]
        num_rows = 2
        if "t_power_map_norm_variance" in d_map_estimate.keys():
            num_rows += 1

        fig, m_axis = plt.subplots(nrows=num_rows,
                                   ncols=num_maps_in_row,
                                   figsize=(12, 7))
        m_axes = np.reshape(m_axis, (num_rows, num_maps_in_row))

        vmax_power = t_true_map[0, :, :].max()
        vmin_power = t_true_map[0, :, :].min()
        im = self._plot_row(t_true_map,
                            m_measurement_loc,
                            m_axes[0],
                            vmin_power,
                            vmax_power,
                            "True Power, Ch. ",
                            draw_lines=draw_lines)

        t_power_map_estimate = d_map_estimate["t_power_map_estimate"]
        self._plot_row(t_power_map_estimate,
                       m_measurement_loc,
                       m_axes[1],
                       vmin_power,
                       vmax_power,
                       "Estimated Power, Ch. ",
                       draw_lines=draw_lines)

        if "t_power_map_norm_variance" in d_map_estimate.keys():
            t_power_map_norm_variance = d_map_estimate[
                "t_power_map_norm_variance"]
            self._plot_row(t_power_map_norm_variance,
                           m_measurement_loc,
                           m_axes[2],
                           vmin=0,
                           vmax=1,
                           str_title_base="Uncertainty, Ch. ",
                           draw_lines=draw_lines)

        # Color bar
        plt.tight_layout()
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.35, 0.02, 0.5])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Power [dBm]')

    def compare_service_maps(self, t_true_map, d_map_estimate,
                             m_measurement_loc, min_service_power):

        num_maps_in_row = t_true_map.shape[0]
        num_rows = 3
        fig, m_axis = plt.subplots(nrows=num_rows,
                                   ncols=num_maps_in_row,
                                   figsize=(12, 7))
        m_axes = np.reshape(m_axis, (num_rows, num_maps_in_row))

        # True service map
        t_map_service = (t_true_map > min_service_power).astype(int)
        im_binary = self._plot_row(t_map_service, m_measurement_loc, m_axes[0],
                                   0, 1, "True Service, Ch. ")
        t_service_map_estimate = d_map_estimate["t_service_map_estimate"]
        self._plot_row(t_service_map_estimate > 0.5, m_measurement_loc,
                       m_axes[1], 0, 1, "Estimated Service, Ch. ")
        t_service_map_entropy = d_map_estimate["t_service_map_entropy"]
        self._plot_row(t_service_map_entropy, m_measurement_loc, m_axes[2], 0,
                       1, "Uncertainty, Ch. ")

        # Color bars
        plt.tight_layout()
        fig.subplots_adjust(right=0.9)
        # cbar_ax = fig.add_axes([0.92, 0.75, 0.02, 0.2])
        # cbar = fig.colorbar(im, cax=cbar_ax)
        # cbar.set_label('Power [dBm]')
        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.5])
        cbar = fig.colorbar(im_binary, cax=cbar_ax)
        # cbar.set_label('Service Availability')

    def compare_power_and_service_maps_mlsp(self, t_true_map, d_map_estimate,
                                            m_measurement_loc,
                                            min_service_power):

        num_maps_in_row = 5
        num_rows = 1
        fig, m_axis = plt.subplots(nrows=num_rows,
                                   ncols=num_maps_in_row,
                                   figsize=(15, 3))
        m_axes = np.reshape(m_axis, (num_rows, num_maps_in_row))

        m_true_map = watt_to_dbm(np.sum(dbm_to_watt(t_true_map), axis=0))
        vmax_power = m_true_map.max()
        vmin_power = m_true_map.min()

        im = self.plot_map(m_true_map,
                           m_measurement_loc,
                           m_axes[0, 0],
                           vmin_power,
                           vmax_power,
                           "True Power",
                           ind_maps_in_row=0,
                           interp='bilinear')

        t_power_map_estimate = d_map_estimate["t_power_map_estimate"]
        m_power_map_estimate = watt_to_dbm(
            np.sum(dbm_to_watt(t_power_map_estimate), axis=0))
        self.plot_map(m_power_map_estimate,
                      m_measurement_loc,
                      m_axes[0, 1],
                      vmin_power,
                      vmax_power,
                      "Estimated Power",
                      interp='bilinear')

        # power maps colorbar
        plt.tight_layout()
        fig.subplots_adjust(right=0.91)

        cbar_ax = fig.add_axes([0.01, 0.26, 0.01, 0.54])
        cbar = fig.colorbar(im, cax=cbar_ax)

        cbar.set_label('Power [dBm]')

        #  service maps
        t_map_service = (t_true_map > min_service_power).astype(int)

        # m_map_service = np.logical_or(t_map_service[0, :, :],
        #                               t_map_service[1, :, :])
        m_map_service = t_map_service[0]
        im_binary = self.plot_map(m_map_service, m_measurement_loc,
                                  m_axes[0, 2], 0, 1, "True Service")
        t_service_map_estimate = d_map_estimate["t_service_map_estimate"]

        # m_service_map_estimate = np.maximum(t_service_map_estimate[0, :, :],
        #                                     t_service_map_estimate[1, :, :])

        m_service_map_estimate = t_service_map_estimate[0]
        self.plot_map(m_service_map_estimate > 0.5, m_measurement_loc,
                      m_axes[0, 3], 0, 1, "Estimated Service")

        # if m_measurement_loc.shape[1] == 1:
        #     t_service_map_entropy_first = d_map_estimate["t_service_map_entropy"]
        #     self.first_m_service_map_entropy = np.mean(t_service_map_entropy_first, axis=0)
        #     self.first_m_measurement_loc = m_measurement_loc
        # self.plot_map(self.first_m_service_map_entropy, self.first_m_measurement_loc, m_axes[0, 4], 0,
        #               1, "Uncertainty ", interp='bilinear')

        t_service_map_entropy = d_map_estimate["t_service_map_entropy"]
        t_service_map_entropy = np.mean(t_service_map_entropy, axis=0)


        self.plot_map(t_service_map_entropy, m_measurement_loc, m_axes[0, 4], 0,
                      1, "Uncertainty ", interp='bilinear')

        # Service and entropy color bar
        cbar_ax = fig.add_axes([0.95, 0.26, 0.01, 0.54])
        cbar = fig.colorbar(im_binary, cax=cbar_ax)
        plt.subplots_adjust(left=0.12)

    def plot_map(self,
                 t_map,
                 m_measurement_loc,
                 m_axes_row,
                 vmin=None,
                 vmax=None,
                 str_title_base="",
                 interp='nearest',
                 ind_maps_in_row=1):

        im = m_axes_row.imshow(
            t_map,
            interpolation=interp,
            cmap='jet',
            # origin='lower',
            extent=[
                self.grid.t_coordinates[0, -1,
                                        0], self.grid.t_coordinates[0, -1, -1],
                self.grid.t_coordinates[1, -1,
                                        0], self.grid.t_coordinates[1, 0, 0]
            ],
            vmax=vmax,
            vmin=vmin)

        m_axes_row.plot(m_measurement_loc[0, :],
                        m_measurement_loc[1, :],
                        '+',
                        color="white")
        # Last position
        m_axes_row.plot(m_measurement_loc[0, -1],
                        m_measurement_loc[1, -1],
                        's',
                        color="white")
        m_axes_row.set_xlabel('x [m]')
        if ind_maps_in_row == 0:
            m_axes_row.set_ylabel('y [m]')

        m_axes_row.set_title(str_title_base)

        return im

    def _plot_row(self,
                  t_map,
                  m_measurement_loc,
                  m_axes_row,
                  vmin=None,
                  vmax=None,
                  str_title_base="",
                  draw_lines=True):

        for ind_maps_in_row in range(len(m_axes_row)):
            axis = m_axes_row[ind_maps_in_row]

            im = axis.imshow(
                t_map[ind_maps_in_row, :, :],
                # interpolation='spline16',
                cmap='jet',
                # origin='lower',
                extent=[
                    self.grid.t_coordinates[0, -1, 0],
                    self.grid.t_coordinates[0, -1, -1],
                    self.grid.t_coordinates[1, -1,
                                            0], self.grid.t_coordinates[1, 0,
                                                                        0]
                ],
                vmax=vmax,
                vmin=vmin)

            if draw_lines:
                style = '+-'
            else:
                style = '+'

            axis.plot(m_measurement_loc[0, :],
                      m_measurement_loc[1, :],
                      style,
                      color="white")
            if draw_lines:
                # Last position
                axis.plot(m_measurement_loc[0, -1],
                          m_measurement_loc[1, -1],
                          's',
                          color="white")
            axis.set_xlabel('x [m]')
            if ind_maps_in_row == 0:
                axis.set_ylabel('y [m]')

            axis.set_title(str_title_base + str(ind_maps_in_row))

        return im

    def plot_metrics(self, l_metrics):
        """
        `l_metrics` is a list of tuples (legend_str, d_metrics)
        """

        # if "v_service_error_rate" in l_metrics[0][1].keys():
        #     num_cols = 2
        # else:
        #     num_cols = 1
        # make it more flexible after changing GFig to use rows as 1st dim

        G_power = GFigure(num_subplot_rows=2, figsize=(12, 7))
        if "v_service_error_rate" in l_metrics[0][1].keys():
            G_service = GFigure(num_subplot_rows=2, figsize=(12, 7))

        for str_legend, d_metrics in l_metrics:

            if "v_nmse" in d_metrics.keys():
                G_power.select_subplot(0, ylabel="NMSE")
                G_power.add_curve(yaxis=d_metrics["v_nmse"]
                                  )  # , legend=route_planner.name_on_figs)

            else:
                G_power.select_subplot(0, ylabel="MSE")
                G_power.add_curve(yaxis=d_metrics["v_mse"]
                                  )  # , legend=route_planner.name_on_figs)

            G_power.select_subplot(1,
                                   xlabel='Number of measurements',
                                   ylabel="Total power variance")
            G_power.add_curve(
                yaxis=d_metrics["v_norm_avg_power_variance"],
                legend=str_legend)  # , legend=route_planner.name_on_figs)

            if "v_service_error_rate" in d_metrics.keys():
                G_service.select_subplot(0, ylabel="Service error rate [%]")
                G_service.add_curve(yaxis=100 *
                                    d_metrics["v_service_error_rate"]
                                    )  # , legend=route_planner.name_on_figs)

                G_service.select_subplot(1,
                                         xlabel='Number of measurements',
                                         ylabel="Service Uncertainty")
                G_service.add_curve(yaxis=d_metrics["v_avg_service_entropy"],
                                    legend=str_legend)

        if "v_service_error_rate" in d_metrics.keys():
            return [G_power, G_service]
        else:
            return G_power

    #
    # if metric == "power":
    #     plt.ylabel("NMSE")
    # elif metric == "service":
    #     plt.ylabel("Service error rate")

    @property
    def v_source_power(self):
        raise AttributeError("Property `v_source_power` is deprecated."
                    " Please use `source_power` instead.")

    @v_source_power.setter
    def v_source_power(self, val):
        raise AttributeError("Property `v_source_power` is deprecated."
                    " Please use `source_power` instead.")
    
