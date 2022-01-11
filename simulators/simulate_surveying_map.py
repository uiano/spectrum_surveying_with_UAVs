from utilities import empty_array, np
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
from utilities import watt_to_dbm, dbm_to_watt, natural_to_dB
from gsim.gfigure import GFigure
from IPython.core.debugger import set_trace
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
b_do_not_plot_power_maps = True

def simulate_surveying_map(map,
                           num_measurements=1,
                           min_service_power=None,
                           grid=None,
                           route_planner=None,
                           sampler=None,
                           estimator=None,
                           num_measurements_to_plot=None,
                           m_meta_data=None,
                           num_frozen_uncertainty=None,
                           ):

    assert m_meta_data is not None
    assert grid
    # Metric placeholders
    v_mse = empty_array((num_measurements,))
    v_norm_avg_power_variance = empty_array((num_measurements,))
    # metrics for loss computation
    v_mean_loss_in_sample = empty_array((num_measurements,))
    v_mean_loss_out_sample = empty_array((num_measurements,))
    v_sigma_loss_in_sample = empty_array((num_measurements,))
    v_sigma_loss_out_sample = empty_array((num_measurements,))

    if min_service_power:
        # Check external error in parameter setting
        assert min_service_power == estimator.min_service_power
        v_service_error_rate = empty_array((num_measurements,))
        v_avg_service_entropy = empty_array((num_measurements,))

    m_all_measurement_loc = np.zeros((3, 0))  # 3 x num_measurements_so_far
    num_sources = map.shape[0]
    m_all_measurements = np.zeros(shape=(num_sources, 0)) # num_sources x num_measurements so far
    d_map_estimate = None
    l_measurement_indices = [] # list of num_measurements-length indices of measurement locations
    # size of the map excluding the building locations
    map_size_without_building = len(np.where(m_meta_data == 0)[0])

    m_building_locations = grid.convert_grid_meta_data_to_standard_form(
        m_meta_data=m_meta_data)

    def evaluate_performance():

        # Performance evaluation
        v_mse[ind_measurement] = np.linalg.norm((1 - m_meta_data) *
                                                (map -
                                                 d_map_estimate["t_power_map_estimate"])) ** 2 \
                                 / map_size_without_building

        # For average normalized power variance at unobserved locations
        if d_map_estimate["t_power_map_norm_variance"] is not None:
            v_meas_indices = grid.nearest_gridpoint_inds(v_measurement_location)
            l_measurement_indices.append(v_meas_indices)

            # replace the estimated variance value with a nan at the building locations
            t_power_map_norm_variance = np.where(m_meta_data == 1,
                                                 np.nan,
                                                 d_map_estimate["t_power_map_norm_variance"])

            # replace the value with a nan at observed locations
            for v_meas_indices in l_measurement_indices:
                t_power_map_norm_variance[:, v_meas_indices[0], v_meas_indices[1]] = np.nan

            v_power_map_norm_variance = t_power_map_norm_variance.flatten()
            v_power_map_norm_variance = v_power_map_norm_variance[
                ~np.isnan(v_power_map_norm_variance)]

            # Average normalized power variance at unobserved locations
            v_norm_avg_power_variance[ind_measurement] = np.mean(v_power_map_norm_variance)

        # Prob. error
        if min_service_power:
            assert "t_service_map_estimate" in d_map_estimate.keys()
            v_service_error_rate[ind_measurement] = np.sum(
                ((d_map_estimate["t_service_map_estimate"] > 0.5) !=
                 (map > estimator.min_service_power)) * (1 - m_meta_data)) / map_size_without_building
            t_service_map_entropy = np.where(m_meta_data == 1,
                                             np.nan,
                                             d_map_estimate["t_service_map_entropy"])
            v_avg_service_entropy[ind_measurement] = np.mean(
                t_service_map_entropy[~np.isnan(t_service_map_entropy)])

    def eval_perf_at_in_out_sample_locations():
        _, t_mask = grid.convert_measurements_to_grid_form(m_all_measurements_loc=m_all_measurement_loc,
                                                           m_all_measurements=m_all_measurements)

        t_mask = t_mask - m_meta_data
        m_map_diff = map - d_map_estimate["t_power_map_estimate"]
        # calculate the mean loss for in sample and out sample locations
        v_mean_loss_in_sample[ind_measurement] = np.mean((m_map_diff[t_mask == 1])**2)

        v_mean_loss_out_sample[ind_measurement] = np.mean((m_map_diff[t_mask == 0])**2)

        m_std_deviation =  np.sqrt(d_map_estimate["t_power_map_norm_variance"])
        # calculate sigma loss for in sample and out sample locations
        v_sigma_loss_in_sample[ind_measurement] = np.mean((abs(m_map_diff[t_mask == 1])
                                                           - m_std_deviation[t_mask == 1])**2)
        v_sigma_loss_out_sample[ind_measurement] = np.mean((abs(m_map_diff[t_mask == 0])
                                                           - m_std_deviation[t_mask == 0])**2)
        if num_measurements_to_plot is not None:
            print(f"ind_measurement= {ind_measurement}, "
                  f"Mean rmse in sample loss: {np.sqrt(v_mean_loss_in_sample[ind_measurement])}, "
                  f"Mean rmse out sample : {np.sqrt(v_mean_loss_out_sample[ind_measurement])}, "
                  f"Sigma rmse in sample loss: {np.sqrt(v_sigma_loss_in_sample[ind_measurement])}, "
                  f"Sigma rmse out sample loss: {np.sqrt(v_sigma_loss_out_sample[ind_measurement])}")
    m_uncertainty_buffer = None
    for ind_measurement in range(num_measurements):
        if ind_measurement % 10 == 0:
            print(f"ind_measurement={ind_measurement}")

        # Measurement collection
        v_measurement_location = route_planner.next_measurement_location(
            d_map_estimate, m_meta_data)
        v_destination = route_planner.destination_point

        m_all_measurement_loc = np.hstack(
            (m_all_measurement_loc, np.reshape(v_measurement_location,
                                               (3, 1))))
        # 1D vector of length num_sources
        v_measurement = sampler.sample_map(map, v_measurement_location)
        m_all_measurements = np.hstack((m_all_measurements, np.reshape(v_measurement,
                                                                       (num_sources, 1))))

        # Estimation
        d_map_estimate = estimator.estimate(v_measurement_location[None, :],
                                            v_measurement[None, :],
                                            m_building_locations)
        # performance
        evaluate_performance()

        if d_map_estimate["t_power_map_norm_variance"] is not None:
            eval_perf_at_in_out_sample_locations()
        if num_frozen_uncertainty is not None:
            if ind_measurement == num_frozen_uncertainty:
                m_uncertainty_buffer = d_map_estimate["t_power_map_norm_variance"]
        if num_measurements_to_plot and (ind_measurement % num_measurements_to_plot == 0):
            compare_maps(map, d_map_estimate, grid=grid, min_service_power=min_service_power,
                         m_measurement_loc=m_all_measurement_loc, m_building_meta_data=m_meta_data,
                         destination_point=v_destination, m_uncertainty_buffer=m_uncertainty_buffer,
                         num_frozen_uncertainty=num_frozen_uncertainty)
    plt.show()

    d_metrics = {
        "v_mse": v_mse,
        # "v_rmse": np.sqrt(v_mse),
        "v_norm_avg_power_variance": v_norm_avg_power_variance
    }
    if min_service_power:
        d_metrics["v_service_error_rate"] = v_service_error_rate
        d_metrics["v_avg_service_entropy"] = v_avg_service_entropy

    if d_map_estimate["t_power_map_norm_variance"] is not None:
        d_metrics["v_mean_loss_in_sample"] = v_mean_loss_in_sample
        d_metrics["v_mean_loss_out_sample"] = v_mean_loss_out_sample
        d_metrics["v_sigma_loss_in_sample"] = v_sigma_loss_in_sample
        d_metrics["v_sigma_loss_out_sample"] = v_sigma_loss_out_sample

    return d_map_estimate, m_all_measurement_loc, d_metrics


def simulate_surveying_montecarlo(num_mc_iterations=None, map_generator=None, route_planner=None, estimator=None,
                                  grid=None, parallel_proc=False, **kwargs):
    """Returns a dictionary similar to the one returned by simulate_surveying_map.
    """
    assert num_mc_iterations
    assert map_generator
    assert estimator
    assert grid

    def simulate_one_run(ind_run):
        """ Note that some MapEstimators (e.g. the Kriging MapEstimators) use a function that provides
        the mean of the map at every point. This function uses the source locations. By looking at the
        present function, it may seem that if the map_generator generates the source locations at
        random, the estimator will not know the resulting source locations. However, the estimator may
        receive this information through estimator.f_mean, if the latter is set to a method of the object
        `map_generator`."""

        # source_height = 20
        # m_source_locations = grid.random_points_in_the_area(2)
        # m_source_locations[:, 2] = source_height
        # map_generator.m_source_locations = m_source_locations.T
        map, m_meta_data = map_generator.generate_map()
        num_building_locations = np.count_nonzero(m_meta_data)
        # Avoid maps that contains building location greater than 700
        while num_building_locations >= 700:
            map, m_meta_data = map_generator.generate_map()
            num_building_locations = np.count_nonzero(m_meta_data)

        estimator.reset()
        route_planner.reset()
        _, _, d_metrics = simulate_surveying_map(map, grid=grid, route_planner=route_planner,
                                                 estimator=estimator, m_meta_data=m_meta_data,
                                                  **kwargs)
        map_size_without_building = len(np.where(m_meta_data == 0)[0])
        v_map_norm_one_run = np.linalg.norm((1 - m_meta_data) * map) ** 2 / map_size_without_building
        return d_metrics, v_map_norm_one_run

    if parallel_proc:
        num_cores = int(multiprocessing.cpu_count() / 2)
        all_metrics_and_norms = Parallel(n_jobs=num_cores)(delayed(simulate_one_run)(i)
                                                           for i in range(num_mc_iterations))
        all_metrics_and_norms_ar = np.array(all_metrics_and_norms)
        ld_metrics = all_metrics_and_norms_ar[:, 0]
        v_map_norm = all_metrics_and_norms_ar[:, 1]
    else:
        ld_metrics = []
        v_map_norm = empty_array((num_mc_iterations,))  # Used for normalization
        for ind_mc_iteration in range(num_mc_iterations):
            d_metrics, v_map_norm_ind = simulate_one_run(ind_mc_iteration)
            ld_metrics.append(d_metrics)
            v_map_norm[ind_mc_iteration] = v_map_norm_ind

    # Average the metrics
    def avg_metric(str_metric):
        l_vals = [d_metrics[str_metric] for d_metrics in ld_metrics]
        m_vals = np.array(l_vals)
        return np.mean(m_vals, 0)

    d_metrics = {"v_mse": avg_metric("v_mse"),
                 "v_rmse": np.sqrt(avg_metric("v_mse")),
                 "v_nmse": avg_metric("v_mse") / np.mean(v_map_norm),
                 "v_norm_avg_power_variance": avg_metric("v_norm_avg_power_variance")}

    if "v_service_error_rate" in ld_metrics[0].keys():
        d_metrics["v_service_error_rate"] = avg_metric("v_service_error_rate")
        d_metrics["v_avg_service_entropy"] = avg_metric(
            "v_avg_service_entropy")

    if "v_mean_loss_in_sample" in ld_metrics[0].keys():
        d_metrics["v_mean_loss_in_sample"] = np.sqrt(avg_metric("v_mean_loss_in_sample"))
        d_metrics["v_mean_loss_out_sample"] = np.sqrt(avg_metric("v_mean_loss_out_sample"))
        d_metrics["v_sigma_loss_in_sample"] = np.sqrt(avg_metric("v_sigma_loss_in_sample"))
        d_metrics["v_sigma_loss_out_sample"] = np.sqrt(avg_metric("v_sigma_loss_out_sample"))

    return d_metrics


def compare_maps(t_true_map, d_map_estimate, grid, min_service_power=None, m_measurement_loc=None,
                 m_building_meta_data=None, t_posterior_std=None, name_on_figs=None,
                 t_map_diff = None, destination_point=None,
                 t_map_diff_minus_std=None, m_uncertainty_buffer=None, num_frozen_uncertainty=None):
    """- when `min_service_power` is not provided, this function plots a 3
    x num_sources subplot with the true and power estimates.
    - when `min_service_power` is provided, it plots an additional
    figure with 3 rows. First row for the true service map, second for the
    estimated service map, third for the uncertainty.
    """

    if min_service_power is not None:
        compare_power_map_journal(t_true_map,
                                               d_map_estimate,
                                               m_measurement_loc,
                                               min_service_power,
                                               grid,
                                               m_building_meta_data,
                                               destination_point=destination_point,
                                  m_uncertainty_buffer=m_uncertainty_buffer,
                                  num_frozen_uncertainty=num_frozen_uncertainty)
    else:
        compare_power_map(t_true_map, d_map_estimate,
                          m_measurement_loc, grid,
                          m_building_meta_data, t_posterior_std,
                          name_on_figs, t_map_diff,
                          t_map_diff_minus_std=t_map_diff_minus_std)


def compare_power_map(t_true_map, t_map_estimate,
                      m_measurement_loc, grid,
                      m_building_meta_data=None,
                      t_posterior_std=None, name_on_figs=None,
                      t_map_diff=None, t_map_diff_minus_std=None):


    if t_posterior_std is None:
        num_maps_in_row = 2
    else:
        num_maps_in_row = 3
    if len(t_map_estimate)==2:
        num_rows = 2
    else:
        num_rows = 1
    if t_map_diff is not None:
        num_maps_in_row += 1
    if t_map_diff_minus_std is not None:
        num_maps_in_row += 1
    fig, m_axis = plt.subplots(nrows=num_rows,
                               ncols=num_maps_in_row,
                               figsize=(16, 6))
    m_axes = np.reshape(m_axis, (num_rows, num_maps_in_row))
    m_true_map = watt_to_dbm(np.sum(dbm_to_watt(t_true_map), axis=0))


    if m_building_meta_data is not None:
        """set the values of grid point inside building as np.nan"""
        m_true_map_with_nan = np.where(m_building_meta_data == 1, np.nan, m_true_map)
        m_true_map_without_nan = m_true_map_with_nan[~np.isnan(m_true_map_with_nan)]
        vmax_power = m_true_map_without_nan.max()
        vmin_power = m_true_map_without_nan.min()
    else:
        vmax_power = m_true_map.max()
        vmin_power = m_true_map.min()


    im = plot_map(m_true_map, m_measurement_loc,
                  m_axes[0, 0], vmin_power, vmax_power,
                  "True Power", interp='bilinear',
                  ind_maps_in_row=0, grid=grid,
                  m_building_meta_data=m_building_meta_data)

    if num_rows == 2:
        t_map_estimate1 = t_map_estimate[0]
    else:
        t_map_estimate1 = t_map_estimate

    m_power_map_estimate = watt_to_dbm(
        np.sum(dbm_to_watt(t_map_estimate1), axis=0))

    plot_map(m_power_map_estimate, m_measurement_loc, m_axes[0, 1],
             vmin_power, vmax_power, "Posterior Mean Power",
             interp='bilinear', grid=grid,
             m_building_meta_data=m_building_meta_data)
    # power maps colorbar
    # plt.tight_layout()
    # fig.subplots_adjust(right=0.91)
    #
    cbar_ax = fig.add_axes([0.0001, 0.26, 0.01, 0.54])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Power [dBm]')

    if t_map_diff is not None:
        if m_building_meta_data is not None:
            m_map_diff_with_nan = np.where(m_building_meta_data == 1, np.nan, t_map_diff[0])
            m_map_diff_without_nan = m_map_diff_with_nan[~np.isnan(m_map_diff_with_nan)]
            vmax_diff = m_map_diff_without_nan.max()
            vmin_diff = m_map_diff_without_nan.min()
        else:
            vmax_diff = t_map_diff[0].max()
            vmin_diff = t_map_diff[0].min()

    if t_posterior_std is not None:
        if num_rows == 2:
            # t_posterior_std is 2 length list of posterior_std
            t_posterior_std1 = t_posterior_std[0]
        else:
            t_posterior_std1 = t_posterior_std
        if m_building_meta_data is not None:
            m_posterior_std_with_nan = np.where(m_building_meta_data == 1, np.nan, t_posterior_std[0])
            m_posterior_std_without_nan = m_posterior_std_with_nan[~np.isnan(m_posterior_std_with_nan)]
            vmax_std = m_posterior_std_without_nan.max()
            vmin_std = m_posterior_std_without_nan.min()
        else:
            vmax_std = t_posterior_std[0].max()
            vmin_std = t_posterior_std[0].min()
        # im_posterior_std_1 = plot_map(t_posterior_std1[0], m_measurement_loc, m_axes[0, 2],
        #                             vmin_std, vmax_std,
        #                             'Posterior Standard Deviation', interp='bilinear', grid=grid,
        #                             m_building_meta_data=m_building_meta_data)
        im_posterior_std_1 = plot_map(t_posterior_std1[0], m_measurement_loc, m_axes[0, 2],
                                      vmin_diff, vmax_diff,
                                      'Posterior Standard Deviation', interp='bilinear', grid=grid,
                                      m_building_meta_data=m_building_meta_data)

        # Posterior Variance color bar
        # cbar_bx = fig.add_axes([0.95, 0.26, 0.01, 0.54])
        # cbar = fig.colorbar(im_posterior_std_1, cax=cbar_bx)
        # cbar.set_label('Standard Deviation')
        # fig.subplots_adjust(left=0.12)

    if t_map_diff is not None:
        # if m_building_meta_data is not None:
        #     m_map_diff_with_nan = np.where(m_building_meta_data == 1, np.nan, t_map_diff[0])
        #     m_map_diff_without_nan = m_map_diff_with_nan[~np.isnan(m_map_diff_with_nan)]
        #     vmax_diff = m_map_diff_without_nan.max()
        #     vmin_diff = m_map_diff_without_nan.min()
        # else:
        #     vmax_diff = t_map_diff[0].max()
        #     vmin_diff = t_map_diff[0].min()
        im_map_diff = plot_map(t_map_diff[0], m_measurement_loc, m_axes[0, 3],
                                      vmin_diff, vmax_diff,
                                      '|Mean power difference|', interp='bilinear', grid=grid,
                                      m_building_meta_data=m_building_meta_data)
        # cbar_bx = fig.add_axes([0.95, 0.26, 0.01, 0.54])
        axins = inset_axes(m_axes[0, 3],
                           width="100%",
                           height="5%",
                           loc='lower center',
                           borderpad=-5
                           )
        cbar = fig.colorbar(im_map_diff, cax=axins, orientation="horizontal")
        # cbar = fig.colorbar(im_map_diff, shrink= 0.6, ax=m_axes[0, 2:4], location='bottom')
        cbar.set_label('Map Diff & Std')
        # fig.subplots_adjust(bottom=0.012)

    if t_map_diff_minus_std is not None:
        if m_building_meta_data is not None:
            m_map_diff_std_with_nan = np.where(m_building_meta_data == 1, np.nan, t_map_diff_minus_std[0])
            m_map_diff_std_without_nan = m_map_diff_std_with_nan[~np.isnan(m_map_diff_std_with_nan)]
            vmax_diff_std = m_map_diff_std_without_nan.max()
            vmin_diff_std = m_map_diff_std_without_nan.min()
        else:
            vmax_diff_std = t_map_diff[0].max()
            vmax_diff_std = t_map_diff[0].min()
        im_map_diff_minus_std = plot_map(t_map_diff_minus_std[0], m_measurement_loc, m_axes[0, 4],
                                      vmin_diff_std, vmax_diff_std,
                                      '|Mean pow. diff. minus std|.', interp='bilinear', grid=grid,
                                      m_building_meta_data=m_building_meta_data)

        cbar_bx = fig.add_axes([0.95, 0.26, 0.01, 0.54])
        cbar = fig.colorbar(im_map_diff_minus_std, cax=cbar_bx)
        cbar.set_label('Map diff -')
        fig.subplots_adjust(left=0.12)

    if num_rows ==2:
        plot_map(m_true_map, m_measurement_loc,
                 m_axes[1, 0], vmin_power, vmax_power,
                 "True Power", interp='bilinear',
                 ind_maps_in_row=0, grid=grid,
                 m_building_meta_data=m_building_meta_data)

        t_map_estimate = t_map_estimate[1]

        m_power_map_estimate = watt_to_dbm(
            np.sum(dbm_to_watt(t_map_estimate), axis=0))

        plot_map(m_power_map_estimate, m_measurement_loc, m_axes[1, 1],
                 vmin_power, vmax_power, "Posterior Mean Power",
                 interp='bilinear', grid=grid,
                 m_building_meta_data=m_building_meta_data)
        # power maps colorbar
        # plt.tight_layout()
        # fig.subplots_adjust(right=0.91)
        #
        # cbar_ax = fig.add_axes([0.0001, 0.26, 0.01, 0.54])
        # cbar = fig.colorbar(im, cax=cbar_ax)

        # cbar.set_label('Power [dBm]')

        if t_posterior_std is not None:
            t_posterior_std = t_posterior_std[1]
            # if m_building_meta_data is not None:
            #     m_posterior_std_with_nan = np.where(m_building_meta_data == 1, np.nan, t_posterior_std[0])
            #     m_posterior_std_without_nan = m_posterior_std_with_nan[~np.isnan(m_posterior_std_with_nan)]
            #     vmax_std = m_posterior_std_without_nan.max()
            #     vmin_std = m_posterior_std_without_nan.min()
            # else:
            #     vmax_std = t_posterior_std[0].max()
            #     vmin_std = t_posterior_std[0].min()
            im_posterior_std = plot_map(t_posterior_std[0], m_measurement_loc, m_axes[1, 2],
                                        vmin_std, vmax_std,
                                        'Posterior Standard Deviation', interp='bilinear', grid=grid,
                                        m_building_meta_data=m_building_meta_data)


    if name_on_figs is not None:
        fig.suptitle(name_on_figs)


def compare_power_map_journal(t_true_map, d_map_estimate,
                                   m_measurement_loc,
                                   min_service_power,
                                   grid,
                                   m_building_meta_data,
                                   destination_point=None,
                              m_uncertainty_buffer=None,
                              num_frozen_uncertainty=None):

    num_maps_in_row = 3
    num_rows = 1
    fig, m_axis = plt.subplots(nrows=num_rows,
                               ncols=num_maps_in_row,
                               figsize=(12, 4))
    m_axes = np.reshape(m_axis, (num_rows, num_maps_in_row))

    m_true_map = watt_to_dbm(np.sum(dbm_to_watt(t_true_map), axis=0))
    if m_building_meta_data is not None:
        """set the values of grid point inside building as np.nan"""
        m_true_map_with_nan = np.where(m_building_meta_data == 1, np.nan, m_true_map)
        m_true_map_without_nan = m_true_map_with_nan[~np.isnan(m_true_map_with_nan)]
        vmax_power = m_true_map_without_nan.max()
        vmin_power = m_true_map_without_nan.min()
    else:
        vmax_power = m_true_map.max()
        vmin_power = m_true_map.min()

    im = plot_map(m_true_map, m_measurement_loc,
                  m_axes[0, 0], vmin_power, vmax_power,
                  "True Power", interp='bilinear',
                  ind_maps_in_row=0, grid=grid, m_building_meta_data=m_building_meta_data,
                  num_frozen_uncertainty=num_frozen_uncertainty)

    t_power_map_estimate = d_map_estimate["t_power_map_estimate"]
    m_power_map_estimate = watt_to_dbm(
        np.sum(dbm_to_watt(t_power_map_estimate), axis=0))
    plot_map(m_power_map_estimate, m_measurement_loc, m_axes[0, 1],
             vmin_power, vmax_power, "Estimated Power",
             interp='bilinear', grid=grid, m_building_meta_data=m_building_meta_data,
             num_frozen_uncertainty=num_frozen_uncertainty)

    # power maps colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.91)

    cbar_ax = fig.add_axes([0.01, 0.26, 0.01, 0.54])
    cbar = fig.colorbar(im, cax=cbar_ax)

    cbar.set_label('Power [dBm]')

    #  service maps
    t_map_service = (t_true_map > min_service_power).astype(int)
    t_service_map_estimate = d_map_estimate["t_service_map_estimate"]

    if t_map_service.shape[0] == 2:
        m_map_service = np.logical_or(t_map_service[0, :, :],
                                      t_map_service[1, :, :])

        m_service_map_estimate = np.maximum(t_service_map_estimate[0, :, :],
                                            t_service_map_estimate[1, :, :])

    else:
        m_map_service = t_map_service[0]
        m_service_map_estimate = t_service_map_estimate[0]

    vmin_var = 0
    vmax_var = 5

    t_power_map_norm_variance = d_map_estimate["t_power_map_norm_variance"]
    m_power_map_norm_variance = np.mean(t_power_map_norm_variance, axis=0)
    if m_uncertainty_buffer is not None:
        m_power_map_norm_variance = np.sqrt(np.mean(m_uncertainty_buffer, axis=0))

    # t_service_map_entropy = d_map_estimate["t_service_map_entropy"]
    if m_measurement_loc.shape[1] == 1:
        vmax_var = m_power_map_norm_variance.max() * 2
        vmin_var = m_power_map_norm_variance.min() * 2

    im_std= plot_map(m_power_map_norm_variance, m_measurement_loc, m_axes[0, 2],
             vmin_var, vmax_var, "Uncertainty Metric ",
                         interp='bilinear', grid=grid, m_building_meta_data=m_building_meta_data,
                         destination_point=destination_point,
                     num_frozen_uncertainty=num_frozen_uncertainty)

    # Service and entropy color bar
    cbar_ax = fig.add_axes([0.95, 0.26, 0.01, 0.54])
    cbar = fig.colorbar(im_std, cax=cbar_ax)
    cbar.set_label('Uncertainty')
    plt.subplots_adjust(left=0.12)



def plot_map(t_map, m_measurement_loc,
             m_axes_row, vmin=None,
             vmax=None, str_title_base="",
             interp='nearest',
             ind_maps_in_row=1, grid=None,
             m_building_meta_data=None,
             destination_point=None,
             num_frozen_uncertainty=None):
    if m_building_meta_data is not None:
        t_map = np.where(m_building_meta_data == 1, np.nan, t_map)

    im = m_axes_row.imshow(
        t_map,
        interpolation=interp,
        cmap='jet',
        # origin='lower',
        extent=[grid.t_coordinates[0, -1, 0] - grid.gridpoint_spacing/2,
                grid.t_coordinates[0, -1, -1] + grid.gridpoint_spacing/2,
                grid.t_coordinates[1, -1, 0] - grid.gridpoint_spacing/2,
                grid.t_coordinates[1, 0, 0] + grid.gridpoint_spacing/2],
        vmax=vmax,
        vmin=vmin)
    if num_frozen_uncertainty is None:
        m_axes_row.plot(m_measurement_loc[0, :],
                        m_measurement_loc[1, :],
                        '+',
                        color="red")
    else:
        if len(m_measurement_loc[0])<=num_frozen_uncertainty:
            m_axes_row.plot(m_measurement_loc[0, :],
                            m_measurement_loc[1, :],
                            '+',
                            color="red")
        else:
            m_axes_row.plot(m_measurement_loc[0, 0:num_frozen_uncertainty+1],
                            m_measurement_loc[1, 0:num_frozen_uncertainty+1],
                            '+',
                            color="red")
            m_axes_row.plot(m_measurement_loc[0, num_frozen_uncertainty+1:],
                            m_measurement_loc[1, num_frozen_uncertainty+1:],
                            '+',
                            color="white")
    # Last position
    m_axes_row.plot(m_measurement_loc[0, -1],
                    m_measurement_loc[1, -1],
                    's',
                    color="white")
    # destination point
    if destination_point is not None:
        m_axes_row.plot(destination_point[0],
                        destination_point[1],
                        's',
                        color="magenta")

    m_axes_row.set_xlabel('x [m]')
    if ind_maps_in_row == 0:
        m_axes_row.set_ylabel('y [m]')

    m_axes_row.set_title(str_title_base)

    return im

def _plot_row(
              t_map,
              m_measurement_loc,
              m_axes_row,
              grid=None,
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
                grid.t_coordinates[0, -1, 0],
                grid.t_coordinates[0, -1, -1],
                grid.t_coordinates[1, -1,
                                        0], grid.t_coordinates[1, 0,
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
                  color="red")
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

def plot_metrics(l_metrics):
    """
    `l_metrics` is a list of tuples (legend_str, d_metrics)
    """

    # if "v_service_error_rate" in l_metrics[0][1].keys():
    #     num_cols = 2
    # else:
    #     num_cols = 1
    # make it more flexible after changing GFig to use rows as 1st dim

    G_power = GFigure(num_subplot_rows=1, figsize=(12, 7))
    if "v_service_error_rate" in l_metrics[0][1].keys():
        G_service = GFigure(num_subplot_rows=2, figsize=(12, 7))
    G_power_var = GFigure(num_subplot_rows=1, figsize=(12, 7))
    for str_legend, d_metrics in l_metrics:

        if "v_mean_loss_in_sample" in d_metrics.keys():
            # return the test loss of for the estimator that estimates post. std. deviation
            G_loss = GFigure(num_subplot_rows=1, figsize=(7, 5))
            G_loss.select_subplot(0, ylabel="Test Loss [dB]", xlabel="Number of measurements",
                                  )
            G_loss.add_curve(yaxis=d_metrics["v_mean_loss_in_sample"],
                             xaxis= np.arange(1, len(d_metrics["v_mean_loss_in_sample"])+1),
                              legend="Mean RMSE In Sample loss", styles='-r')
            G_loss.add_curve(yaxis=d_metrics["v_mean_loss_out_sample"],
                             xaxis=np.arange(1, len(d_metrics["v_mean_loss_in_sample"]) + 1),
                              legend="Mean RMSE Out Sample loss", styles ='--r')
            G_loss.add_curve(yaxis=d_metrics["v_sigma_loss_in_sample"],
                             xaxis=np.arange(1, len(d_metrics["v_mean_loss_in_sample"]) + 1),
                              legend="Sigma RMSE In Sample loss", styles='-b')
            G_loss.add_curve(yaxis=d_metrics["v_sigma_loss_out_sample"],
                             xaxis=np.arange(1, len(d_metrics["v_mean_loss_in_sample"]) + 1),
                              legend="Sigma RMSE Out Sample loss", styles='--b')

            # return G_loss

        if "v_rmse" in d_metrics.keys():
            G_power.select_subplot(0, ylabel="RMSE [dB]", xlabel='Number of measurements')
            G_power.add_curve(yaxis=d_metrics["v_rmse"],
                              legend=str_legend,
                              xaxis=np.arange(1, len(d_metrics["v_rmse"]) + 1),
                              )  # , legend=route_planner.name_on_figs)

        else:
            G_power.select_subplot(0, ylabel="MSE")
            G_power.add_curve(yaxis=d_metrics["v_mse"],
                              xaxis=np.arange(1, len(d_metrics["v_mse"]) + 1),
                              legend=str_legend)  # , legend=route_planner.name_on_figs)

        G_power_var.select_subplot(0,
                               xlabel='Number of measurements',
                               ylabel="Total unobserved power variance")

        G_power_var.add_curve(
            yaxis=d_metrics["v_norm_avg_power_variance"],
            xaxis=np.arange(1, len(d_metrics["v_norm_avg_power_variance"]) + 1),
            legend=str_legend,
            )  # , legend=route_planner.name_on_figs)

        if "v_service_error_rate" in d_metrics.keys():
            G_service.select_subplot(0, ylabel="Service error rate [%]")
            G_service.add_curve(yaxis=100 *
                                d_metrics["v_service_error_rate"],
                                xaxis=np.arange(1, len(d_metrics["v_service_error_rate"]) + 1),
                                )  # , legend=route_planner.name_on_figs)

            G_service.select_subplot(1,
                                     xlabel='Number of measurements',
                                     ylabel="Service Uncertainty")
            G_service.add_curve(yaxis=d_metrics["v_avg_service_entropy"],
                                xaxis=np.arange(1, len(d_metrics["v_avg_service_entropy"]) + 1),
                                legend=str_legend)

    if "v_service_error_rate" in d_metrics.keys():
        return [G_power, G_service, G_loss]
    else:
        # return [G_power, G_loss]
        return [G_power, G_loss, G_power_var]