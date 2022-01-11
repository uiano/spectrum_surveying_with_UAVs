import numpy as np
from IPython.core.debugger import set_trace

import tensorflow as tf


class MapEstimationSimulator:
    def test_rmse(test_data,
                  estimator,
                  num_measurements,
                  num_iter=None,
                  shuffle=True):
        """Estimates RMSE using the test data in `test_data`. The format of
        `test_data` is the one of MeasurementDataset.data. For each
        entry in this Dataset, `num_measurements` measurements are
        selected at random if `shuffle==True`. Then, `estimator` is
        requested to estimate the map given these observed
        measurements at the locations of the measurements that were
        not observed. This operation is performed `num_iter` times,
        cycling over `test_data` if necessary. If `num_iter` is None,
        then each entry of `test_data` is used exactly once.

        """

        if isinstance(test_data, tf.data.Dataset):
            if num_iter is not None:
                test_data = test_data.repeat(num_iter)
            ds_it = iter(test_data)

        if num_iter is None:
            num_iter = len(test_data)

        mse = 0
        for ind_iter in range(num_iter):

            if isinstance(test_data, list):
                m_locations, m_measurements = test_data[ind_iter %
                                                        len(test_data)]
            elif isinstance(test_data, tf.data.Dataset):
                m_locations, m_measurements = next(ds_it)
                m_locations = m_locations.numpy()
                m_measurements = m_measurements.numpy()

            assert num_measurements < len(
                m_measurements), "Not enough measurements"

            if shuffle:
                v_permutation = np.random.permutation(len(m_locations))
                m_locations = m_locations[v_permutation]
                m_measurements = m_measurements[v_permutation]

            m_estimates = estimator.estimate_at_loc(
                m_locations[:num_measurements],
                m_measurements[:num_measurements],
                m_locations[num_measurements:])["power_est"]

            debug = False
            if debug:
                print(f"""estimator = {estimator.__class__},
                input loc = {m_locations[:num_measurements]},
                input meas = {m_measurements[:num_measurements]},
                test loc = {m_locations[num_measurements:]},
                estimate = {m_estimates}
                """)

            mse += np.sum((m_measurements[num_measurements:] - m_estimates)**
                          2) / m_estimates.size

        rmse = np.sqrt(mse / num_iter)

        return rmse
