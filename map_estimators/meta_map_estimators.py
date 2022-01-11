""" This module contains MapEstimators that regard measurements
collected at a given propagation scenario as a task.

A MetaMapEstimator will perform data adaptation to invoke a
MetaEstimator.

"""

import numpy as np
import dill
import copy
from datetime import datetime
from collections import OrderedDict
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace

import tensorflow as tf

from estimators.metaestimator import GradientMetaestimator, compare_metrics
#from estimators.gradient_metatrainable_estimators import \
#    MamlSubNetwork, CaviaSubNetwork
from estimators.metatrainable_estimators import CNPSubNetwork, FixedTaskLengthSubNetwork, SoftNNSubNetwork

#from ..map_generator import RectangularGrid
from .map_estimator import MapEstimator

import logging
log = logging.getLogger("metaestimators")
log.setLevel(logging.DEBUG)
logging.debug(f"Logging with level {log.root.level}")


def normalize_to_grid(grid, ds, min_dB=-200, max_dB=20):
    """Returns:

       `ds_out`: the normalized data set. The input coordinates and
       the targets lie in [-1, 1]. To this end, the following
       transformation is applied:

       f(x) = 2 x / (b - a)  -  (b + a)/(b - a)

       where a and b are respectively the lowest and highest endpoints
       of the interval where x is contained.

       The endpoints for spatial coordinates are obtained from grid. 
       The endpoints for the targets are [min_dB, max_dB]

       `unnorm_pars`: tuple (a, b) such that the unnormalized target
       corresponding to the target t is a * t + b.

    """
    def coefs(a, b):
        if b == a:
            return (0, 0)
        else:
            a = tf.convert_to_tensor(a, dtype=tf.double)
            b = tf.convert_to_tensor(b, dtype=tf.double)
            return (2 / (b - a), -(b + a) / (b - a))

    def normalize(inputs, targets):

        inputs = inputs * alpha_coord + beta_coord

        targets = alpha_targets * targets + beta_targets

        return inputs, targets

    alpha_x, beta_x = coefs(grid.min_x(), grid.max_x())
    alpha_y, beta_y = coefs(grid.min_y(), grid.max_y())
    alpha_z, beta_z = 0, 0

    alpha_coord = tf.convert_to_tensor([[alpha_x, alpha_y, alpha_z]],
                                       dtype=tf.double)
    beta_coord = tf.convert_to_tensor([[beta_x, beta_y, beta_z]],
                                      dtype=tf.double)

    alpha_targets, beta_targets = coefs(min_dB, max_dB)

    unnorm_pars = (1 / alpha_targets, -beta_targets / alpha_targets)

    ds_out = ds.map(normalize)

    return ds_out, unnorm_pars


class MetaMapEstimator(MapEstimator):
    def __init__(self, metaestimator, **kwargs):

        super().__init__(**kwargs)

        self._train_history = OrderedDict()

        self.metaestimator = metaestimator

    def train(self,
              metatrain_tasks=None,
              metaval_tasks=None,
              metalearning_rate=0.01,
              num_epochs=None,
              num_meas_train=None,
              info=dict(),
              save_to=None,
              **kwargs):
        """Args:

           metatrain_tasks: list or tf.data.Dataset of mapping tasks,
           each task is a tuple (m_in, m_out), where m_in is
           num_measurements x num_dims_grid and m_out is
           num_measurements x num_chanels.

           metaval_tasks=None: same form as metatrain_tasks.

           num_meas_train: number of measurements of each task used
           for training. The rest are used for testing in the inner
           loop.

           num_metaepochs=1,
           metalearning_rate=0.01,
           metavalidation_interval=5,
           verbose=0,
           tasks_per_batch=8,
           iterative_validation=True,           
           clip_norm=1e100
           shuffle_tasks=True

          `random_split`: if True and the global seed is not set, then
          the set of measurements used for training within each task
          is randomly selected at each epoch. The split is always
          random in the validation set if the global seed is not set.

           `info`: dictionary with information about training. The
           keys can be freely selected but must not equal "d_metrics".

        Returns: dict with metrics.

        See metaestimator.Metaestimator.metafit for
        more info.

        """
        def check_num_meas_train(num_meas_train):

            if num_meas_train is None:
                raise ValueError("`num_meas_train` must be provided")
            if isinstance(metatrain_tasks, tf.data.Dataset):
                first_entry = next(iter(metatrain_tasks))
            else:
                first_entry = metatrain_tasks[0]

            if num_meas_train >= len(first_entry[0]):
                raise ValueError(
                    "`num_meas_train` must be smaller than the number of measurements."
                )

        if num_epochs is None:
            # TODO: remove this argument from metafit.
            if hasattr(self, "num_epochs"):
                num_epochs = self.num_epochs
            else:
                num_epochs = 0

        if "split" in kwargs.keys():
            raise TypeError("Argument `split` is no longer supported. ")

        check_num_meas_train(num_meas_train)

        metatrain_tasks = self._separate_channels(metatrain_tasks)
        if metaval_tasks:
            metaval_tasks = self._separate_channels(metaval_tasks)

        # metaoptimizer = tf.keras.optimizers.Adam(
        #     learning_rate=metalearning_rate)

        d_metrics = self.metaestimator.metafit_with_initialization_search(
            metatrain_tasks=metatrain_tasks,
            metaval_tasks=metaval_tasks,
            #metaoptimizer=metaoptimizer,
            metalearning_rate=metalearning_rate,
            num_epochs=num_epochs,
            split=num_meas_train,
            fun_save=lambda d_metrics, dt_start=datetime.now(): self.save(
                save_to, info=info, d_metrics=d_metrics, dt_start=dt_start)
            if save_to else None,
            **kwargs)

        return d_metrics

    # TODO: rename as "estimate_on_grid" (also in other MapEstimators)
    def estimate(self, m_measurement_loc, m_measurements, num_epochs=None):
        """Args:

            `m_measurement_loc`: num_measurements x num_dims_grid matrix

            `m_measurements`: num_measurements x num_channels matrix

            `num_epochs=self.num_epochs`

        Returns:

            dictionary with keys:

                `t_power_map_est`: num_grid_points_x x
                num_grid_points_y x num_channels tensor with the map
                estimate.

        """
        if num_epochs is None:
            # TODO: remove this argument from metafit.
            if hasattr(self, "num_epochs"):
                num_epochs = self.num_epochs
            else:
                num_epochs = 0

        t_power_map_est = np.array([
            self._estimate_power_map_one_channel(m_measurement_loc,
                                                 m_measurements[:, ind_ch],
                                                 num_epochs)
            for ind_ch in range(m_measurements.shape[1])
        ])

        d_map_est = {"t_power_map_estimate": t_power_map_est}

        return d_map_est

    # TODO: Rename as _estimate_power_map_one_channel_on_grid
    def _estimate_power_map_one_channel(self, m_measurement_loc,
                                        v_measurements, num_epochs):
        """
        Args:

            `m_measurement_loc`: num_meas x num_dims_grid matrix
        
            `v_measurements`: length num_meas vector

        """

        self.metaestimator.estimator.fit_task(m_measurement_loc,
                                              v_measurements[:,
                                                             None], num_epochs)

        #        if not isinstance(self.grid, RectangularGrid):
        #            raise NotImplementedError

        # Can be thought of as a matrix of the same size as the
        # grid. Each entry of the matrix is a 3-vector.
        coord_array = np.transpose(self.grid.t_coordinates, axes=(1, 2, 0))

        # Estimate for each row separately
        l_map_out = []
        for row in coord_array:
            l_map_out.append(self.metaestimator.estimator(row)[:, 0])

        return np.array(l_map_out)

    def _estimate_power_map_one_channel_at_loc(self,
                                               m_measurement_loc,
                                               v_measurements,
                                               m_test_loc,
                                               num_epochs=None):
        """See parent.
        """

        if num_epochs is None:
            if hasattr(self, "num_epochs"):
                num_epochs = self.num_epochs
            else:
                num_epochs = 1

        self.metaestimator.estimator.fit_task(
            tf.convert_to_tensor(m_measurement_loc),
            tf.convert_to_tensor(v_measurements[:, None]), num_epochs)
        return self.metaestimator.estimator(tf.convert_to_tensor(m_test_loc))

    @staticmethod
    def _separate_channels(tasks_in):
        """Args:

           tasks_in: list or tf.data.Dataset of mapping tasks, each
           task is a tuple (m_in, m_out), where m_in is
           num_measurements x num_dims_grid and m_out is
           num_measurements x num_chanels.

        Returns:
        
           tasks_out: list or tf.data.Dataset of mapping tasks, each
           task is a tuple (m_in, m_out), where m_in is
           num_measurements x num_dims_grid and m_out is
           num_measurements x 1. From each entry in tasks_in,
           num_channels entries in tasks_out are produced.

        """

        if isinstance(tasks_in, tf.data.Dataset):
            first = next(iter(tasks_in))
            _, targets = first
            if targets.shape[1] > 1:
                raise NotImplementedError
            else:
                # Nothing to separate
                return tasks_in
        else:

            tasks_out = []
            for m_measurement_loc, m_measurements in tasks_in:

                for v_channel_meas in tf.transpose(m_measurements):

                    task = (m_measurement_loc, v_channel_meas[:, None])
                    tasks_out.append(task)

            return tasks_out

    def print_train_history(self):

        for key, item in self.train_history.items():
            print("* ", key)
            copy_item = copy.deepcopy(item)
            if "d_metrics" in item.keys():
                copy_item["d_metrics"] = "d_metrics info available"
            print("     ", copy_item)

    @property
    def train_history(self):
        return self._train_history

    @classmethod
    def load(cls, filename):
        d_objs = dill.load(open(filename, "rb"))

        metapars = d_objs["metapars"]
        del d_objs["metapars"]
        _train_history = d_objs["train_history"]
        del d_objs["train_history"]

        assert cls == d_objs[
            "class"], f"The estimator stored in file {filename} is not of class {cls}"
        del d_objs["class"]

        # est = cls(grid=d_objs["grid"],
        #           l_dim_layers=d_objs["l_dim_layers"],
        #           num_context_pars=d_objs["num_context_pars"],
        #           num_epochs=d_objs["num_epochs"],
        #           learning_rate=d_objs["learning_rate"],
        #           output_distr=d_objs["output_distr"],
        #           min_std=d_objs["min_std"])

        est = cls(**d_objs)

        est.metaestimator.estimator.metapars = metapars
        est._train_history = _train_history

        return est

    def save(self, filename, info=dict(), d_metrics=None, dt_start=None):
        """Saves `self` as well as additonal information to the file with name
        `filename`. Among the additional information is an ordered
        dictionary whose keys are time instants and the values are
        dictionaries with information about the training session
        starting at that time instant.

        Args: 

        `info`: optional dictionary provided by the user with info
        about the training session.

        """

        if dt_start is None:
            dt_start = datetime.now()

        info["d_metrics"] = d_metrics
        self._train_history[dt_start] = info

        d_objs = {
            "class": self.__class__,
            "grid": self.grid,
            "output_distr": self.metaestimator.estimator.output_distr,
            "min_std": self.metaestimator.estimator.min_std,
            "metapars": self.metaestimator.estimator.metapars,
            "train_history": self._train_history
        }

        d_objs.update(self._subclass_properties())

        print(f"    Saving estimator to {filename}")
        dill.dump(d_objs, open(filename, "wb"))

    # To be extended by subclasses
    def _subclass_properties(self):
        raise NotImplementedError


class CaviaMapEstimator(MetaMapEstimator):
    def __init__(self,
                 grid,
                 l_dim_layers=[4, 5, 1],
                 num_context_pars=4,
                 num_epochs=1,
                 learning_rate=0.05,
                 output_distr=None,
                 min_std=0.01,
                 **kwargs):

        self._num_epochs = num_epochs
        self.grid = grid

        estimator = CaviaSubNetwork(dim_input=self.grid.dim,
                                    l_dim_layers=l_dim_layers,
                                    num_context_pars=num_context_pars,
                                    max_num_iter=self.num_epochs,
                                    learning_rate=learning_rate,
                                    output_distr=output_distr,
                                    min_std=min_std)

        metaestimator = GradientMetaestimator(estimator=estimator)
        super().__init__(metaestimator, **kwargs)

    def _subclass_properties(self):
        """See parent class."""
        return {
            "l_dim_layers": self.metaestimator.estimator.l_dim_layer,
            "num_context_pars": self.metaestimator.estimator.num_context_pars,
            "num_epochs": self.num_epochs,
            "learning_rate": self.metaestimator.estimator.learning_rate,
        }

    @property
    def num_epochs(self):
        return self._num_epochs

    @num_epochs.setter
    def num_epochs(self, val):
        self._num_epochs = val
        #set_trace()
        self.metaestimator.estimator.trace__fit_task(self._num_epochs + 1)


class CNPMapEstimator(MetaMapEstimator):

    name_on_figs = "CNP"

    # TODO: merge constructor with the one of CaviaMapEstimator
    def __init__(self,
                 grid,
                 l_dim_layers_enc=[],
                 l_dim_layers_dec=[],
                 output_distr=None,
                 min_std=0.01,
                 **kwargs):

        self.grid = grid

        estimator = CNPSubNetwork(dim_input=self.grid.dim,
                                  l_dim_layers_enc=l_dim_layers_enc,
                                  l_dim_layers_dec=l_dim_layers_dec,
                                  output_distr=output_distr,
                                  min_std=min_std)

        metaestimator = GradientMetaestimator(estimator=estimator)

        super().__init__(metaestimator, **kwargs)

    def _subclass_properties(self):
        """See parent class."""
        return {
            "l_dim_layers_enc": self.metaestimator.estimator.l_dim_layer_enc,
            "l_dim_layers_dec": self.metaestimator.estimator.l_dim_layer_dec,
        }


class GradMetaMapEstimator(MetaMapEstimator):
    def __init__(self,
                 grid,
                 output_distr=None,
                 min_std=0.01,
                 estimator=None,
                 name_on_figs=None,
                 **kwargs):

        self.grid = grid  # needed?

        assert estimator

        if name_on_figs is None:
            self.name_on_figs = str(estimator.__class__)
        else:
            self.name_on_figs = name_on_figs

        metaestimator = GradientMetaestimator(estimator=estimator)

        super().__init__(metaestimator, **kwargs)

        
class FixedTaskLengthEstimator(GradMetaMapEstimator):
    def __init__(self,
                 grid,
                 output_distr=None,
                 min_std=0.01,
                 name_on_figs="Fixed length",
                 **kwargs):

        self.grid = grid  # needed?

        self.name_on_figs = name_on_figs

        estimator = FixedTaskLengthSubNetwork(dim_input=self.grid.dim,
                                              output_distr=output_distr,
                                              min_std=min_std)

        super().__init__(grid,
                         estimator=estimator,
                         name_on_figs=name_on_figs,
                         **kwargs)

    def _subclass_properties(self):
        """See parent class."""
        return {}


class SoftNNMapEstimator(GradMetaMapEstimator):
    def __init__(self,
                 grid,
                 output_distr=None,
                 min_std=0.01,
                 name_on_figs="Fixed length",
                 **kwargs):

        self.grid = grid  # needed?

        self.name_on_figs = name_on_figs

        estimator = SoftNNSubNetwork(dim_input=self.grid.dim,
                                     output_distr=output_distr,
                                     min_std=min_std)

        super().__init__(grid,
                         estimator=estimator,
                         name_on_figs=name_on_figs,
                         **kwargs)

        # Force weight initialization
        self.estimate_at_loc(tf.zeros((1,self.grid.dim), dtype=tf.double),
                             tf.zeros((1,1), dtype=tf.double),
                             tf.zeros((1,self.grid.dim), dtype=tf.double)
                             )

    def _subclass_properties(self):
        """See parent class."""
        return {}
