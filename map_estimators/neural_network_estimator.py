import scipy
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import norm
from utilities import empty_array, np
from map_estimators.map_estimator import MapEstimator
from IPython.core.debugger import set_trace
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Conv2DTranspose, Reshape, MaxPool2D, UpSampling2D
from tensorflow.keras import Model
import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from map_estimators.fully_conv_nn_arch import FullyConvolutionalNeuralNetwork
tfd = tfp.distributions


class NeuralNetworkEstimator(MapEstimator):
    """
    `metric`: indicates the estimated metric:
      - "power": metric = received power
      - "service": metric = posterior probability of received power >= `self.min_service_power`.
      -`f_shadowing_covariance` is a function of a distance argument that returns
            the covariance between the map values at two points separated by
            that distance.
        -`f_mean`: if None, map assumed zero mean. Else, this is a function of the ind_channel and
        location.
    """
    name_on_figs = "NeuralNetworkEstimator"
    estimation_fun_type = 'g2g'

    def __init__(
            self,
            # grid,
            estimator=None,
            nn_arch_id='1',
            # metric="power",  # values: "power", "service"
            # f_shadowing_covariance=None,
            **kwargs):

        # self.grid = grid
        # self.metric = metric
        # self.f_shadowing_covariance = f_shadowing_covariance
        # self.f_mean = f_mean

        # self.dumping_factor = dumping_factor

        # self.m_all_measurements = None  # num_sources x num_measurements_so_far
        # self.m_all_mask = None
        # self.m_all_measurement_loc = None  # 3 x num_measurements_so_far

        # self.m_building_meta_data = None
        # self._m_building_meta_data_grid = None

        if estimator is None:
            # self.estimator = FullyConvolutionalNeuralNetwork(nn_arch_id=nn_arch_id)
            self.estimator = FullyConvolutionalNeuralNetwork.\
                get_fully_conv_nn_arch(nn_arch_id=nn_arch_id)
        else:
            self.estimator = estimator

        super().__init__(**kwargs)

    @property
    def m_building_meta_data_grid(self):
        """
        Returns:
        `_m_building_meta_data_grid`, Ny x Nx matrix whose (i,j) entry is 1 if the
        grid point is inside the building.
        """
        # if self.m_meta_data is None:
        #     print('Please provide the building meta_data.')
        # self._m_building_meta_data_grid = np.zeros((self.grid.num_points_y,
        #                                             self.grid.num_points_x))
        if self.m_building_meta_data is None and self._m_building_meta_data_grid is None:
            self._m_building_meta_data_grid = np.zeros((self.grid.num_points_y,
                                                        self.grid.num_points_x))

        if self._m_building_meta_data_grid is None and self.m_building_meta_data is not None:
            self._m_building_meta_data_grid = np.zeros((self.grid.num_points_y,
                                                        self.grid.num_points_x))
            for v_building_meta_data in self.m_building_meta_data:
                v_meta_data_inds = self.grid.nearest_gridpoint_inds(v_building_meta_data)
                self._m_building_meta_data_grid[v_meta_data_inds[0],
                                                v_meta_data_inds[1]] = 1

        return self._m_building_meta_data_grid

    def estimate_g2g(self, measurement_loc=None, measurements=None, building_meta_data=None, test_loc=None):
        """
        Args:
            - `measurements`: num_sources x self.grid.num_points_y x self.grid.num_points_x
            - `building_meta_data`: self.grid.num_points_y x self.grid.num_points_x whose
                    value at each grid point is 1 if the grid point is inside a building
            - `measurement_locs`: self.grid.num_points_y x self.grid.num_points_x, is
                    a sampling mask whose value is 1 at the grid point where
                    the samples were taken.

        Returns:
            `d_map_estimate`: dictionary whose fields are:

           - "t_power_map_estimate" :  tensor whose (i,j,k)-th entry is the
           estimated power of the i-th channel at grid point (j,k).
           - "t_power_map_norm_variance" : Contains the variance a
           posteriori normalized to the interval (0,1).
           - "t_service_map_estimate": [only if self.min_service_power
           is not None]. Each entry contains the posterior probability
           that the power is above `self.min_service_power`.
           - "t_service_map_entropy": [only if self.min_service_power
           is not None] Entropy of the service indicator.
        """

        d_map_estimate = self.estimate_metric_per_channel(measurement_loc, measurements,
                                                          building_meta_data, test_loc,
                                                          f_power_est=self.estimate_power_one_channel,
                                                          f_service_est=self.estimate_service_one_gaussian_channel)
        return d_map_estimate

    def estimate_power_one_channel(self, ind_channel, measurement_loc, measurements, building_meta_data):
        """
        Args:
            - `measurements`: num_sources x self.grid.num_points_y x self.grid.num_points_x
            - `building_meta_data`: self.grid.num_points_y x self.grid.num_points_x whose
                    value at each grid point is 1 if the grid point is inside a building
            - `measurement_locs`: self.grid.num_points_y x self.grid.num_points_x, is
                    a sampling mask whose value is 1 at the grid point where
                    the samples were taken.

        Returns: self.grid.num_points_y x self.grid.num_points_x matrices of
            the estimated posterior mean and the posterior variance of ind_channel.
        """
        if building_meta_data is None:
            building_meta_data = np.zeros((self.grid.num_points_y,
                                           self.grid.num_points_x))

        # mask whose (j, k) entry is 1 if sampled taken, 0 if not taken,
        # and -1 if it is inside the building
        m_mask_with_meta_data = measurement_loc - building_meta_data

        m_estimated_pow_of_source, m_variance = self.estimate_power_one_channel_grid(
            ind_channel=ind_channel,
            t_measurements_gird=measurements,
            t_mask=m_mask_with_meta_data[None, :, :])

        return m_estimated_pow_of_source, m_variance

    def estimate_power_one_channel_grid(self, ind_channel, t_measurements_gird, t_mask):
        """
            Args:
                - `t_measurements_gird`: num_sources x self.grid.num_points_y x self.grid.num_points_x
                - `t_mask`: self.grid.num_points_y x self.grid.num_points_x is a concatenation of
                        a sampling mask and building meta data whose value is:
                        1 at the grid point where the samples were taken,
                        -1 if the grid point is inside a building, 0 elsewhere.

            Returns: self.grid.num_points_y x self.grid.num_points_x matrices of
                the estimated posterior mean and the posterior variance of ind_channel.
        """
        x_test = t_measurements_gird[ind_channel, :, :]
        meas_pow = x_test
        # get the mask and concatenate with the measurement
        m_mask = t_mask[0, :, :]
        # x_test = np.hstack((x_test, m_mask))

        # stack along the first dimension for channels x Ny x Nx
        x_test = np.stack((x_test, m_mask))

        # make measured data of dim = 4 for compatibility with the training data set of NN
        x_test = x_test[tf.newaxis, ...].astype("float32")

        # mu_pred, sigma_pred = self.estimator(x_test)

        # output from ForkConvNn architecture is a single tensor of
        # shape (1, 2, Ny, Nx)
        prediction = self.estimator(x_test)
        mu_pred = prediction[:,0,...]
        sigma_pred = prediction[:, 1,...]
        # set_trace()

        predicted_map = mu_pred  # mean posterior power
        # variance
        # sigma_pred = y_predict.scale.numpy()
        sigma_pred = sigma_pred

        # y_predict = tfd.Normal(loc=mu_pred, scale=sigma_pred).sample().numpy()
        m_estimated_pow_of_source = np.reshape(predicted_map[0],
                                               (self.grid.num_points_y,
                                                self.grid.num_points_x))
        # m_estimated_pow_of_source = np.where(m_mask == 1, meas_pow, m_estimated_pow_of_source)
        # (i,j)-th entry contains the posterior variance of the power at the (i,j)-th grid point
        m_variance = np.reshape(np.square(sigma_pred[0]),
                                (self.grid.num_points_y, self.grid.num_points_x))
        # m_variance = np.where(m_mask == 1, 0.03, m_variance)

        if (m_variance < 0).any():
            set_trace()
        # m_uncertainty = np.zeros((self.grid.num_points_x, self.grid.num_points_y))

        return m_estimated_pow_of_source, m_variance

    ###########################################################################################
    def estimate_g2g_old(self, measurement_loc=None, measurements=None, building_meta_data=None):
        """Args:
            `measurement` : num_measurements x num_sources matrix
            with the measurements at each channel.
            `measurement_locs`: num_measurements x 3 matrix with the
            3D locations of the measurements.
            `m_meta_data`, num_points x 3 matrix with the 3D locations of the buildings
           Returns:
           -`d_map_estimate`: dictionary whose fields are num_sources
           x self.grid.num_points_y x self.grid.num_points_x
           tensors. They are:
           "t_power_map_estimate" :  tensor whose (i,j,k)-th entry is the
           estimated power of the i-th channel at grid point (j,k).
           "t_power_map_norm_variance" : Contains the variance a
           posteriori normalized to the interval (0,1).
           "t_service_map_estimate": [only if self.min_service_power
           is not None]. Each entry contains the posterior probability
           that the power is above `self.min_service_power`.
           "t_service_map_entropy": [only if self.min_service_power
           is not None] Entropy of the service indicator.
        """
        # Ny x Nx buffer matrix for a building meta data
        if self.m_building_meta_data is None:
            self.m_building_meta_data = building_meta_data

        num_sources = measurements.shape[1]
        # get the measurements
        for loc, meas in zip(measurement_loc, measurements):
            self.store_measurement_old(loc, meas)

        # Now estimate metric per channel
        t_power_map_est = empty_array(
            (num_sources, self.grid.num_points_y, self.grid.num_points_x))
        t_power_map_var = empty_array(
            (num_sources, self.grid.num_points_y, self.grid.num_points_x))
        if self.min_service_power is not None:
            t_service_map_est = empty_array(
                (num_sources, self.grid.num_points_y, self.grid.num_points_x))
            t_service_map_ent = empty_array(
                (num_sources, self.grid.num_points_y, self.grid.num_points_x))
        for ind_src in range(num_sources):
            t_power_map_est[ind_src, :, :], t_power_map_var[ind_src, :, :] = \
                self.estimate_power_one_channel_old(ind_src)
            if self.min_service_power is not None:
                t_service_map_est[ind_src, :, :], t_service_map_ent[ind_src, :, :] = \
                    self.estimate_service_one_gaussian_channel(t_power_map_est[ind_src, :, :],
                                                               t_power_map_var[ind_src, :, :])
                # self.estimate_service_one_gaussian_channel(self.m_all_measurements[ind_src, :], ind_src)

        # m_entropy = np.mean(t_power_map_var, axis=0)

        d_map_estimate = {
            "t_power_map_estimate":
            t_power_map_est,
            "t_power_map_norm_variance":
            t_power_map_var / self.f_shadowing_covariance(0)  # ensure in (0,1)
        }
        if self.min_service_power:
            d_map_estimate["t_service_map_estimate"] = t_service_map_est
            d_map_estimate["t_service_map_entropy"] = t_service_map_ent

        return d_map_estimate

    def convert_measurements_to_grid_form(self, m_all_measurements_loc, m_all_measurements):
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
            (num_sources, self.grid.num_points_y, self.grid.num_points_x))

        t_mask = np.zeros(
            (1, self.grid.num_points_y, self.grid.num_points_x))

        m_all_measurements_loc_trans = m_all_measurements_loc.T

        m_all_measurements_col_index = 0  # to iterate through column of measurements

        # buffer counter to count repeated measurement in the grid point
        m_counter = np.zeros(np.shape(t_mask))

        for v_measurement_loc in m_all_measurements_loc_trans:

            # Find the nearest indices of the grid point closet to v_measurement_loc
            v_meas_loc_inds = self.grid.nearest_gridpoint_inds(v_measurement_loc)

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

    def estimate_power_one_channel_grid_old(self, ind_channel, t_measurements_gird, t_mask):


        x_test = t_measurements_gird[ind_channel, :, :]

        # get the mask and concatenate with the measurement
        m_mask = t_mask[0, :, :]
        # x_test = np.hstack((x_test, m_mask))

        # stack along the first dimension for channels x Ny x Nx
        x_test = np.stack((x_test, m_mask))

        # make measured data of dim = 4 for compatibility with the training data set of NN
        x_test = x_test[tf.newaxis, ...].astype("float32")

        mu_pred, sigma_pred = self.estimator(x_test)

        # y_predict = self.estimator(x_test)
        # predicted_map = y_predict.sample().numpy()
        # predicted_map = y_predict.loc.numpy()  # mean posterior power

        predicted_map = mu_pred  # mean posterior power
        # variance
        # sigma_pred = y_predict.scale.numpy()
        sigma_pred = sigma_pred

        # y_predict = tfd.Normal(loc=mu_pred, scale=sigma_pred).sample().numpy()
        m_estimated_pow_of_source = np.reshape(predicted_map[0],
                                               (self.grid.num_points_y,
                                                self.grid.num_points_x))
        # (i,j)-th entry contains the posterior variance of the power at the (i,j)-th grid point
        m_variance = np.reshape(np.square(sigma_pred[0]),
                                (self.grid.num_points_y, self.grid.num_points_x))

        if (m_variance < 0).any():
            set_trace()
        # m_uncertainty = np.zeros((self.grid.num_points_x, self.grid.num_points_y))

        return m_estimated_pow_of_source, m_variance

    def estimate_power_one_channel_old(self, ind_channel):

        output_shape = self.grid.num_points_y * self.grid.num_points_x

        t_measurements_gird, t_mask = \
            self.convert_measurements_to_grid_form(m_all_measurements=self.m_all_measurements,
                                                   m_all_measurements_loc=self.m_all_measurement_loc)

        m_estimated_pow_of_source, m_variance = self.estimate_power_one_channel_grid_old(
            ind_channel=ind_channel,
            t_measurements_gird=t_measurements_gird,
            t_mask=t_mask)

        return m_estimated_pow_of_source, m_variance


class ConvolutionalVAE(Model):
    def __init__(self, latent_dim):
        super(ConvolutionalVAE, self).__init__()

        self.conv1 = Conv2D(filters=64,
                            kernel_size=3,
                            activation=tf.nn.leaky_relu,
                            padding='same')
        self.conv1T = Conv2DTranspose(64, 3,
                                      activation=tf.nn.leaky_relu,
                                      padding='same')
        self.conv2 = Conv2D(32, 3,
                            activation=tf.nn.leaky_relu,
                            padding='same')
        self.conv2T = Conv2DTranspose(32, 3,
                                      activation=tf.nn.leaky_relu,
                                      padding='same')

        self.max_pool_1 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_1 = UpSampling2D(size=(2, 2))

        self.conv3 = Conv2D(16, 3,
                            activation=tf.nn.leaky_relu,
                            padding='same')
        self.conv3T = Conv2DTranspose(16, 3,
                                      activation=tf.nn.leaky_relu,
                                      padding='same')

        self.conv4 = Conv2D(16, 3,
                            activation=tf.nn.leaky_relu,
                            padding='same')
        self.conv4T = Conv2DTranspose(16, 3,
                                      activation=tf.nn.leaky_relu,
                                      padding='same')

        self.max_pool_2 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_2 = UpSampling2D(size=(2, 2))

        self.conv5 = Conv2D(8, 3,
                            activation=tf.nn.leaky_relu,
                            padding='same')
        self.conv5T = Conv2DTranspose(8, 3,
                                      activation=tf.nn.leaky_relu,
                                      padding='same')
        self.mu_encoder = Conv2D(latent_dim, 3,
                                 activation=tf.nn.leaky_relu,
                                 padding='same')

        self.sigma_encoder = Conv2DTranspose(latent_dim, 3,
                                             activation=lambda x: tf.nn.elu(x) + 1,
                                             padding='same')

        self.conv_out_mu = Conv2DTranspose(1, 3,
                                           activation=tf.nn.leaky_relu,
                                           padding='same')

        self.conv_out_sigma = Conv2DTranspose(1, 3,
                                              activation=lambda x: tf.nn.elu(x) + 1,
                                              padding='same')
        # self.conv1 = Conv2D(64, 3, strides=2, activation=tf.nn.leaky_relu, padding="same")
        # self.conv1T = Conv2DTranspose(64, 3, strides=2, activation=tf.nn.leaky_relu, padding="same")
        # self.conv2T = Conv2DTranspose(1, 3, activation=tf.nn.leaky_relu, padding="same")
        # self.flatten = Flatten()
        # self.Layer_1 = Dense(256, activation=tf.nn.leaky_relu)
        # self.Layer_2 = Dense(9 * 10 * 64, activation=tf.nn.leaky_relu)
        # self.reshape = Reshape(target_shape=(9, 10, 64))
        # self.mu = Dense(latent_dim, tf.nn.leaky_relu)
        # self.sigma = Dense(latent_dim, activation=lambda x: tf.nn.elu(x) + 1)
        # self.mu_decoder = Dense(output_shape, tf.nn.leaky_relu)
        # self.sigma_decoder = Dense(output_shape, activation=lambda x: tf.nn.elu(x) + 1)
        self.l_train_loss = []
        self.l_validation_loss = []
        self.l_epochs = []
        self.optimizer = None
        self.train_loss = None
        self.test_loss = None

    def encoder(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool_1(x)
        x = self.conv4(x)
        # x = self.max_pool_2(x)
        x = self.conv5(x)
        mu = self.mu_encoder(x)
        sigma = self.sigma_encoder(x)

        # return tfd.Normal(loc=self.mu(x), scale=self.sigma(x))  # dstr of latent variables
        return [mu, sigma]

    def decoder(self, latent_dist):

        z = latent_dist.sample()
        x = self.conv5T(z)
        # x = self.up_sampling_2(x)
        x = self.conv4T(x)
        x = self.up_sampling_1(x)
        x = self.conv3T(x)
        x = self.conv2T(x)
        x = self.conv1T(x)
        mu = self.conv_out_mu(x)
        sigma = self.conv_out_sigma(x)
        # return tfd.Normal(loc=self.mu_decoder(x), scale=self.sigma_decoder(x))  # posterior map distribution
        return [mu, sigma]

    def loss(self, true_targets, posterior):
        """ ELBO """
        # y_true = self.flatten(true_targets)  # to make shape = (batchsize, output_shape)
        reconstruction_loss = -tf.reduce_mean(posterior.log_prob(true_targets))
        kl_loss = 1 + tf.math.log(tf.square(posterior.scale)) - \
                  tf.square(posterior.loc) - \
                  tf.square(posterior.scale)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss
        return total_loss  # -tf.reduce_mean(posterior.log_prob(y_true))

    def call(self, x):
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        # x = self.InputLayer
        mu_encoder, sigma_encoder = self.encoder(x)
        return self.decoder(tfd.Normal(loc=mu_encoder, scale=sigma_encoder))

    # # Optimizer
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    #
    # train_loss = tf.keras.metrics.Mean(name='loss')
    #
    # val_loss = tf.keras.metrics.Mean(name='val_loss')

    @tf.function
    def train_step(self, data, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).

            # predictions = self(data, training=True)
            mu_pred, sigma_pred = self(data, training=True)
            predictions = tfd.Normal(loc=mu_pred, scale=sigma_pred)

            # Negative log likelihood
            loss = self.loss(true_targets=labels, posterior=predictions)

        gradients = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)

    @tf.function
    def test_step(self, data, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        # predictions = self(data, training=False)
        mu_pred, sigma_pred = self(data, training=False)
        predictions = tfd.Normal(loc=mu_pred, scale=sigma_pred)

        t_loss = self.loss(labels, predictions)

        self.test_loss(t_loss)

        # return mu_pred, sigma_pred

    def fit(self, dataset, epochs, learning_rate=None):

        if learning_rate is None:
            raise Exception('Learning rate is not provided.')

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.train_loss = tf.keras.metrics.Mean(name='loss')
        self.test_loss = tf.keras.metrics.Mean(name='val_loss')

        for ind_epoch in range(epochs):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.test_loss.reset_states()

            for x_data, y_labels in dataset[0]:
                """Convert the input data into the`NHWC` format
                for tensorflow, ie. channels_last"""
                # x_data = tf.transpose(x_data, perm=[0, 2, 3, 1])
                y_labels = tf.transpose(y_labels, perm=[0, 2, 3, 1])

                self.train_step(x_data, y_labels)

            for test_x_data, test_y_labels in dataset[1]:
                """Convert the input data into the `NHWC` format 
                for tensorflow, ie. channels_last"""
                # test_x_data = tf.transpose(test_x_data, perm=[0, 2, 3, 1])
                test_y_labels = tf.transpose(test_y_labels, perm=[0, 2, 3, 1])

                self.test_step(test_x_data, test_y_labels)

            print(f'Epoch {ind_epoch + 1}, '
                  f'Loss: {self.train_loss.result()}, '
                  f'Validation Loss: {self.test_loss.result()}')

            # store the train and test loss
            self.l_train_loss = np.append(self.l_train_loss, self.train_loss.result())
            self.l_validation_loss = np.append(self.l_validation_loss, self.test_loss.result())
            self.l_epochs = np.append(self.l_epochs, ind_epoch)
