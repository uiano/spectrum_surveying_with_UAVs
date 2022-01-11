import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, \
    Conv2DTranspose, Reshape, MaxPool2D, UpSampling2D, BatchNormalization, add, \
    Dropout, concatenate
from tensorflow.keras import Model, regularizers
import numpy as np
import tensorflow_probability as tfp
from IPython.core.debugger import set_trace
from utilities import list_are_close
tfd = tfp.distributions
import pickle
import keras.backend as K
from measurement_generation.measurement_dataset import GridMeasurementDataset
tf.config.run_functions_eagerly(True)
class FullyConvolutionalNeuralNetwork(Model):
    """Convolutional Neural Network class to get the posterior
    Gaussian distribution of the predicted power map."""

    def __init__(self):
        super(FullyConvolutionalNeuralNetwork, self).__init__()

        # buffer for loss
        self.l_train_loss = []
        self.l_validation_loss = []
        self.l_epochs = []
        self.l_train_loss_mean = []
        self.l_train_loss_sigma = []
        self.l_val_loss_mean = []
        self.l_val_loss_sigma = []

        # metrics
        self.optimizer = None
        self.train_loss = None
        self.val_loss = None
        self.loss_metric = None
        self.b_posterior_target = False
        # self.l_loss_weightage = [1.0, 1.1]
        self.alpha = None
        self._b_trainable_combined_layers = True # if False freeze the combined layers
        self._b_trainable_mean_layers = True    # if False freeze the mean block layers
        self._b_trainable_std_layers = True     # if False freeze the std. block layers
        self.train_loss_mean = None
        self.train_loss_sigma = None
        self.val_loss_sigma = None
        self.val_loss_mean = None

    # a factory method that would instantiate the class given its name:
    @staticmethod
    def get_fully_conv_nn_arch(nn_arch_id):
        return globals()[f'FullyConvNnArch' + nn_arch_id]()

    @tf.function
    def call(self, x):
        """A __call() method of keras. First the input `x`
        in the form 'NCHW' is converted into the form `NHWC` format
        i.e. channels_last  in tensorflow.
        Returns:
            [mu, sigma]: where `mu` and `sigma` are the learned mean and
            standard deviation of the distribution by a NN. Their shape is in the form
            NHWC ( 1 x num_points_y x num_points_x x 1)"""
        # # Convert the input data into the`NHWC` format
        # #    for tensorflow, ie. channels_last
        # x = tf.transpose(x, perm=[0, 2, 3, 1])
        pass

    # loss function
    def loss_function(self, y_true, y_predict):
        """
        Args:
            if self.b_posterior_target:
                -`y_true`: is N x H x W x C tensor with channels C=2, where first channel
                    contains a posterior mean target and the other contains
                    a posterior standard deviation.
            else:
                -`y_true`: is N x H x W x C tensor tha contains C= num_sources true maps as a target.
                    if channels are combined then C = 1.

            -`y_predict`: is N x H x W x 1 tensorflow normal distribution whose 'loc' is the estimated
                    posterior mean and 'scale' is the estimated  standard

        """
        y_true = tf.transpose(y_true, perm=[0, 2, 3, 1])
        # y_true = y_true[..., None] # to make the shape equivalent to that of the output of a neural estimator
        # y_predict = self.flatten(y_predict)

        if self.loss_metric is None or self.loss_metric == 'NLL':
            'The loss function is a Negative Log Likelihood'
            return -tf.reduce_mean(y_predict.log_prob(y_true))

        elif self.loss_metric == 'MSE':
            'The loss function is Mean Square Error'
            if self.b_posterior_target:
                "A target contains a posterior mean and a posterior standard deviation"
                power_mse = tf.keras.losses.MSE(y_true[..., 0][..., None], y_predict.loc)
                std_mse = tf.keras.losses.MSE(y_true[..., 1][..., None], y_predict.scale)
            else:
                power_mse = tf.keras.losses.MSE(y_true, y_predict.loc)
                std_mse = tf.keras.losses.MSE(5*np.ones(y_true.shape), y_predict.scale)
            return self.l_loss_weightage[0] * power_mse + \
                   self.l_loss_weightage[1] * std_mse #[power_mse, std_mse]
        elif self.loss_metric == 'Custom':
            """The loss function is inspired from the paper
            Efficient estimation of conditional variance
            functions in stochastic regression.
            If self.alpha = 0, then freeze combined and standard dev. block layers.
            Elif self.alpha =1, then freeze combined and mean block layers.
            Else do not freeze any layers."""

            t_delta = (tf.square(y_true - y_predict.loc))
            # loss_sigma = tf.reduce_mean(tf.square(t_delta - tf.square(y_predict.scale)))
            loss_sigma = tf.reduce_mean(tf.square(tf.sqrt(t_delta) - y_predict.scale))
            loss_mean = tf.reduce_mean(t_delta)
            loss = self.alpha * loss_sigma + (1 - self.alpha) * loss_mean
            return loss, loss_sigma, loss_mean
        else:
            raise NotImplementedError

    # @tf.function
    def train_step(self, data, labels):

        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).

            # predictions = self(data, training=True)
            mu_pred, sigma_pred = self(data, training=True)
            predictions = tfd.Normal(loc=mu_pred, scale=sigma_pred)
            # loss, loss_mean, loss_sigma = self.loss_function(labels, predictions)
            # loss function
            if self.loss_metric == "Custom":
                loss, loss_mean, loss_sigma = self.loss_function(labels, predictions)
                self.train_loss_mean(loss_mean)
                self.train_loss_sigma(loss_sigma)
            else:
                loss = self.loss_function(labels, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)
        # self.train_loss_mean(loss1)

    # @tf.function
    def test_step(self, data, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        # predictions = self(data, training=False)

        mu_pred, sigma_pred = self(data, training=False)
        predictions = tfd.Normal(loc=mu_pred, scale=sigma_pred)

        if self.loss_metric == "Custom":
            val_loss, val_loss_mean, val_loss_sigma = self.loss_function(labels, predictions)
            self.val_loss_mean(val_loss_mean)
            self.val_loss_sigma(val_loss_sigma)
        else:
            val_loss = self.loss_function(labels, predictions)

        self.val_loss(val_loss)

        # return mu_pred, sigma_pred

    def fit(self, dataset, epochs, learning_rate=None,
            loss_metric=None, b_posterior_target = False,
            alpha = None, test_changes=False):
        """
        Args:
            -`dataset`: a list containing the training and validation dataset
            -`loss_metric`: a string that represent which loss function to be used
                if `loss_metric` is 'None' or 'NLL', then negative loglikelihood loss
                is calculated in the loss_function().
                elif `loss_metric` is MSE , then MSE loss is calculated in the loss_function()
                elif: `loss_metric` is Custom, then the combined conditional MSE loss is calculated
                in the loss_function()
            -`b_posterior_target`: if True use the loss function is MSE between posterior
                target and estimate posterior from the network.
            -`alpha`: a constant between 1 and 0 such that:
                if `loss_metric` = 'Custom':
                    if `alpha`==0, then the combined layers in CombinedLayers()
                    and the std. deviation layers in StdDeviationLayers() of
                    the network is set to freeze.
                elif:
                    if `alpha`== 1, then the combined layers in CombinedLayers()
                    and the mean layers in MeanLayers() of
                    the network is set to freeze.
                else:
                    all the layers are trainable.

        """

        if learning_rate is None:
            raise Exception('Learning rate is not provided.')

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.train_loss = tf.keras.metrics.Mean(name='loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.train_loss_mean = tf.keras.metrics.Mean(name='loss_mean')
        self.train_loss_sigma = tf.keras.metrics.Mean(name='loss_sigma')
        self.val_loss_mean = tf.keras.metrics.Mean(name='val_loss_mean')
        self.val_loss_sigma = tf.keras.metrics.Mean(name='val_loss_sigma')

        # loss metric
        if loss_metric == 'Custom':
            assert alpha is not None
            self.alpha = alpha
            if alpha == 0:
                print("alpha 0= ", alpha)
                # freeze combined lsayers and std. deviation layers
                self.combined_layers.trainable = False
                self.std_deviation_layers.trainable = False
                self.mean_layers.trainable = True

            elif alpha == 1:
                print("alpha 1 =", alpha)
                # freeze combined layers and mean layers
                self.combined_layers.trainable = False
                self.std_deviation_layers.trainable = True
                self.mean_layers.trainable = False
            else:
                print("alpha 0. 5")
                self.combined_layers.trainable = True
                self.std_deviation_layers.trainable = True
                self.mean_layers.trainable = True

        if self.loss_metric is None:
            self.loss_metric = loss_metric

        # posterior target variable
        self.b_posterior_target = b_posterior_target

        for ind_epoch in range(epochs):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.val_loss.reset_states()
            self.train_loss_mean.reset_states()
            self.train_loss_sigma.reset_states()
            self.val_loss_mean.reset_states()
            self.val_loss_sigma.reset_states()

            # def loss(*args):
            #     return self.loss_function(*args)
            for x_data, y_labels in dataset[0]:
                """Convert the input data into the`NHWC` format
                for tensorflow, ie. channels_last"""
                # x_data = tf.transpose(x_data, perm=[0, 2, 3, 1])
                # y_labels = tf.transpose(y_labels, perm=[0, 2, 3, 1])
                self.train_step(x_data, y_labels)
                if test_changes:
                    if ind_epoch==1:
                        l_initial_weight = [self.combined_layers.get_weights(),
                                            self.mean_layers.get_weights(),
                                            self.std_deviation_layers.get_weights()]

            for test_x_data, test_y_labels in dataset[1]:
                """Convert the input data into the `NHWC` format 
                for tensorflow, ie. channels_last"""
                # test_x_data = tf.transpose(test_x_data, perm=[0, 2, 3, 1])
                # test_y_labels = tf.transpose(test_y_labels, perm=[0, 2, 3, 1])

                self.test_step(test_x_data, test_y_labels)

            print(f'Epoch {ind_epoch + 1}, '
                  f'Loss: {self.train_loss.result():0.4f}, '
                  f'Validation Loss: {self.val_loss.result()}'
                  f' loss1: {self.train_loss_mean.result()}')

            # store the train and test loss
            self.l_train_loss = np.append(self.l_train_loss, self.train_loss.result())
            self.l_validation_loss = np.append(self.l_validation_loss, self.val_loss.result())
            self.l_epochs = np.append(self.l_epochs, ind_epoch)
            self.l_train_loss_mean = np.append(self.l_train_loss_mean, self.train_loss_mean.result())
            self.l_train_loss_sigma = np.append(self.l_train_loss_sigma, self.train_loss_sigma.result())
            self.l_val_loss_mean = np.append(self.l_val_loss_mean, self.val_loss_mean.result())
            self.l_val_loss_sigma = np.append(self.l_val_loss_sigma, self.val_loss_sigma.result())

        if test_changes:
            l_final_weight = [self.combined_layers.get_weights(),
                              self.mean_layers.get_weights(),
                              self.std_deviation_layers.get_weights()]

            print("Weight values of Combined layer are the same? ",
                  list_are_close(l_initial_weight[0], l_final_weight[0]))
            print("Weight values of mean layer are the same? ",
                  list_are_close(l_initial_weight[1], l_final_weight[1]))
            print("Weight values of std. layer are the same? ",
                  list_are_close(l_initial_weight[2], l_final_weight[2]))


class FullyConvNnArch1(FullyConvolutionalNeuralNetwork):

    """ Fully Convolutional Neural Network with 5 convolution
    and 2 MaxPool Layers"""

    def __init__(self):
        super(FullyConvNnArch1, self).__init__()

        self.conv1 = Conv2D(filters=64,
                            kernel_size=3,
                            activation=tf.nn.leaky_relu)
        self.conv1T = Conv2DTranspose(64, 3, activation=tf.nn.leaky_relu)
        self.conv2 = Conv2D(64, 3, activation=tf.nn.leaky_relu)
        self.conv2T = Conv2DTranspose(64, 3, activation=tf.nn.leaky_relu)

        self.max_pool_1 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_1 = UpSampling2D(size=(2, 2))

        self.conv3 = Conv2D(128, 3, activation=tf.nn.leaky_relu)
        self.conv3T = Conv2DTranspose(128, 3, activation=tf.nn.leaky_relu)
        self.conv4 = Conv2D(128, 3, activation=tf.nn.leaky_relu)
        self.conv4T = Conv2DTranspose(128, 3, activation=tf.nn.leaky_relu)

        self.max_pool_2 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_2 = UpSampling2D(size=(2, 2))

        self.conv5 = Conv2D(256, 3, activation=tf.nn.leaky_relu)
        self.conv5T = Conv2DTranspose(256, 3, activation=tf.nn.leaky_relu)
        # self.conv6 = Conv2D(128, 3, activation=tf.nn.leaky_relu)
        # self.conv6T = Conv2DTranspose(128, 3, activation=tf.nn.leaky_relu)
        # self.max_pool_3 = MaxPool2D(pool_size=(2, 2))
        # self.up_sampling_3 = UpSampling2D(size=(2, 2))
        # self.conv7 = Conv2D(256, 3, activation=tf.nn.leaky_relu)
        # self.conv7T = Conv2DTranspose(256, 3, activation=tf.nn.leaky_relu)

        self.conv_out_mu = Conv2DTranspose(1, 3,
                                           activation=tf.nn.leaky_relu,
                                           padding='same')

        self.conv_out_sigma = Conv2DTranspose(1, 3,
                                              # activation=lambda x: tf.nn.tanh(x) + 1,
                                              activation=lambda x: tf.nn.elu(x) + 1,
                                              padding='same')

    @tf.function
    def call(self, x):
        # Convert the input data into the`NHWC` format
        #    for tensorflow, ie. channels_last
        x = tf.transpose(x, perm=[0, 2, 3, 1])

        # x = self.InputLayer
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool_1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool_2(x)
        x = self.conv5(x)
        # x = self.conv6(x)
        # x = self.max_pool_3(x)
        # x = self.conv7(x)
        # x = self.conv7T(x)
        # x = self.up_sampling_3(x)
        # x = self.conv6T(x)
        x = self.conv5T(x)
        x = self.up_sampling_2(x)
        x = self.conv4T(x)
        x = self.conv3T(x)
        x = self.up_sampling_1(x)
        x = self.conv2T(x)
        x = self.conv1T(x)
        mu = self.conv_out_mu(x)
        sigma = self.conv_out_sigma(x)

        return [mu, sigma]


class FullyConvNnArch2(FullyConvolutionalNeuralNetwork):

    """ Fully Convolutional Neural Network with 7 convolution
        and 3 MaxPool Layers"""

    def __init__(self):
        super(FullyConvNnArch2, self).__init__()

        self.conv1 = Conv2D(filters=32,
                            kernel_size=3,
                            activation=tf.nn.leaky_relu,
                            padding='same')
        self.conv1T = Conv2DTranspose(32, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv2 = Conv2D(32, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv2T = Conv2DTranspose(32, 3, activation=tf.nn.leaky_relu, padding='same')

        self.max_pool_1 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_1 = UpSampling2D(size=(2, 2))

        self.conv3 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv3T = Conv2DTranspose(64, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv4 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv4T = Conv2DTranspose(64, 3, activation=tf.nn.leaky_relu, padding='same')

        self.max_pool_2 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_2 = UpSampling2D(size=(2, 2))

        self.conv5 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv5T = Conv2DTranspose(128, 3, activation=tf.nn.leaky_relu, padding='same')

        self.conv_out_mu = Conv2DTranspose(1, 3,
                                           activation=tf.nn.leaky_relu,
                                           padding='same')

        self.conv_out_sigma = Conv2DTranspose(1, 3,
                                              activation=lambda x: tf.nn.elu(x) + 1,
                                              padding='same')

    @tf.function
    def call(self, x):
        # Convert the input data into the`NHWC` format
        #    for tensorflow, ie. channels_last
        x = tf.transpose(x, perm=[0, 2, 3, 1])

        # x = self.InputLayer
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool_1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool_2(x)
        x = self.conv5(x)
        x = self.conv5T(x)
        x = self.up_sampling_2(x)
        x = self.conv4T(x)
        x = self.conv3T(x)
        x = self.up_sampling_1(x)
        x = self.conv2T(x)
        x = self.conv1T(x)
        mu = self.conv_out_mu(x)
        sigma = self.conv_out_sigma(x)
        # x = self.flatten(x)
        # x = self.Layer_1(x)
        # x = self.Layer_2(x)
        # x = self.Layer_3(x)

        return [mu, sigma]


class FullyConvNnArch3(FullyConvolutionalNeuralNetwork):
    """ Fully Convolutional Neural Network with 7 convolution
        and 3 MaxPool Layers"""

    def __init__(self):

        super(FullyConvNnArch3, self).__init__()

        self.conv1 = Conv2D(filters=64,
                            kernel_size=3,
                            activation=tf.nn.leaky_relu,
                            padding='same')
        self.conv1T = Conv2DTranspose(64, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv2 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv2T = Conv2DTranspose(64, 3, activation=tf.nn.leaky_relu, padding='same')

        self.max_pool_1 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_1 = UpSampling2D(size=(2, 2))

        self.conv3 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv3T = Conv2DTranspose(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv4 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv4T = Conv2DTranspose(128, 3, activation=tf.nn.leaky_relu, padding='same')

        self.max_pool_2 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_2 = UpSampling2D(size=(2, 2))

        self.conv5 = Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv5T = Conv2DTranspose(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv6 = Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv6T = Conv2DTranspose(256, 3, activation=tf.nn.leaky_relu, padding='same')

        self.max_pool_3 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_3 = UpSampling2D(size=(2, 2))

        self.conv7 = Conv2D(512, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv7T = Conv2DTranspose(512, 3, activation=tf.nn.leaky_relu, padding='same')

        self.conv_out_mu = Conv2DTranspose(1, 3,
                                           activation=tf.nn.leaky_relu,
                                           padding='same')

        self.conv_out_sigma = Conv2DTranspose(1, 3,
                                              activation=lambda x: tf.nn.elu(x) + 1,
                                              padding='same')

    @tf.function
    def call(self, x):
        # Convert the input data into the`NHWC` format
        #    for tensorflow, ie. channels_last
        x = tf.transpose(x, perm=[0, 2, 3, 1])

        # x = self.InputLayer
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool_1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool_2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max_pool_3(x)
        x = self.conv7(x)
        x = self.conv7T(x)
        x = self.up_sampling_3(x)
        x = self.conv6T(x)
        x = self.conv5T(x)
        x = self.up_sampling_2(x)
        x = self.conv4T(x)
        x = self.conv3T(x)
        x = self.up_sampling_1(x)
        x = self.conv2T(x)
        x = self.conv1T(x)
        mu = self.conv_out_mu(x)
        sigma = self.conv_out_sigma(x)

        return [mu, sigma]


class FullyConvNnArch4(FullyConvolutionalNeuralNetwork):
    """ Fully Convolutional Neural Network with 10 convolution,
        10 transpose convolution
        and 3 MaxPool Layers"""

    def __init__(self):

        super(FullyConvNnArch4, self).__init__()

        self.conv1 = Conv2D(filters=64,
                            kernel_size=3,
                            activation=tf.nn.leaky_relu,
                            padding='same')
        self.conv1T = Conv2DTranspose(64, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv2 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv2T = Conv2DTranspose(64, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv3 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv3T = Conv2DTranspose(64, 3, activation=tf.nn.leaky_relu, padding='same')

        self.max_pool_1 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_1 = UpSampling2D(size=(2, 2))

        self.conv4 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv4T = Conv2DTranspose(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv5 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv5T = Conv2DTranspose(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv6 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv6T = Conv2DTranspose(128, 3, activation=tf.nn.leaky_relu, padding='same')

        self.max_pool_2 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_2 = UpSampling2D(size=(2, 2))

        self.conv7 = Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv7T = Conv2DTranspose(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv8 = Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv8T = Conv2DTranspose(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv9 = Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv9T = Conv2DTranspose(256, 3, activation=tf.nn.leaky_relu, padding='same')

        self.max_pool_3 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_3 = UpSampling2D(size=(2, 2))

        self.conv10 = Conv2D(512, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv10T = Conv2DTranspose(512, 3, activation=tf.nn.leaky_relu, padding='same')

        self.conv_out_mu = Conv2DTranspose(1, 3,
                                           activation=tf.nn.leaky_relu,
                                           padding='same')

        self.conv_out_sigma = Conv2DTranspose(1, 3,
                                              activation=lambda x: tf.nn.elu(x) + 1,
                                              padding='same')

    @tf.function
    def call(self, x):
        # Convert the input data into the`NHWC` format
        #    for tensorflow, ie. channels_last
        x = tf.transpose(x, perm=[0, 2, 3, 1])

        # x = self.InputLayer
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool_1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max_pool_2(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.max_pool_3(x)
        x = self.conv10T(x)
        x = self.up_sampling_3(x)
        x = self.conv9T(x)
        x = self.conv8T(x)
        x = self.conv7T(x)
        x = self.up_sampling_2(x)
        x = self.conv6T(x)
        x = self.conv5T(x)
        x = self.conv4T(x)
        x = self.up_sampling_1(x)
        x = self.conv3T(x)
        x = self.conv2T(x)
        x = self.conv1T(x)
        mu = self.conv_out_mu(x)
        sigma = self.conv_out_sigma(x)

        return [mu, sigma]


class FullyConvNnArch5(FullyConvolutionalNeuralNetwork):
    """ Fully Convolutional Neural Network with 10 + (3+3) =16 convolution,
        10 transpose convolution
        and 3 MaxPool Layers"""

    def __init__(self):

        super(FullyConvNnArch5, self).__init__()

        self.conv1 = Conv2D(filters=64,
                            kernel_size=3,
                            activation=tf.nn.leaky_relu,
                            padding='same')
        self.conv1T = Conv2DTranspose(64, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv2 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv2T = Conv2DTranspose(64, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv3 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv3T = Conv2DTranspose(64, 3, activation=tf.nn.leaky_relu, padding='same')

        self.max_pool_1 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_1 = UpSampling2D(size=(2, 2))

        self.conv4 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv4T = Conv2DTranspose(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv5 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv5T = Conv2DTranspose(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv6 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv6T = Conv2DTranspose(128, 3, activation=tf.nn.leaky_relu, padding='same')

        self.max_pool_2 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_2 = UpSampling2D(size=(2, 2))

        self.conv7 = Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv7T = Conv2DTranspose(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv8 = Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv8T = Conv2DTranspose(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv9 = Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv9T = Conv2DTranspose(256, 3, activation=tf.nn.leaky_relu, padding='same')

        self.max_pool_3 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_3 = UpSampling2D(size=(2, 2))

        self.conv10 = Conv2D(512, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv10T = Conv2DTranspose(512, 3, activation=tf.nn.leaky_relu, padding='same')

        self.conv11 = Conv2D(32, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv12 = Conv2D(32, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv13 = Conv2D(16, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv14 = Conv2D(16, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv15 = Conv2D(8, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv16 = Conv2D(8, 3, activation=tf.nn.leaky_relu, padding='same')

        self.conv_out_mu = Conv2DTranspose(1, 3,
                                           activation=tf.nn.leaky_relu,
                                           padding='same')

        self.conv_out_sigma = Conv2DTranspose(1, 3,
                                              activation=lambda x: tf.nn.elu(x) + 1,
                                              padding='same')

    @tf.function
    def call(self, x):
        # Convert the input data into the`NHWC` format
        #    for tensorflow, ie. channels_last
        x = tf.transpose(x, perm=[0, 2, 3, 1])

        # x = self.InputLayer
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool_1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max_pool_2(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.max_pool_3(x)
        x = self.conv10T(x)
        x = self.up_sampling_3(x)
        x = self.conv9T(x)
        x = self.conv8T(x)
        x = self.conv7T(x)
        x = self.up_sampling_2(x)
        x = self.conv6T(x)
        x = self.conv5T(x)
        x = self.conv4T(x)
        x = self.up_sampling_1(x)
        x = self.conv3T(x)
        x = self.conv2T(x)
        x = self.conv1T(x)

        mu = self.conv11(x)
        mu = self.conv13(mu)
        mu = self.conv15(mu)
        mu = self.conv_out_mu(mu)

        sigma = self.conv12(x)
        sigma = self.conv14(sigma)
        sigma = self.conv16(sigma)
        sigma = self.conv_out_sigma(sigma)

        return [mu, sigma]


class FullyConvNnArch6(FullyConvolutionalNeuralNetwork):
    """ Fully Convolutional Neural Network with 10 + 3 =13 convolution,
        10 transpose convolution
        and 3 MaxPool Layers"""

    def __init__(self):

        super(FullyConvNnArch6, self).__init__()

        self.conv1 = Conv2D(filters=64,
                            kernel_size=3,
                            activation=tf.nn.leaky_relu,
                            padding='same')
        self.conv1T = Conv2DTranspose(64, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv2 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv2T = Conv2DTranspose(64, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv3 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv3T = Conv2DTranspose(64, 3, activation=tf.nn.leaky_relu, padding='same')

        self.max_pool_1 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_1 = UpSampling2D(size=(2, 2))

        self.conv4 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv4T = Conv2DTranspose(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv5 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv5T = Conv2DTranspose(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv6 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv6T = Conv2DTranspose(128, 3, activation=tf.nn.leaky_relu, padding='same')

        self.max_pool_2 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_2 = UpSampling2D(size=(2, 2))

        self.conv7 = Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv7T = Conv2DTranspose(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv8 = Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv8T = Conv2DTranspose(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv9 = Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv9T = Conv2DTranspose(256, 3, activation=tf.nn.leaky_relu, padding='same')

        self.max_pool_3 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_3 = UpSampling2D(size=(2, 2))

        self.conv10 = Conv2D(512, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv10T = Conv2DTranspose(512, 3, activation=tf.nn.leaky_relu, padding='same')

        self.conv11 = Conv2D(32, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv12 = Conv2D(16, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv13 = Conv2D(8, 3, activation=tf.nn.leaky_relu, padding='same')

        self.conv_out_mu = Conv2DTranspose(1, 3,
                                           activation=tf.nn.leaky_relu,
                                           padding='same')

        self.conv_out_sigma = Conv2DTranspose(1, 3,
                                              activation=lambda x: tf.nn.elu(x) + 1,
                                              padding='same')

    @tf.function
    def call(self, x):
        # Convert the input data into the`NHWC` format
        #    for tensorflow, ie. channels_last
        x = tf.transpose(x, perm=[0, 2, 3, 1])

        # x = self.InputLayer
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool_1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max_pool_2(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.max_pool_3(x)
        x = self.conv10T(x)
        x = self.up_sampling_3(x)
        x = self.conv9T(x)
        x = self.conv8T(x)
        x = self.conv7T(x)
        x = self.up_sampling_2(x)
        x = self.conv6T(x)
        x = self.conv5T(x)
        x = self.conv4T(x)
        x = self.up_sampling_1(x)
        x = self.conv3T(x)
        x = self.conv2T(x)
        x = self.conv1T(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        mu = self.conv_out_mu(x)
        sigma = self.conv_out_sigma(x)

        return [mu, sigma]


class FullyConvNnArch7(FullyConvolutionalNeuralNetwork):
    """ Fully Convolutional Neural Network with 10 + 3 =13 convolution,
        10 transpose convolution
        and 3 MaxPool Layers"""

    def __init__(self):

        super(FullyConvNnArch7, self).__init__()

        self.conv1 = Conv2D(filters=128,
                            kernel_size=3,
                            activation=tf.nn.leaky_relu,
                            padding='same')
        self.conv1T = Conv2DTranspose(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv2 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv2T = Conv2DTranspose(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv3 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv3T = Conv2DTranspose(128, 3, activation=tf.nn.leaky_relu, padding='same')

        self.max_pool_1 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_1 = UpSampling2D(size=(2, 2))

        self.conv4 = Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv4T = Conv2DTranspose(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv5 = Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv5T = Conv2DTranspose(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv6 = Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv6T = Conv2DTranspose(256, 3, activation=tf.nn.leaky_relu, padding='same')

        self.max_pool_2 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_2 = UpSampling2D(size=(2, 2))

        self.conv7 = Conv2D(512, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv7T = Conv2DTranspose(512, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv8 = Conv2D(512, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv8T = Conv2DTranspose(512, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv9 = Conv2D(512, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv9T = Conv2DTranspose(256, 3, activation=tf.nn.leaky_relu, padding='same')

        self.max_pool_3 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_3 = UpSampling2D(size=(2, 2))

        self.conv10 = Conv2D(1024, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv10T = Conv2DTranspose(1024, 3, activation=tf.nn.leaky_relu, padding='same')

        self.conv11 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv12 = Conv2D(32, 3, activation=tf.nn.leaky_relu, padding='same')
        self.conv13 = Conv2D(16, 3, activation=tf.nn.leaky_relu, padding='same')

        self.conv_out_mu = Conv2DTranspose(1, 3,
                                           activation=tf.nn.leaky_relu,
                                           padding='same')

        self.conv_out_sigma = Conv2DTranspose(1, 3,
                                              activation=lambda x: tf.nn.elu(x) + 1,
                                              padding='same')

    @tf.function
    def call(self, x):
        # Convert the input data into the`NHWC` format
        #    for tensorflow, ie. channels_last
        x = tf.transpose(x, perm=[0, 2, 3, 1])

        # x = self.InputLayer
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool_1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max_pool_2(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.max_pool_3(x)
        x = self.conv10T(x)
        x = self.up_sampling_3(x)
        x = self.conv9T(x)
        x = self.conv8T(x)
        x = self.conv7T(x)
        x = self.up_sampling_2(x)
        x = self.conv6T(x)
        x = self.conv5T(x)
        x = self.conv4T(x)
        x = self.up_sampling_1(x)
        x = self.conv3T(x)
        x = self.conv2T(x)
        x = self.conv1T(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        mu = self.conv_out_mu(x)
        sigma = self.conv_out_sigma(x)

        return [mu, sigma]


class FullyConvNnArch8(FullyConvolutionalNeuralNetwork):
    """ Fully Convolutional Neural Network with total 26 layers, 10 convolution,
        10 transpose convolution, 3 max pooling, and 3 upsampling layers.
       """

    def __init__(self):

        super(FullyConvNnArch8, self).__init__()

        self.conv1 = Conv2D(filters=64,
                            kernel_size=3,
                            activation=tf.nn.leaky_relu,
                            padding='same',
                            kernel_regularizer=regularizers.l2(0.01))
        self.conv1T = Conv2DTranspose(64, 3, activation=tf.nn.leaky_relu, padding='same',
                                      kernel_regularizer=regularizers.l2(0.01))
        self.conv2 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=regularizers.l2(0.01))
        self.conv2T = Conv2DTranspose(64, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=regularizers.l2(0.01))
        self.conv3 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=regularizers.l2(0.01))
        self.conv3T = Conv2DTranspose(64, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=regularizers.l2(0.01))

        self.max_pool_1 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_1 = UpSampling2D(size=(2, 2))

        self.conv4 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=regularizers.l2(0.01))
        self.conv4T = Conv2DTranspose(128, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=regularizers.l2(0.01))
        self.conv5 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=regularizers.l2(0.01))
        self.conv5T = Conv2DTranspose(128, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=regularizers.l2(0.01))
        self.conv6 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=regularizers.l2(0.01))
        self.conv6T = Conv2DTranspose(128, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=regularizers.l2(0.01))

        self.max_pool_2 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_2 = UpSampling2D(size=(2, 2))

        self.conv7 = Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=regularizers.l2(0.01))
        self.conv7T = Conv2DTranspose(256, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=regularizers.l2(0.01))
        self.conv8 = Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=regularizers.l2(0.01))
        self.conv8T = Conv2DTranspose(256, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=regularizers.l2(0.01))
        self.conv9 = Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=regularizers.l2(0.01))
        self.conv9T = Conv2DTranspose(256, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=regularizers.l2(0.01))

        self.max_pool_3 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_3 = UpSampling2D(size=(2, 2))

        self.conv10 = Conv2D(512, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=regularizers.l2(0.01))
        self.conv10T = Conv2DTranspose(512, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=regularizers.l2(0.01))

        self.conv_out_mu = Conv2DTranspose(1, 3,
                                           activation=tf.nn.leaky_relu,
                                           padding='same',
                            kernel_regularizer=regularizers.l2(0.01))

        self.conv_out_sigma = Conv2DTranspose(1, 3,
                                              activation=lambda x: tf.nn.elu(x) + 1,
                                              padding='same',
                            kernel_regularizer=regularizers.l2(0.01))

    @tf.function
    def call(self, x):
        # Convert the input data into the`NHWC` format
        #    for tensorflow, ie. channels_last
        x = tf.transpose(x, perm=[0, 2, 3, 1])

        # x = self.InputLayer
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool_1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max_pool_2(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.max_pool_3(x)
        x = self.conv10T(x)
        x = self.up_sampling_3(x)
        x = self.conv9T(x)
        x = self.conv8T(x)
        x = self.conv7T(x)
        x = self.up_sampling_2(x)
        x = self.conv6T(x)
        x = self.conv5T(x)
        x = self.conv4T(x)
        x = self.up_sampling_1(x)
        x = self.conv3T(x)
        x = self.conv2T(x)
        x = self.conv1T(x)

        mu = self.conv_out_mu(x)
        sigma = self.conv_out_sigma(x)

        return [mu, sigma]


class FullyConvNnArch9(FullyConvolutionalNeuralNetwork):
    """ Fully Convolutional Neural Network with total 26 layers, 10 convolution,
        10 transpose convolution, 3 max pooling, and 3 upsampling layers.
       """

    def __init__(self):

        super(FullyConvNnArch9, self).__init__()
        self.regularizers = regularizers.l2(0.001)

        self.conv1 = Conv2D(filters=256,
                            kernel_size=4,
                            activation=tf.nn.leaky_relu,
                            padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                      kernel_regularizer=self.regularizers)
        self.conv2 = Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv2T = Conv2DTranspose(64, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv3 = Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv3T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)

        self.max_pool_1 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_1 = UpSampling2D(size=(2, 2))

        self.conv4 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv4T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv5 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv5T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv6 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv6T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)

        self.max_pool_2 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_2 = UpSampling2D(size=(2, 2))

        self.conv7 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv7T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv8 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv8T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv9 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv9T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)

        self.max_pool_3 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_3 = UpSampling2D(size=(2, 2))

        self.conv10 = Conv2D(2024, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv10T = Conv2DTranspose(2024, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)

        self.conv_out_mu = Conv2DTranspose(1, 4,
                                           activation=tf.nn.leaky_relu,
                                           padding='same',
                            kernel_regularizer=self.regularizers)

        self.conv_out_sigma = Conv2DTranspose(1, 4,
                                              activation=lambda x: tf.nn.elu(x) + 1,
                                              padding='same',
                            kernel_regularizer=self.regularizers)

    @tf.function
    def call(self, x):
        # Convert the input data into the`NHWC` format
        #    for tensorflow, ie. channels_last
        x = tf.transpose(x, perm=[0, 2, 3, 1])

        # x = self.InputLayer
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool_1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max_pool_2(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.max_pool_3(x)
        x = self.conv10T(x)
        x = self.up_sampling_3(x)
        x = self.conv9T(x)
        x = self.conv8T(x)
        x = self.conv7T(x)
        x = self.up_sampling_2(x)
        x = self.conv6T(x)
        x = self.conv5T(x)
        x = self.conv4T(x)
        x = self.up_sampling_1(x)
        x = self.conv3T(x)
        x = self.conv2T(x)
        x = self.conv1T(x)

        mu = self.conv_out_mu(x)
        sigma = self.conv_out_sigma(x)

        return [mu, sigma]


class FullyConvNnArch10(FullyConvolutionalNeuralNetwork):
    """ Fully Convolutional Neural Network with total 26 layers, 10 convolution,
        10 transpose convolution, 3 max pooling, and 3 upsampling layers.
       """

    def __init__(self):

        super(FullyConvNnArch10, self).__init__()
        self.regularizers = regularizers.l2(0.001)

        self.conv1 = Conv2D(filters=128,
                            kernel_size=4,
                            activation=tf.nn.leaky_relu,
                            padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                      kernel_regularizer=self.regularizers)
        self.conv2 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv2T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv3 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv3T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)

        self.max_pool_1 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_1 = UpSampling2D(size=(2, 2))

        self.conv4 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv4T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv5 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv5T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv6 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv6T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)

        self.max_pool_2 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_2 = UpSampling2D(size=(2, 2))

        self.conv7 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv7T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv8 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv8T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv9 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv9T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)

        self.max_pool_3 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_3 = UpSampling2D(size=(2, 2))

        self.conv10 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv10T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)

        ###### different branch for other output after maxpool 2


        self.conv1_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                      kernel_regularizer=self.regularizers)

        self.conv2_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)

        self.conv3_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)

        self.up_sampling_1_1 = UpSampling2D(size=(2, 2))

        self.conv4_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv5_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv6_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.up_sampling_2_1 = UpSampling2D(size=(2, 2))


        self.conv7_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv7_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                      kernel_regularizer=self.regularizers)
        self.conv8_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv8_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                      kernel_regularizer=self.regularizers)
        self.conv9_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv9_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                      kernel_regularizer=self.regularizers)

        self.max_pool_3_1 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_3_1 = UpSampling2D(size=(2, 2))

        self.conv10_1 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                             kernel_regularizer=self.regularizers)
        self.conv10_1T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                       kernel_regularizer=self.regularizers)

        #### output layer
        self.conv_out_mu = Conv2DTranspose(1, 4,
                                           activation=tf.nn.leaky_relu,
                                           padding='same',
                            kernel_regularizer=self.regularizers)

        self.conv_out_sigma = Conv2DTranspose(1, 4,
                                              activation=lambda x: tf.nn.elu(x) + 1,
                                              padding='same',
                            kernel_regularizer=self.regularizers)

    @tf.function
    def call(self, x):
        # Convert the input data into the`NHWC` format
        #    for tensorflow, ie. channels_last
        x = tf.transpose(x, perm=[0, 2, 3, 1])

        # x = self.InputLayer
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool_1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max_pool_2(x)

        ## separate branch from this point
        y = self.conv7_1(x)
        x = self.conv7(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.max_pool_3(x)
        x = self.conv10T(x)
        x = self.up_sampling_3(x)
        x = self.conv9T(x)
        x = self.conv8T(x)
        x = self.conv7T(x)
        x = self.up_sampling_2(x)
        x = self.conv6T(x)
        x = self.conv5T(x)
        x = self.conv4T(x)
        x = self.up_sampling_1(x)
        x = self.conv3T(x)
        x = self.conv2T(x)
        x = self.conv1T(x)


        y = self.conv8_1(y)
        y = self.conv9_1(y)
        y = self.conv10_1(y)
        y = self.max_pool_3_1(y)
        y = self.conv10_1T(y)
        y = self.up_sampling_3_1(y)
        y = self.conv9_1T(y)
        y = self.conv8_1T(y)
        y = self.conv7_1T(y)
        y = self.up_sampling_2_1(y)
        y = self.conv6_1T(y)
        y = self.conv5_1T(y)
        y = self.conv4_1T(y)
        y = self.up_sampling_1_1(y)
        y = self.conv3_1T(y)
        y = self.conv2_1T(y)
        y = self.conv1_1T(y)

        mu = self.conv_out_mu(x)
        sigma = self.conv_out_sigma(y)

        return [mu, sigma]


class FullyConvNnArch11(FullyConvolutionalNeuralNetwork):
    """ Fully Convolutional Neural Network with total 26 layers, 10 convolution,
        10 transpose convolution, 3 max pooling, and 3 upsampling layers.
    """

    def __init__(self):

        super(FullyConvNnArch11, self).__init__()
        self.regularizers = regularizers.l2(0.001)

        self.conv1 = Conv2D(filters=128,
                            kernel_size=4,
                            activation=tf.nn.leaky_relu,
                            padding='same',
                            kernel_regularizer=self.regularizers,
                            trainable=self._b_trainable_combined_layers)
        self.conv1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                      kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        self.conv2 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_combined_layers)
        self.conv2T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        self.conv3 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_combined_layers)
        self.conv3T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)

        self.max_pool_1 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_1 = UpSampling2D(size=(2, 2))

        self.conv4 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_combined_layers)
        self.conv4T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        self.conv5 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_combined_layers)
        self.conv5T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        self.conv6 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_combined_layers)
        self.conv6T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)

        self.max_pool_2 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_2 = UpSampling2D(size=(2, 2))

        self.conv7 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        self.conv7T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        self.conv8 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        self.conv8T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        self.conv9 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        self.conv9T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)

        self.max_pool_3 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_3 = UpSampling2D(size=(2, 2))

        self.conv10 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        self.conv10T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)

        ###### different branch for other output after maxpool 2


        self.conv1_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                      kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)

        self.conv2_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)

        self.conv3_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)

        self.up_sampling_1_1 = UpSampling2D(size=(2, 2))

        self.conv4_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        self.conv5_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        self.conv6_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        self.up_sampling_2_1 = UpSampling2D(size=(2, 2))


        self.conv7_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        self.conv7_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                      kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        self.conv8_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        self.conv8_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                      kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        self.conv9_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        self.conv9_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                      kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)

        self.max_pool_3_1 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_3_1 = UpSampling2D(size=(2, 2))

        self.conv10_1 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                             kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        self.conv10_1T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                       kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)

        #### output layer
        self.conv_out_mu = Conv2DTranspose(1, 4,
                                           activation=tf.nn.leaky_relu,
                                           padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)

        self.conv_out_sigma = Conv2DTranspose(1, 4,
                                              activation=lambda x: tf.nn.elu(x) + 1,
                                              padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)

    @tf.function
    def call(self, x):
        # Convert the input data into the`NHWC` format
        #    for tensorflow, ie. channels_last
        x = tf.transpose(x, perm=[0, 2, 3, 1])

        # x = self.InputLayer
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool_1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max_pool_2(x)

        ## separate branch from this point
        y = self.conv7_1(x)
        x = self.conv7(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.max_pool_3(x)
        x = self.conv10T(x)
        x = self.up_sampling_3(x)
        x = self.conv9T(x)
        x = self.conv8T(x)
        x = self.conv7T(x)
        x = self.up_sampling_2(x)
        x = self.conv6T(x)
        x = self.conv5T(x)
        x = self.conv4T(x)
        x = self.up_sampling_1(x)
        x = self.conv3T(x)
        x = self.conv2T(x)
        x = self.conv1T(x)


        y = self.conv8_1(y)
        y = self.conv9_1(y)
        y = self.conv10_1(y)
        y = self.max_pool_3_1(y)
        y = self.conv10_1T(y)
        y = self.up_sampling_3_1(y)
        y = self.conv9_1T(y)
        y = self.conv8_1T(y)
        y = self.conv7_1T(y)
        y = self.up_sampling_2_1(y)
        y = self.conv6_1T(y)
        y = self.conv5_1T(y)
        y = self.conv4_1T(y)
        y = self.up_sampling_1_1(y)
        y = self.conv3_1T(y)
        y = self.conv2_1T(y)
        y = self.conv1_1T(y)

        mu = self.conv_out_mu(x)
        sigma = self.conv_out_sigma(y)

        return [mu, sigma]


class FullyConvNnArch12(FullyConvolutionalNeuralNetwork):
    """ Fully Convolutional Neural Network with total 26 layers, 10 convolution,
        10 transpose convolution, 3 max pooling, and 3 upsampling layers.
    """

    def __init__(self):

        super(FullyConvNnArch12, self).__init__()
        self.regularizers = regularizers.l2(0.001)

        self.block1 = block1()
        self.block1.kernel_regularizer = self.regularizers
        # self.conv1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
        #                               kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        self.conv2 = Conv2D(1, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_combined_layers)
        # self.conv2T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        self.conv3 = Conv2D(1, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_combined_layers)
        # self.conv3T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        #
        # self.max_pool_1 = MaxPool2D(pool_size=(2, 2))
        # self.up_sampling_1 = UpSampling2D(size=(2, 2))
        #
        # self.conv4 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_combined_layers)
        # self.conv4T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        # self.conv5 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_combined_layers)
        # self.conv5T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        # self.conv6 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_combined_layers)
        # self.conv6T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        #
        # self.max_pool_2 = MaxPool2D(pool_size=(2, 2))
        # self.up_sampling_2 = UpSampling2D(size=(2, 2))
        #
        # self.conv7 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        # self.conv7T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        # self.conv8 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        # self.conv8T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        # self.conv9 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        # self.conv9T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        #
        # self.max_pool_3 = MaxPool2D(pool_size=(2, 2))
        # self.up_sampling_3 = UpSampling2D(size=(2, 2))
        #
        # self.conv10 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        # self.conv10T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)
        #
        # ###### different branch for other output after maxpool 2
        #
        #
        # self.conv1_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
        #                               kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        #
        # self.conv2_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        #
        # self.conv3_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        #
        # self.up_sampling_1_1 = UpSampling2D(size=(2, 2))
        #
        # self.conv4_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        # self.conv5_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        # self.conv6_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        # self.up_sampling_2_1 = UpSampling2D(size=(2, 2))
        #
        #
        # self.conv7_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        # self.conv7_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
        #                               kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        # self.conv8_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        # self.conv8_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
        #                               kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        # self.conv9_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
        #                     kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        # self.conv9_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
        #                               kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        #
        # self.max_pool_3_1 = MaxPool2D(pool_size=(2, 2))
        # self.up_sampling_3_1 = UpSampling2D(size=(2, 2))
        #
        # self.conv10_1 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
        #                      kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        # self.conv10_1T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
        #                                kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)

        #### output layer
        self.conv_out_mu = Conv2DTranspose(1, 4,
                                           activation=tf.nn.leaky_relu,
                                           padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_mean_layers)

        self.conv_out_sigma = Conv2DTranspose(1, 4,
                                              activation=lambda x: tf.nn.elu(x) + 1,
                                              padding='same',
                            kernel_regularizer=self.regularizers, trainable=self._b_trainable_std_layers)
        print("INtialized...............")
    @tf.function
    def call(self, x):
        # Convert the input data into the`NHWC` format
        #    for tensorflow, ie. channels_last
        x = tf.transpose(x, perm=[0, 2, 3, 1])

        # x = self.InputLayer
        print(self._b_trainable_combined_layers)
        x = self.block1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        mu = self.conv_out_mu(x)
        sigma = self.conv_out_sigma(x)

        return [mu, sigma]
    #
    # def block1(self):
    #     self.conv1 = Conv2D(filters=1,
    #                         kernel_size=4,
    #                         activation=tf.nn.leaky_relu,
    #                         padding='same',
    #                         kernel_regularizer=self.regularizers,
    #                         trainable=self._b_trainable_combined_layers)


class FullyConvNnArch13(FullyConvolutionalNeuralNetwork):
    """ Fully Convolutional Neural Network with total 26 layers, 10 convolution,
        10 transpose convolution, 3 max pooling, and 3 upsampling layers.
    """

    def __init__(self):
        super(FullyConvNnArch13, self).__init__()
        self.regularizers = regularizers.l2(0.001)
        self.combined_layers = CombinedLayers()
        self.std_deviation_layers = StdDeviationlayers()
        self.mean_layers = MeanLayers()

    @tf.function
    def call(self, x):
        # Convert the input data into the`NHWC` format
        #    for tensorflow, ie. channels_last
        x = tf.transpose(x, perm=[0, 2, 3, 1])

        # x = self.InputLayer
        x = self.combined_layers(x)
        mu = self.mean_layers(x)
        sigma = self.std_deviation_layers(x)

        return [mu, sigma]


class FullyConvNnArch15(FullyConvolutionalNeuralNetwork):
    """ Fully Convolutional Neural Network with total 26 layers, 10 convolution,
        10 transpose convolution, 3 max pooling, and 3 upsampling layers.
    """

    def __init__(self):

        super(FullyConvNnArch15, self).__init__()
        self.regularizers = regularizers.l2(0.001)
        self.combined_layers = self.CombinedLayers()
        self.mean_layers = self.MeanLayers()
        self.std_deviation_layers = self.StdDeviationlayers()

    class CombinedLayers(Model):
        def __init__(self):
            super().__init__()
            self.regularizers = regularizers.l2(0.001)
            self.conv1 = Conv2D(filters=128,
                                kernel_size=4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv2 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv3 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.max_pool_1 = MaxPool2D(pool_size=(2, 2))


            self.conv4 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv5 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv6 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.max_pool_2 = MaxPool2D(pool_size=(2, 2))

        @tf.function
        def call(self, x):
            # x = self.InputLayer
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.max_pool_1(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.max_pool_2(x)
            return x

    class MeanLayers(Model):
        def __init__(self):
            super().__init__()
            self.regularizers = regularizers.l2(0.001)
            self.conv1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.conv2T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.conv3T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.up_sampling_1 = UpSampling2D(size=(2, 2))

            self.conv4T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv5T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv6T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.up_sampling_2 = UpSampling2D(size=(2, 2))

            self.conv7 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv7T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv8 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv8T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv9 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv9T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.max_pool_3 = MaxPool2D(pool_size=(2, 2))
            self.up_sampling_3 = UpSampling2D(size=(2, 2))

            self.conv10 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                 kernel_regularizer=self.regularizers)
            self.conv10T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                           kernel_regularizer=self.regularizers)

            self.conv_out_mu = Conv2DTranspose(1, 4,
                                               activation=tf.nn.leaky_relu,
                                               padding='same',
                                               kernel_regularizer=self.regularizers)

        @tf.function
        def call(self, x):
            ## separate branch from this point

            x = self.conv7(x)
            x = self.conv8(x)
            x = self.conv9(x)
            x = self.max_pool_3(x)
            x = self.conv10(x)
            x = self.conv10T(x)
            x = self.up_sampling_3(x)
            x = self.conv9T(x)
            x = self.conv8T(x)
            x = self.conv7T(x)
            x = self.up_sampling_2(x)
            x = self.conv6T(x)
            x = self.conv5T(x)
            x = self.conv4T(x)
            x = self.up_sampling_1(x)
            x = self.conv3T(x)
            x = self.conv2T(x)
            x = self.conv1T(x)

            mu = self.conv_out_mu(x)

            return mu

    class StdDeviationlayers(Model):
        def __init__(self):
            super().__init__()
            self.regularizers = regularizers.l2(0.001)
            self.conv1_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.conv2_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv3_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.up_sampling_1_1 = UpSampling2D(size=(2, 2))

            self.conv4_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv5_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv6_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.up_sampling_2_1 = UpSampling2D(size=(2, 2))


            self.conv7_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv7_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv8_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv8_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv9_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv9_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.max_pool_3_1 = MaxPool2D(pool_size=(2, 2))
            self.up_sampling_3_1 = UpSampling2D(size=(2, 2))

            self.conv10_1 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                 kernel_regularizer=self.regularizers)
            self.conv10_1T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                           kernel_regularizer=self.regularizers)

            self.conv_out_sigma = Conv2DTranspose(1, 4,
                                                  activation=lambda x: tf.nn.elu(x) + 1,
                                                  padding='same',
                                                  kernel_regularizer=self.regularizers)

        @tf.function
        def call(self, y):
            ## separate branch from this point
            y = self.conv7_1(y)
            y = self.conv8_1(y)
            y = self.conv9_1(y)
            y = self.max_pool_3_1(y)
            y = self.conv10_1(y)
            y = self.conv10_1T(y)
            y = self.up_sampling_3_1(y)
            y = self.conv9_1T(y)
            y = self.conv8_1T(y)
            y = self.conv7_1T(y)
            y = self.up_sampling_2_1(y)
            y = self.conv6_1T(y)
            y = self.conv5_1T(y)
            y = self.conv4_1T(y)
            y = self.up_sampling_1_1(y)
            y = self.conv3_1T(y)
            y = self.conv2_1T(y)
            y = self.conv1_1T(y)

            sigma = self.conv_out_sigma(y)

            return sigma

    @tf.function
    def call(self, x):
        # Convert the input data into the`NHWC` format
        #    for tensorflow, ie. channels_last
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        x = self.combined_layers(x)
        sigma = self.std_deviation_layers(x)
        mu = self.mean_layers(x)


        return [mu, sigma]


# class ForkConvNn(Model):
#     def __init__(self,
#                  sample_scaling=0.5,
#                  meas_output_same=False):
#         super().__init__()
#         self.sample_scaling = sample_scaling
#         self.meas_output_same = meas_output_same
#
#     # a factory method that would instantiate the class given its name:
#     @staticmethod
#     def get_fork_conv_nn_arch(nn_arch_id):
#         return globals()[f'ForkConvNnArch' + nn_arch_id]()
#
#     @tf.function
#     def call(self, x):
#         # Convert the input data into the`NHWC` format
#         #    for tensorflow, ie. channels_last
#         x = tf.transpose(x, perm=[0, 2, 3, 1])
#         sampled_map = x[..., 0][..., None]
#         mask = x[..., 1][..., None]
#         x = self.combined_layers(x)
#         mu = self.mean_layers(x)
#
#         # mu = tf.where(mask == 1, sampled_map, mu)
#         sigma = self.std_deviation_layers(x)
#
#         # z = tf.concat((mu, sigma), axis=3)
#         z = tf.concat((mu, sigma), axis=-1)
#
#         output = tf.transpose(z, perm=[0, 3, 1, 2])
#
#         return output # tf.transpose(z, perm=[0, 3, 1, 2])
#
#     # @tf.function
#     # def call(self, x):
#     #     """
#     #     Args:
#     #         `x`: 'NCHW' format batch tf dataset.
#     #     Returns:
#     #         `output`: a tensor of format NHWC (C=3) that is formed
#     #             by concatenation of mu, sigma and m_sample_scaling
#     #             along last dimension.
#     #     """
#     #     # Convert the input data into the`NHWC` format
#     #     #    for tensorflow, ie. channels_last
#     #     x = tf.transpose(x, perm=[0, 2, 3, 1])
#     #
#     #     comb_layer_output = self.combined_layers(x)
#     #     mu = self.mean_layers(comb_layer_output)
#     #     sigma = self.std_deviation_layers(comb_layer_output)
#     #     # z = tf.concat((mu, sigma), axis=3)
#     #
#     #     sampled_map = x[..., 0][..., None]
#     #     mask = x[..., 1][..., None]
#     #     if self.meas_output_same:
#     #         # Keep the estimated to be the same as measured value at
#     #         # observed locations.
#     #         # mu = tf.where(mask == 1, sampled_map, mu)
#     #         mu[mask == 1] = sampled_map[mask == 1]
#     #
#     #     m_sample_scaling = tf.where(mask == 1, self.sample_scaling, 1 - self.sample_scaling)
#     #     m_sample_scaling = tf.where(mask == 1, self.sample_scaling, 1 - self.sample_scaling)
#     #     print("shape mask", mask.shape)
#     #     # m_sample_scaling = mask
#     #     # m_sample_scaling[mask == 1] = self.sample_scaling   #observed locations
#     #     # m_sample_scaling[mask == 0] = 1 - self.sample_scaling   # unobserved locations
#     #     # m_sample_scaling[mask == -1] = 0        # building locations
#     #
#     #     mean_n_sigma = tf.concat((mu, sigma), axis=-1)
#     #     mean_sigma_n_sample_scaling = tf.concat((mean_n_sigma, m_sample_scaling), axis=-1)
#     #
#     #     output = tf.transpose(mean_sigma_n_sample_scaling, perm=[0, 3, 1, 2])
#     #     # print("output shape ", output.shape)
#     #     return output # tf.transpose(z, perm=[0, 3, 1, 2])
#
#
#     def fit(self, dataset=None, l_alpha=[], l_epochs=[],
#             test_changes=False, loss_metric=None,
#             l_learning_rate=[], batch_size=None, validation_data=None,
#             callbacks=[],
#             nn_arch_id='1',
#             save_weight_in_callback='True'):
#         """
#         Returns:
#             dic_history: a dictionary that contians
#                     "train_mean_rmse_loss": a list of length len(l_apha)x epochs that contains
#                     a concatenated train loss for all entries of l_alpha.
#         """
#
#         dict_history ={"train_mean_rmse_loss": [],
#                        "train_sigma_rmse_error": [],
#                        "val_mean_rmse_loss": [],
#                        "val_sigma_rmse_error": [],
#                        "alpha_vector": [],
#                        "train_loss": [],
#                        "val_loss": [],
#                        }
#         assert l_learning_rate != []
#
#         def loss_function_sigma(y_true, y_predict):
#             """
#             Args:
#                 if self.b_posterior_target:
#                     -`y_true`: is N x H x W x C tensor with channels C=2, where first channel
#                         contains a posterior mean target and the other contains
#                         a posterior standard deviation.
#                 else:
#                     -`y_true`: is N x H x W x C tensor tha contains C= num_sources true maps as a target.
#                         if channels are combined then C = 1.
#
#                 -`y_predict`: is N x H x W x 1 tensorflow normal distribution whose 'loc' is the estimated
#                         posterior mean and 'scale' is the estimated  standard
#
#             """
#             # y_true = tf.transpose(y_true, perm=[0, 2, 3, 1])
#             t_delta = tf.abs(y_true[:, 0, ...] - y_predict[:, 0, ...])
#             loss_sigma = tf.square(t_delta - y_predict[:, 1, ...])
#             # loss_sigma = tf.reduce_mean(tf.square(tf.sqrt(t_delta) - y_predict[:, 1, ...]))
#             # loss_sigma = tf.square(t_delta - y_predict[:, 1, ...])
#
#             return loss_sigma
#
#         def loss_function_mean(y_true, y_predict):
#             """
#             Args:
#                     -`y_true`: is N x C x H x W  tensor with channels C=1, where first channel
#                         contains a posterior mean target and the other contains
#                         a posterior standard deviation.
#
#             """
#             # y_true = tf.transpose(y_true, perm=[0, 2, 3, 1])
#             t_delta = tf.square(y_true[:, 0, ...] - y_predict[:, 0, ...])
#             loss_mean = t_delta # rmse
#
#             # loss_mean_1 = tf.keras.losses.MSE(y_true[:, 0, ...], y_predict[:, 0, ...])
#             # print(tf.executing_eagerly())
#             # print("inside loss", loss_mean, loss_mean_1)
#             # set_trace()
#             # loss_mean = tf.keras.losses.MSE(y_true[:, 0, ...], y_predict[:, 0, ...])
#             return loss_mean
#
#         def loss_function_keras(y_true, y_predict):
#             loss_mean_keras = tf.keras.losses.MSE(y_true[:, 0, ...], y_predict[:, 0, ...])
#             return loss_mean_keras
#
#         for alpha, epochs, learning_rate in zip(l_alpha, l_epochs, l_learning_rate):
#
#             # Loss function
#             def loss_function(y_true, y_predict):
#                 """
#                 Args:
#                     if self.b_posterior_target:
#                         -`y_true`: is N x H x W x C tensor with channels C=2, where first channel
#                             contains a posterior mean target and the other contains
#                             a posterior standard deviation.
#                     else:
#                         -`y_true`: is N x H x W x C tensor tha contains C= num_sources true maps as a target.
#                             if channels are combined then C = 1.
#
#                     -`y_predict`: is N x H x W x 1 tensorflow normal distribution whose 'loc' is the estimated
#                             posterior mean and 'scale' is the estimated  standard
#
#                 """
#                 # print("Y_true", y_true.shape)
#                 # y_true = tf.transpose(y_true, perm=[0, 2, 3, 1])
#                 # print("Y-predict shape", y_predict.shape)
#                 loss_mean = loss_function_mean(y_true, y_predict)
#                 loss_sigma = loss_function_sigma(y_true, y_predict)
#                 loss = alpha * loss_sigma + (1 - alpha) * loss_mean
#
#                 return loss
#
#             # loss metric
#             if loss_metric == 'Custom':
#                 assert alpha is not None
#                 if alpha == 0:
#                     # freeze combined lsayers and std. deviation layers
#                     self.combined_layers.trainable = False
#                     self.std_deviation_layers.trainable = False
#                     self.mean_layers.trainable = True
#
#                 elif alpha == 1:
#                     # freeze combined layers and mean layers
#                     self.combined_layers.trainable = False
#                     self.std_deviation_layers.trainable = True
#                     self.mean_layers.trainable = False
#
#                 elif alpha == 0.00001:
#                     # freeze std. deviation layer only.
#                     self.combined_layers.trainable = True
#                     self.std_deviation_layers.trainable = False
#                     self.mean_layers.trainable = True
#
#                 else:
#                     self.combined_layers.trainable = True
#                     self.std_deviation_layers.trainable = True
#                     self.mean_layers.trainable = True
#
#             print("learning rate ", learning_rate)
#
#             weightfolder = "./saved_weights/" + \
#                            f"ckpt_nn_arch_{nn_arch_id}_alpha={alpha}_lr={learning_rate}_epochs={epochs}.ckpt"
#             cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weightfolder,
#                                                              save_weights_only=save_weight_in_callback,
#                                                              verbose=0,
#                                                              # mode='min',
#                                                              # save_best_only=True
#                                                              )
#             tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", update_freq=1,
#                                                                   histogram_freq=1,
#                                                                   )
#
#             optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#
#             self.compile(optimizer=optimizer, loss=loss_function,
#                          # loss_weights=[alpha, 1 - alpha],
#                          metrics=[loss_function,
#                                   loss_function_mean,
#                                   loss_function_sigma,
#                                   # loss_function_keras,
#                                   ],
#                          )
#             # self.compile(optimizer=optimizer, loss=[loss_function_sigma,
#             #                                         loss_function_mean],
#             #              loss_weights=[alpha, 1 - alpha],
#             #              # metrics=[[tf.keras.metrics.MSE],
#             #              #          [tf.keras.metrics.MSE]]
#             #              )
#             # # self.compile(optimizer=optimizer, loss={"sigma_output": loss_function_sigma,
#             #                                         "mean_output": loss_function_mean},
#             #              loss_weights={"sigma_output": alpha, "mean_output": 1-alpha}, metrics={"sigma_output": [tf.keras.metrics.MSE],
#             #                                                      "mean_output": [tf.keras.metrics.MSE]}
#             #              )
#
#             if test_changes:
#                 l_initial_weight = [self.combined_layers.get_weights(),
#                                     self.mean_layers.get_weights(),
#                                     self.std_deviation_layers.get_weights()]
#
#             history = super().fit(x=dataset, epochs=epochs,
#                                   validation_data=validation_data,
#                                   callbacks=[cp_callback, tensorboard_callback], verbose=1)
#
#             vector_alpha = alpha * np.ones(shape=(epochs,))
#             vector_alpha.tolist()
#
#             dict_history[f"train_mean_rmse_loss"] += history.history["loss_function_mean"]
#             dict_history[f"train_sigma_rmse_error"] += history.history["loss_function_sigma"]
#             dict_history[f"val_mean_rmse_loss"] += history.history["val_loss_function_mean"]
#             dict_history[f"val_sigma_rmse_error"] += history.history["val_loss_function_sigma"]
#             dict_history[f"train_loss"] += history.history["loss"]
#             dict_history[f"val_loss"] += history.history["val_loss"]
#             dict_history[f"alpha_vector"]+= vector_alpha.tolist()
#
#             if test_changes:
#                 l_final_weight = [self.combined_layers.get_weights(),
#                                   self.mean_layers.get_weights(),
#                                   self.std_deviation_layers.get_weights()]
#
#                 print("Weight values of Combined layer are the same? ",
#                       list_are_close(l_initial_weight[0], l_final_weight[0]))
#                 print("Weight values of mean layer are the same? ",
#                       list_are_close(l_initial_weight[1], l_final_weight[1]))
#                 print("Weight values of std. layer are the same? ",
#                       list_are_close(l_initial_weight[2], l_final_weight[2]))
#
#             if save_weight_in_callback:
#                 # Save loss metrics for each value alpha
#                 loss_folder = "./train_data/" + \
#                               f"loss_nn_architecture_{nn_arch_id}_alpha={alpha}_lr={learning_rate}_epochs={epochs}"
#                 outfile = open(loss_folder, 'wb')
#                 pickle.dump(dict_history, outfile)
#                 outfile.close()
#
#         return dict_history
class ForkConvNn(Model):
    def __init__(self,
                 sample_scaling=0.5,
                 meas_output_same=False):
        super().__init__()

        self.sample_scaling = sample_scaling
        self.meas_output_same = meas_output_same

    # a factory method that would instantiate the class given its name:
    @staticmethod
    def get_fork_conv_nn_arch(nn_arch_id):
        return globals()[f'ForkConvNnArch' + nn_arch_id]()

    @tf.function
    def call(self, x):
        """
        Args:
            `x`: 'NCHW' format batch tf dataset.
        Returns:
            `output`: a tensor of format NHWC (C=3) that is formed
                by concatenation of mu, sigma and m_sample_scaling
                along last dimension.
        """
        # Convert the input data into the`NHWC` format
        #    for tensorflow, ie. channels_last
        x = tf.transpose(x, perm=[0, 2, 3, 1])

        comb_layer_output = self.combined_layers(x)
        mu = self.mean_layers(comb_layer_output)
        sigma = self.std_deviation_layers(comb_layer_output)
        # z = tf.concat((mu, sigma), axis=3)

        sampled_map = x[..., 0][..., None]
        mask = x[..., 1][..., None]
        if self.meas_output_same:
            # Keep the estimated to be the same as measured value at
            # observed locations.
            mu = tf.where(mask == 1, sampled_map, mu)
            # mu[mask == 1] = sampled_map[mask == 1]

        m_sample_scaling = tf.where(mask == 1, self.sample_scaling, 1 - self.sample_scaling)
        m_sample_scaling = tf.where(mask == -1, 0.0, m_sample_scaling)

        mean_n_sigma = tf.concat((mu, sigma), axis=-1)
        mean_sigma_n_sample_scaling = tf.concat((mean_n_sigma, m_sample_scaling), axis=-1)

        # in_sample_scaling = tf.where(mask == 1, 1.0, 0.0) # to calculate loss at sample locations
        # mean_sigma_n_sample_scaling = tf.concat((mean_sigma_n_sample_scaling, in_sample_scaling), axis=-1)
        #
        # out_sample_scaling = tf.where(mask == 0, 1.0, 0.0)  # to calculate loss at unobserved locations
        # mean_sigma_n_sample_scaling = tf.concat((mean_sigma_n_sample_scaling, out_sample_scaling), axis=-1)

        mean_sigma_n_sample_scaling = tf.concat((mean_sigma_n_sample_scaling, mask), axis=-1)

        output = tf.transpose(mean_sigma_n_sample_scaling, perm=[0, 3, 1, 2])
        # print("output shape ", output.shape)
        return output # tf.transpose(z, perm=[0, 3, 1, 2])


    def fit(self, train_dataset=None, l_alpha=[],
            l_epochs=[], test_changes=False, loss_metric=None,
            l_learning_rate=[],
            batch_size=None, validation_dataset=None,
            callbacks=[], nn_arch_id='1',
            save_weight_in_callback='True',
            tensorboard_callback=[],
            weightfolder="./saved_weights/",
            loss_folder="./train_data/",
            b_evaluate=False,
            test_dataset=None):
        """
        Returns:
            dic_history: a dictionary that contians
                    "train_mean_rmse_loss": a list of length len(l_apha)x epochs that contains
                    a concatenated train loss for all entries of l_alpha.
        """

        dict_history ={"train_mean_rmse_loss": [],
                       "train_sigma_rmse_error": [],
                       "val_mean_rmse_loss": [],
                       "val_sigma_rmse_error": [],
                       "alpha_vector": [],
                       "train_loss": [],
                       "val_loss": [],
                       }
        assert l_learning_rate != []

        def loss_function_sigma(y_true, y_predict):
            """
            Args:
                -`y_true`: is N x C x H x W tensor tha contains C= num_sources true maps as a target.
                        if channels are combined then C = 1.

                -`y_predict`: is N x 4 x H x W tensor where
                        y_predict[:,0,...] is an estimated mean power,
                        y_predict[:,1,...] is an estimated std. deviation,
                        y_predict[:,2,...] is a sample_scaling tensor
                        y_predict[:,3,...] is a mask

            """
            # y_true = tf.transpose(y_true, perm=[0, 2, 3, 1])
            # t_delta = tf.abs(y_true[:, 0, ...] - y_predict[:, 0, ...])
            # loss_sigma = tf.square(t_delta - y_predict[:, 1, ...])
            m_sample_scaling = y_predict[:, 2, ...]
            m_mean = y_predict[:, 0, ...]
            m_std = y_predict[:, 1, ...]
            t_delta = tf.abs(y_true[:, 0, ...] - m_mean) #* m_sample_scaling
            loss_sigma = tf.square(t_delta - m_std) * m_sample_scaling
            # loss_sigma = tf.reduce_mean(tf.square(tf.sqrt(t_delta) - y_predict[:, 1, ...]))
            # loss_sigma = tf.square(t_delta - y_predict[:, 1, ...])
            # reduce sum except for the batch dimension
            loss_sigma = tf.reduce_sum(loss_sigma, axis=-1)
            loss_sigma = tf.reduce_sum(loss_sigma, axis=-1)
            # get normalization factor in_sample and out_sample only.
            # It is equal to scaling factor * no. of meas loc +
            # (1-scaling factor) * no. of loc outside buildings
            sum_m_sample_scaling = tf.reduce_sum(m_sample_scaling, axis=-1)
            sum_m_sample_scaling = tf.reduce_sum(sum_m_sample_scaling, axis=-1)
            loss_sigma = tf.divide(loss_sigma,
                                   tf.math.maximum(sum_m_sample_scaling, tf.constant([1.0])))
            # print(tf.math.maximum(sum_m_sample_scaling, tf.constant([1.0])))

            return loss_sigma

        def loss_function_mean(y_true, y_predict):
            """
            Args:
                -`y_true`: is N x C x H x W tensor tha contains C= num_sources true maps as a target.
                        if channels are combined then C = 1.

                -`y_predict`: is N x 4 x H x W tensor where
                        y_predict[:,0,...] is an estimated mean power,
                        y_predict[:,1,...] is an estimated std. deviation,
                        y_predict[:,2,...] is a sample_scaling tensor
                        y_predict[:,3,...] is a mask

            """
            # y_true = tf.transpose(y_true, perm=[0, 2, 3, 1])
            # t_delta = tf.square(y_true[:, 0, ...] - y_predict[:, 0, ...])
            m_sample_scaling = y_predict[:, 2, ...]
            m_mean = y_predict[:, 0, ...]
            t_delta = tf.square(y_true[:, 0, ...] - m_mean) * m_sample_scaling
            loss_mean = t_delta # rmse

            # reduce sum except for the batch dimension
            loss_mean = tf.reduce_sum(loss_mean, axis=-1)
            loss_mean = tf.reduce_sum(loss_mean, axis=-1)
            # get normalization factor in_sample and out_sample only.
            # It is equal to scaling factor * no. of meas loc +
            # (1-scaling factor) * no. of loc outside buildings
            sum_m_sample_scaling = tf.reduce_sum(m_sample_scaling, axis=-1)
            sum_m_sample_scaling = tf.reduce_sum(sum_m_sample_scaling, axis=-1)
            loss_mean = tf.divide(loss_mean, tf.math.maximum(sum_m_sample_scaling, tf.constant([1.0])))

            return loss_mean

        def loss_function_keras(y_true, y_predict):
            loss_mean_keras = tf.keras.losses.MSE(y_true[:, 0, ...], y_predict[:, 0, ...])
            return loss_mean_keras

        def loss_mean_in_sample(y_true, y_predict):
            """
            Args:
                -`y_true`: is N x C x H x W tensor tha contains C= num_sources true maps as a target.
                        if channels are combined then C = 1.

                -`y_predict`: is N x 4 x H x W tensor where
                        y_predict[:,0,...] is an estimated mean power,
                        y_predict[:,1,...] is an estimated std. deviation,
                        y_predict[:,2,...] is a sample_scaling tensor
                        y_predict[:,3,...] is a mask

            """
            # y_true = tf.transpose(y_true, perm=[0, 2, 3, 1])
            # t_delta = tf.square(y_true[:, 0, ...] - y_predict[:, 0, ...])
            m_in_sample_scaling = tf.where(y_predict[:, 3, ...] == 1, 1.0, 0.0)
            m_mean = y_predict[:, 0, ...]
            t_delta = tf.square(y_true[:, 0, ...] - m_mean) * m_in_sample_scaling
            loss_mean = t_delta

            # Normalization factor, in this case it is total number of measurement
            # points.
            sum_m_sample_scaling = tf.reduce_sum(m_in_sample_scaling, axis=-1)
            sum_m_sample_scaling = tf.reduce_sum(sum_m_sample_scaling, axis=-1)
            # reduce the sum except the batch dimension
            loss_mean = tf.reduce_sum(loss_mean, axis=-1)
            loss_mean = tf.reduce_sum(loss_mean, axis=-1)
            # average the loss at in sample
            loss_mean = tf.divide(loss_mean, tf.math.maximum(sum_m_sample_scaling, tf.constant([1.0])))

            return loss_mean

        def loss_sigma_in_sample(y_true, y_predict):
            """
            Args:
                -`y_true`: is N x C x H x W tensor tha contains C= num_sources true maps as a target.
                        if channels are combined then C = 1.

                -`y_predict`: is N x 4 x H x W tensor where
                        y_predict[:,0,...] is an estimated mean power,
                        y_predict[:,1,...] is an estimated std. deviation,
                        y_predict[:,2,...] is a sample_scaling tensor
                        y_predict[:,3,...] is a mask

            """
            # y_true = tf.transpose(y_true, perm=[0, 2, 3, 1])
            # t_delta = tf.abs(y_true[:, 0, ...] - y_predict[:, 0, ...])
            # loss_sigma = tf.square(t_delta - y_predict[:, 1, ...])
            m_in_sample_scaling = tf.where(y_predict[:, 3, ...] == 1, 1.0, 0.0)
            m_mean = y_predict[:, 0, ...]
            m_std = y_predict[:, 1, ...]
            t_delta = tf.abs(y_true[:, 0, ...] - m_mean) #* m_sample_scaling
            loss_sigma = tf.square(t_delta - m_std) * m_in_sample_scaling

            sum_m_sample_scaling = tf.reduce_sum(m_in_sample_scaling, axis=-1)
            sum_m_sample_scaling = tf.reduce_sum(sum_m_sample_scaling, axis=-1)
            # num_in_sample = len(tf.where(y_predict[:, 3, ...] == 1))
            loss_sigma = tf.reduce_sum(loss_sigma, axis=-1)
            loss_sigma = tf.reduce_sum(loss_sigma, axis=-1)
            loss_sigma = tf.divide(loss_sigma, tf.math.maximum(sum_m_sample_scaling, tf.constant([1.0])))

            return loss_sigma

        def loss_mean_out_sample(y_true, y_predict):
            """
            Args:
                -`y_true`: is N x C x H x W tensor tha contains C= num_sources true maps as a target.
                        if channels are combined then C = 1.

                -`y_predict`: is N x 4 x H x W tensor where
                        y_predict[:,0,...] is an estimated mean power,
                        y_predict[:,1,...] is an estimated std. deviation,
                        y_predict[:,2,...] is a sample_scaling tensor
                        y_predict[:,3,...] is a mask

            """
            # y_true = tf.transpose(y_true, perm=[0, 2, 3, 1])
            # t_delta = tf.square(y_true[:, 0, ...] - y_predict[:, 0, ...])
            m_out_sample_scaling = tf.where(y_predict[:, 3, ...] == 0, 1.0, 0.0)
            m_mean = y_predict[:, 0, ...]
            t_delta = tf.square(y_true[:, 0, ...] - m_mean) * m_out_sample_scaling
            loss_mean = t_delta  # rmse

            sum_m_sample_scaling = tf.reduce_sum(m_out_sample_scaling, axis=-1)
            sum_m_sample_scaling = tf.reduce_sum(sum_m_sample_scaling, axis=-1)
            # num_out_sample = len(tf.where(y_predict[:, 3, ...] == 0))
            loss_mean = tf.reduce_sum(loss_mean, axis=-1)
            loss_mean = tf.reduce_sum(loss_mean, axis=-1)
            loss_mean = tf.divide(loss_mean, tf.math.maximum(sum_m_sample_scaling, tf.constant([1.0])))

            return loss_mean

        def loss_sigma_out_sample(y_true, y_predict):
            """
            Args:
                -`y_true`: is N x C x H x W tensor tha contains C= num_sources true maps as a target.
                        if channels are combined then C = 1.

                -`y_predict`: is N x 4 x H x W tensor where
                        y_predict[:,0,...] is an estimated mean power,
                        y_predict[:,1,...] is an estimated std. deviation,
                        y_predict[:,2,...] is a sample_scaling tensor
                        y_predict[:,3,...] is a mask

            """
            # y_true = tf.transpose(y_true, perm=[0, 2, 3, 1])
            # t_delta = tf.abs(y_true[:, 0, ...] - y_predict[:, 0, ...])
            # loss_sigma = tf.square(t_delta - y_predict[:, 1, ...])
            m_out_sample_scaling = tf.where(y_predict[:, 3, ...] == 0, 1.0, 0.0)
            m_mean = y_predict[:, 0, ...]
            m_std = y_predict[:, 1, ...]
            t_delta = tf.abs(y_true[:, 0, ...] - m_mean)  # * m_sample_scaling
            loss_sigma = tf.square(t_delta - m_std) * m_out_sample_scaling

            sum_m_sample_scaling = tf.reduce_sum(m_out_sample_scaling, axis=-1)
            sum_m_sample_scaling = tf.reduce_sum(sum_m_sample_scaling, axis=-1)

            loss_sigma = tf.reduce_sum(loss_sigma, axis=-1)
            loss_sigma = tf.reduce_sum(loss_sigma, axis=-1)
            loss_sigma = tf.divide(loss_sigma, tf.math.maximum(sum_m_sample_scaling, tf.constant([1.0])))

            return loss_sigma

        for alpha, epochs, learning_rate in zip(l_alpha, l_epochs, l_learning_rate):

            # Loss function
            def loss_function(y_true, y_predict):
                """
                Args:
                    -`y_true`: is N x C x H x W tensor tha contains C= num_sources true maps as a target.
                            if channels are combined then C = 1.

                    -`y_predict`: is N x 4 x H x W tensor where
                            y_predict[:,0,...] is an estimated mean power,
                            y_predict[:,1,...] is an estimated std. deviation,
                            y_predict[:,2,...] is a sample_scaling tensor
                            y_predict[:,3,...] is a mask

                """
                # print("Y_true", y_true.shape)
                # y_true = tf.transpose(y_true, perm=[0, 2, 3, 1])
                # print("Y-predict shape", y_predict.shape)
                loss_mean = loss_function_mean(y_true, y_predict)
                loss_sigma = loss_function_sigma(y_true, y_predict)
                loss = alpha * loss_sigma + (1 - alpha) * loss_mean

                return loss

            # loss metric
            if loss_metric == 'Custom':
                assert alpha is not None
                if alpha == 0:
                    # freeze combined lsayers and std. deviation layers
                    self.combined_layers.trainable = False
                    self.std_deviation_layers.trainable = False
                    self.mean_layers.trainable = True

                elif alpha == 1:
                    # freeze combined layers and mean layers
                    self.combined_layers.trainable = False
                    self.std_deviation_layers.trainable = True
                    self.mean_layers.trainable = False

                elif alpha == 0.00001:
                    # freeze std. deviation layer only.
                    self.combined_layers.trainable = True
                    self.std_deviation_layers.trainable = False
                    self.mean_layers.trainable = True

                else:
                    self.combined_layers.trainable = True
                    self.std_deviation_layers.trainable = True
                    self.mean_layers.trainable = True

            print("learning rate ", learning_rate)

            weight_folder = weightfolder + \
                           f"nn_arch_{nn_arch_id}_alpha={alpha}_epochs={epochs}" \
                           f"_samp_scale={self.sample_scaling}" \
                           f"_out_same={self.meas_output_same}"
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weight_folder,
                                                             save_weights_only=save_weight_in_callback,
                                                             verbose=0,
                                                             # mode='min',
                                                             # save_best_only=True
                                                             )

            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            self.compile(optimizer=optimizer, loss=loss_function,
                         # loss_weights=[alpha, 1 - alpha],
                         metrics=[loss_function,
                                  loss_function_mean,
                                  loss_function_sigma,
                                  # loss_function_keras,
                                  ],
                         )

            if test_dataset is not None:
                self.compile(optimizer=optimizer, loss=loss_function,
                             # loss_weights=[alpha, 1 - alpha],
                             metrics=[loss_function,
                                      loss_function_mean,
                                      loss_function_sigma,
                                      # loss_function_keras,
                                      loss_mean_in_sample,
                                      loss_mean_out_sample,
                                      loss_sigma_in_sample,
                                      loss_sigma_out_sample,
                                      ],
                             )

            if test_changes:
                l_initial_weight = [self.combined_layers.get_weights(),
                                    self.mean_layers.get_weights(),
                                    self.std_deviation_layers.get_weights()]

            if train_dataset is not None:
                history = super().fit(x=train_dataset, epochs=epochs,
                                      validation_data=validation_dataset,
                                      callbacks=[cp_callback, tensorboard_callback], verbose=1)
            if test_dataset is not None:
                assert test_dataset is not None
                history_eval = super().evaluate(x=test_dataset, return_dict=True)
                return history_eval

            vector_alpha = alpha * np.ones(shape=(epochs,))
            vector_alpha.tolist()

            dict_history[f"train_mean_rmse_loss"] += history.history["loss_function_mean"]
            dict_history[f"train_sigma_rmse_error"] += history.history["loss_function_sigma"]
            dict_history[f"val_mean_rmse_loss"] += history.history["val_loss_function_mean"]
            dict_history[f"val_sigma_rmse_error"] += history.history["val_loss_function_sigma"]
            dict_history[f"train_loss"] += history.history["loss"]
            dict_history[f"val_loss"] += history.history["val_loss"]
            dict_history[f"alpha_vector"]+= vector_alpha.tolist()

            if test_changes:
                l_final_weight = [self.combined_layers.get_weights(),
                                  self.mean_layers.get_weights(),
                                  self.std_deviation_layers.get_weights()]

                print("Weight values of Combined layer are the same? ",
                      list_are_close(l_initial_weight[0], l_final_weight[0]))
                print("Weight values of mean layer are the same? ",
                      list_are_close(l_initial_weight[1], l_final_weight[1]))
                print("Weight values of std. layer are the same? ",
                      list_are_close(l_initial_weight[2], l_final_weight[2]))

            if save_weight_in_callback:
                # Save loss metrics for each value alpha
                lossfolder = loss_folder + \
                              f"nn_arch_{nn_arch_id}_alpha={alpha}_epochs={epochs}" \
                              f"_samp_scale={self.sample_scaling}" \
                              f"_out_same={self.meas_output_same}.pkl"
                outfile = open(lossfolder, 'wb')
                pickle.dump(dict_history, outfile)
                outfile.close()

        return dict_history


class ForkConvNnArch14(ForkConvNn):
    """ Fully Convolutional Neural Network with total 26 layers, 10 convolution,
        10 transpose convolution, 3 max pooling, and 3 upsampling layers.
    """

    def __init__(self):

        super(ForkConvNnArch14, self).__init__()
        self.regularizers = regularizers.l2(0.001)
        self.combined_layers = self.CombinedLayers()
        self.mean_layers = self.MeanLayers()
        self.std_deviation_layers = self.StdDeviationlayers()

    class CombinedLayers(Model):
        def __init__(self):
            super().__init__()
            self.regularizers = regularizers.l2(0.001)
            self.conv1 = Conv2D(filters=128,
                                kernel_size=3,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers,)

            self.conv2 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv3 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.max_pool_1 = MaxPool2D(pool_size=(2, 2))


            self.conv4 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv5 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv6 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.max_pool_2 = MaxPool2D(pool_size=(2, 2))

        @tf.function
        def call(self, x):
            # x = self.InputLayer
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.max_pool_1(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.max_pool_2(x)
            return x

    class MeanLayers(Model):
        def __init__(self):
            super().__init__()
            self.regularizers = regularizers.l2(0.001)
            self.conv1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.conv2T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.conv3T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.up_sampling_1 = UpSampling2D(size=(2, 2))

            self.conv4T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv5T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv6T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.up_sampling_2 = UpSampling2D(size=(2, 2))

            self.conv7 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv7T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv8 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv8T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv9 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv9T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.max_pool_3 = MaxPool2D(pool_size=(2, 2))
            self.up_sampling_3 = UpSampling2D(size=(2, 2))

            self.conv10 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                 kernel_regularizer=self.regularizers)
            self.conv10T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                           kernel_regularizer=self.regularizers)

            self.conv_out_mu = Conv2DTranspose(1, 4,
                                               activation=tf.nn.leaky_relu,
                                               padding='same',
                                               kernel_regularizer=self.regularizers,
                                               name="mean_output")

        @tf.function
        def call(self, x):
            ## separate branch from this point

            x = self.conv7(x)
            x = self.conv8(x)
            x = self.conv9(x)
            x = self.conv10(x)
            x = self.max_pool_3(x)
            x = self.conv10T(x)
            x = self.up_sampling_3(x)
            x = self.conv9T(x)
            x = self.conv8T(x)
            x = self.conv7T(x)
            x = self.up_sampling_2(x)
            x = self.conv6T(x)
            x = self.conv5T(x)
            x = self.conv4T(x)
            x = self.up_sampling_1(x)
            x = self.conv3T(x)
            x = self.conv2T(x)
            x = self.conv1T(x)

            mu = self.conv_out_mu(x)

            return mu

    class StdDeviationlayers(Model):
        def __init__(self):
            super().__init__()
            self.regularizers = regularizers.l2(0.001)
            self.conv1_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.conv2_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv3_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.up_sampling_1_1 = UpSampling2D(size=(2, 2))

            self.conv4_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv5_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv6_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.up_sampling_2_1 = UpSampling2D(size=(2, 2))


            self.conv7_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv7_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv8_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv8_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv9_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv9_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.max_pool_3_1 = MaxPool2D(pool_size=(2, 2))
            self.up_sampling_3_1 = UpSampling2D(size=(2, 2))

            self.conv10_1 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                 kernel_regularizer=self.regularizers)
            self.conv10_1T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                           kernel_regularizer=self.regularizers)

            self.conv_out_sigma = Conv2DTranspose(1, 4,
                                                  activation=lambda x: tf.nn.elu(x) + 1,
                                                  padding='same',
                                                  kernel_regularizer=self.regularizers,
                                                  name="sigma_output")

        @tf.function
        def call(self, y):
            ## separate branch from this point
            y = self.conv7_1(y)
            y = self.conv8_1(y)
            y = self.conv9_1(y)
            y = self.conv10_1(y)
            y = self.max_pool_3_1(y)
            y = self.conv10_1T(y)
            y = self.up_sampling_3_1(y)
            y = self.conv9_1T(y)
            y = self.conv8_1T(y)
            y = self.conv7_1T(y)
            y = self.up_sampling_2_1(y)
            y = self.conv6_1T(y)
            y = self.conv5_1T(y)
            y = self.conv4_1T(y)
            y = self.up_sampling_1_1(y)
            y = self.conv3_1T(y)
            y = self.conv2_1T(y)
            y = self.conv1_1T(y)


            sigma = self.conv_out_sigma(y)

            return sigma


class ForkConvNnArch16(ForkConvNn):
    """ Fully Convolutional Neural Network with total 26 layers, 10 convolution,
        10 transpose convolution, 3 max pooling, and 3 upsampling layers.
    """

    def __init__(self):

        super(ForkConvNnArch16, self).__init__()
        self.regularizers = regularizers.l2(0.001)
        self.combined_layers = self.CombinedLayers()
        self.mean_layers = self.MeanLayers()
        self.std_deviation_layers = self.StdDeviationlayers()

    class CombinedLayers(Model):
        def __init__(self):
            super().__init__()
            self.regularizers = regularizers.l2(0.001)
            self.conv1 = Conv2D(filters=128,
                                kernel_size=4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv2 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv3 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.max_pool_1 = MaxPool2D(pool_size=(2, 2))


            self.conv4 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv5 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv6 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.max_pool_2 = MaxPool2D(pool_size=(2, 2))

        @tf.function
        def call(self, x):
            # x = self.InputLayer
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.max_pool_1(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.max_pool_2(x)
            return x

    class MeanLayers(Model):
        def __init__(self):
            super().__init__()
            self.regularizers = regularizers.l2(0.001)
            self.conv1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.conv2T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.conv3T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.up_sampling_1 = UpSampling2D(size=(2, 2))

            self.conv4T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv5T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv6T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.up_sampling_2 = UpSampling2D(size=(2, 2))

            self.conv7 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv7T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv8 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv8T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv9 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv9T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.max_pool_3 = MaxPool2D(pool_size=(2, 2))
            self.up_sampling_3 = UpSampling2D(size=(2, 2))

            self.conv10 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                 kernel_regularizer=self.regularizers)
            self.conv10T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                           kernel_regularizer=self.regularizers)

            self.conv_out_mu = Conv2DTranspose(1, 4,
                                               activation=tf.nn.leaky_relu,
                                               padding='same',
                                               kernel_regularizer=self.regularizers,
                                               name="mean_output")

        @tf.function
        def call(self, x):
            ## separate branch from this point

            x = self.conv7(x)
            x = self.conv8(x)
            x = self.conv9(x)
            x = self.conv10(x)
            x = self.max_pool_3(x)
            x = self.conv10T(x)
            x = self.up_sampling_3(x)
            x = self.conv9T(x)
            x = self.conv8T(x)
            x = self.conv7T(x)
            x = self.up_sampling_2(x)
            x = self.conv6T(x)
            x = self.conv5T(x)
            x = self.conv4T(x)
            x = self.up_sampling_1(x)
            x = self.conv3T(x)
            x = self.conv2T(x)
            x = self.conv1T(x)

            mu = self.conv_out_mu(x)

            return mu

    class StdDeviationlayers(Model):
        def __init__(self):
            super().__init__()
            self.regularizers = regularizers.l2(0.001)
            self.conv1_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.conv2_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv3_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.up_sampling_1_1 = UpSampling2D(size=(2, 2))

            self.conv4_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv5_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv6_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.up_sampling_2_1 = UpSampling2D(size=(2, 2))


            self.conv7_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv7_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv8_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv8_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv9_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv9_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.max_pool_3_1 = MaxPool2D(pool_size=(2, 2))
            self.up_sampling_3_1 = UpSampling2D(size=(2, 2))

            self.conv10_1 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                 kernel_regularizer=self.regularizers)
            self.conv10_1T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                           kernel_regularizer=self.regularizers)

            self.conv_out_sigma = Conv2DTranspose(1, 4,
                                                  activation=lambda x: tf.nn.elu(x) + 1,
                                                  padding='same',
                                                  kernel_regularizer=self.regularizers,
                                                  name="sigma_output")

        @tf.function
        def call(self, y):
            ## separate branch from this point
            y = self.conv7_1(y)
            y = self.conv8_1(y)
            y = self.conv9_1(y)
            y = self.conv10_1(y)
            y = self.max_pool_3_1(y)
            y = self.conv10_1T(y)
            y = self.up_sampling_3_1(y)
            y = self.conv9_1T(y)
            y = self.conv8_1T(y)
            y = self.conv7_1T(y)
            y = self.up_sampling_2_1(y)
            y = self.conv6_1T(y)
            y = self.conv5_1T(y)
            y = self.conv4_1T(y)
            y = self.up_sampling_1_1(y)
            y = self.conv3_1T(y)
            y = self.conv2_1T(y)
            y = self.conv1_1T(y)


            sigma = self.conv_out_sigma(y)

            return sigma


class ForkConvNnArch17(ForkConvNn):
    """ Fully Convolutional Neural Network with total 26 layers, 10 convolution,
        10 transpose convolution, 3 max pooling, and 3 upsampling layers.
    """

    def __init__(self):

        super(ForkConvNnArch17, self).__init__()
        self.regularizers = regularizers.l2(0.001)
        self.combined_layers = self.CombinedLayers()
        self.mean_layers = self.MeanLayers()
        self.std_deviation_layers = self.StdDeviationlayers()

    class CombinedLayers(Model):

        def __init__(self):

            super().__init__()
            self.regularizers = None
            # self.regularizers = regularizers.l2(0.001)
            self.conv1 = Conv2D(filters=128,
                                kernel_size=4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv2 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv3 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.max_pool_1 = MaxPool2D(pool_size=(2, 2))


            self.conv4 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv5 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv6 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.max_pool_2 = MaxPool2D(pool_size=(2, 2))

        @tf.function
        def call(self, x):
            # x = self.InputLayer
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.max_pool_1(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.max_pool_2(x)
            return x

    class MeanLayers(Model):
        def __init__(self):
            super().__init__()
            self.regularizers = None
            # self.regularizers = regularizers.l2(0.001)
            self.conv1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.conv2T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.conv3T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.up_sampling_1 = UpSampling2D(size=(2, 2))

            self.conv4T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv5T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv6T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.up_sampling_2 = UpSampling2D(size=(2, 2))

            self.conv7 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv7T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv8 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv8T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv9 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv9T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.max_pool_3 = MaxPool2D(pool_size=(2, 2))
            self.up_sampling_3 = UpSampling2D(size=(2, 2))

            self.conv10 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                 kernel_regularizer=self.regularizers)
            self.conv10T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                           kernel_regularizer=self.regularizers)

            self.conv_out_mu = Conv2DTranspose(1, 4,
                                               activation=tf.nn.leaky_relu,
                                               padding='same',
                                               kernel_regularizer=self.regularizers)

        @tf.function
        def call(self, x):
            ## separate branch from this point

            x = self.conv7(x)
            x = self.conv8(x)
            x = self.conv9(x)
            x = self.max_pool_3(x)
            x = self.conv10(x)
            x = self.conv10T(x)
            x = self.up_sampling_3(x)
            x = self.conv9T(x)
            x = self.conv8T(x)
            x = self.conv7T(x)
            x = self.up_sampling_2(x)
            x = self.conv6T(x)
            x = self.conv5T(x)
            x = self.conv4T(x)
            x = self.up_sampling_1(x)
            x = self.conv3T(x)
            x = self.conv2T(x)
            x = self.conv1T(x)

            mu = self.conv_out_mu(x)

            return mu

    class StdDeviationlayers(Model):
        def __init__(self):
            super().__init__()
            self.regularizers = None
            # self.regularizers = regularizers.l2(0.001)
            self.conv1_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.conv2_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv3_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.up_sampling_1_1 = UpSampling2D(size=(2, 2))

            self.conv4_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv5_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv6_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.up_sampling_2_1 = UpSampling2D(size=(2, 2))


            self.conv7_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv7_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv8_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv8_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv9_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv9_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.max_pool_3_1 = MaxPool2D(pool_size=(2, 2))
            self.up_sampling_3_1 = UpSampling2D(size=(2, 2))

            self.conv10_1 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                 kernel_regularizer=self.regularizers)
            self.conv10_1T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                           kernel_regularizer=self.regularizers)

            self.conv_out_sigma = Conv2DTranspose(1, 4,
                                                  activation= tf.keras.activations.exponential, #lambda x: tf.nn.elu(x) + 1,
                                                  padding='same',
                                                  kernel_regularizer=self.regularizers)

        @tf.function
        def call(self, y):
            ## separate branch from this point
            y = self.conv7_1(y)
            y = self.conv8_1(y)
            y = self.conv9_1(y)
            y = self.max_pool_3_1(y)
            y = self.conv10_1(y)
            y = self.conv10_1T(y)
            y = self.up_sampling_3_1(y)
            y = self.conv9_1T(y)
            y = self.conv8_1T(y)
            y = self.conv7_1T(y)
            y = self.up_sampling_2_1(y)
            y = self.conv6_1T(y)
            y = self.conv5_1T(y)
            y = self.conv4_1T(y)
            y = self.up_sampling_1_1(y)
            y = self.conv3_1T(y)
            y = self.conv2_1T(y)
            y = self.conv1_1T(y)

            sigma = self.conv_out_sigma(y)

            return sigma


class ForkConvNnArch18(ForkConvNn):
    """ Fully Convolutional Neural Network with total 26 layers, 10 convolution,
        10 transpose convolution, 3 max pooling, and 3 upsampling layers.
    """

    def __init__(self, **kwargs):

        super(ForkConvNnArch18, self).__init__(**kwargs)
        self.regularizers = regularizers.l2(0.001)
        self.combined_layers = self.CombinedLayers()
        self.mean_layers = self.MeanLayers()
        self.std_deviation_layers = self.StdDeviationlayers()

    class CombinedLayers(Model):

        def __init__(self):

            super().__init__()
            self.regularizers = None
            # self.regularizers = regularizers.l2(0.001)
            self.conv1 = Conv2D(filters=128,
                                kernel_size=4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv2 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv3 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.max_pool_1 = MaxPool2D(pool_size=(2, 2))


            self.conv4 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv5 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv6 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.max_pool_2 = MaxPool2D(pool_size=(2, 2))

        @tf.function
        def call(self, x):
            # x = self.InputLayer
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.max_pool_1(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.max_pool_2(x)
            return x

    class MeanLayers(Model):
        def __init__(self):
            super().__init__()
            self.regularizers = None
            # self.regularizers = regularizers.l2(0.001)
            self.conv1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.conv2T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.conv3T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.up_sampling_1 = UpSampling2D(size=(2, 2))

            self.conv4T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv5T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv6T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.up_sampling_2 = UpSampling2D(size=(2, 2))

            self.conv7 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv7T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv8 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv8T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv9 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv9T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.max_pool_3 = MaxPool2D(pool_size=(2, 2))
            self.up_sampling_3 = UpSampling2D(size=(2, 2))

            self.conv10 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                 kernel_regularizer=self.regularizers)
            self.conv10T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                           kernel_regularizer=self.regularizers)

            self.conv_out_mu = Conv2DTranspose(1, 4,
                                               activation=tf.nn.leaky_relu,
                                               padding='same',
                                               kernel_regularizer=self.regularizers)

        @tf.function
        def call(self, x):
            ## separate branch from this point

            x = self.conv7(x)
            x = self.conv8(x)
            x = self.conv9(x)
            x = self.max_pool_3(x)
            x = self.conv10(x)
            x = self.conv10T(x)
            x = self.up_sampling_3(x)
            x = self.conv9T(x)
            x = self.conv8T(x)
            x = self.conv7T(x)
            x = self.up_sampling_2(x)
            x = self.conv6T(x)
            x = self.conv5T(x)
            x = self.conv4T(x)
            x = self.up_sampling_1(x)
            x = self.conv3T(x)
            x = self.conv2T(x)
            x = self.conv1T(x)

            mu = self.conv_out_mu(x)

            return mu

    class StdDeviationlayers(Model):
        def __init__(self):
            super().__init__()
            self.regularizers = None
            # self.regularizers = regularizers.l2(0.001)
            self.conv1_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.conv2_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv3_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.up_sampling_1_1 = UpSampling2D(size=(2, 2))

            self.conv4_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv5_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv6_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.up_sampling_2_1 = UpSampling2D(size=(2, 2))


            self.conv7_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv7_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv8_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv8_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv9_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv9_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.max_pool_3_1 = MaxPool2D(pool_size=(2, 2))
            self.up_sampling_3_1 = UpSampling2D(size=(2, 2))

            self.conv10_1 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                 kernel_regularizer=self.regularizers)
            self.conv10_1T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                           kernel_regularizer=self.regularizers)

            self.conv_out_sigma = Conv2DTranspose(1, 4,
                                                  activation= tf.keras.activations.exponential, #lambda x: tf.nn.elu(x) + 1,
                                                  padding='same',
                                                  kernel_regularizer=self.regularizers)

        @tf.function
        def call(self, y):
            ## separate branch from this point
            y = self.conv7_1(y)
            y = self.conv8_1(y)
            y = self.conv9_1(y)
            y = self.max_pool_3_1(y)
            y = self.conv10_1(y)
            y = self.conv10_1T(y)
            y = self.up_sampling_3_1(y)
            y = self.conv9_1T(y)
            y = self.conv8_1T(y)
            y = self.conv7_1T(y)
            y = self.up_sampling_2_1(y)
            y = self.conv6_1T(y)
            y = self.conv5_1T(y)
            y = self.conv4_1T(y)
            y = self.up_sampling_1_1(y)
            y = self.conv3_1T(y)
            y = self.conv2_1T(y)
            y = self.conv1_1T(y)

            sigma = self.conv_out_sigma(y)

            return sigma


class ForkConvNnArchExample2(ForkConvNn):
    """ Fully Convolutional Neural Network with total 26 layers, 10 convolution,
        10 transpose convolution, 3 max pooling, and 3 upsampling layers.
    """

    def __init__(self, **kwargs):

        super(ForkConvNnArchExample2, self).__init__(**kwargs)
        self.regularizers = regularizers.l2(0.001)
        self.combined_layers = self.CombinedLayers()
        self.mean_layers = self.MeanLayers()
        self.std_deviation_layers = self.StdDeviationlayers()


    class CombinedLayers(Model):

        def __init__(self):

            super().__init__()
            self.res_layer1 = ResNetLayer(filters=32, kernel_size=3)
            self.res_layer2 = ResNetLayer(filters=64, kernel_size=3)
            self.res_layer3 = ResNetLayer(filters=64, kernel_size=3)

        @tf.function
        def call(self, x):
            # x = self.InputLayer
            x = self.res_layer1(x)
            x = self.res_layer2(x)
            x = self.res_layer3(x)
            return x

    class MeanLayers(Model):
        def __init__(self):
            super().__init__()
            self.regularizers = None
            self.res_layer1 = ResNetLayer(filters=32, kernel_size=3)
            self.res_layer2 = ResNetLayer(filters=64, kernel_size=3)
            self.res_layer3 = ResNetLayer(filters=64, kernel_size=3)
            self.conv_out_mu = Conv2DTranspose(1, 4,
                                               activation=tf.nn.leaky_relu,
                                               padding='same',
                                               kernel_regularizer=self.regularizers)

        @tf.function
        def call(self, x):
            ## separate branch from this point

            x = self.res_layer1(x)
            x = self.res_layer2(x)
            x = self.res_layer3(x)
            mu = self.conv_out_mu(x)

            return mu

    class StdDeviationlayers(Model):
        def __init__(self):
            super().__init__()
            self.res_layer1 = ResNetLayer(filters=32, kernel_size=3)
            self.res_layer2 = ResNetLayer(filters=64, kernel_size=3)
            self.res_layer3 = ResNetLayer(filters=64, kernel_size=3)
            self.regularizers = None
            self.conv_out_sigma = Conv2DTranspose(1, 4,
                                                  activation= tf.keras.activations.exponential, #lambda x: tf.nn.elu(x) + 1,
                                                  padding='same',
                                                  kernel_regularizer=self.regularizers)

        @tf.function
        def call(self, y):
            ## separate branch from this point
            y = self.res_layer1(y)
            y = self.res_layer2(y)
            y = self.res_layer3(y)
            sigma = self.conv_out_sigma(y)
            return sigma


class ResNetLayer(Model):
    def __init__(self,
                 filters=32,
                 kernel_size=3,
                 activation=tf.nn.leaky_relu,
                 pooling=False,
                 dropout=0.0):
        super().__init__()
        self.pooling = pooling
        self.dropout = dropout
        self.conv1 = Conv2D(filters=filters,kernel_size=kernel_size, padding="same")
        self.batch_norm = BatchNormalization()
        self.activation = activation
        self.conv2 = Conv2D(filters=filters, kernel_size=kernel_size, padding="same")
        self.conv3 = Conv2D(filters=filters, kernel_size=kernel_size, padding="same")
        self.max_pool = MaxPool2D((2,2))
        self.drop_out = Dropout

    @tf.function
    def call(self, x):
        # x = self.InputLayer
        temp = x
        temp = self.conv1(temp)
        temp = self.batch_norm(temp)
        temp = self.activation(temp)
        temp = self.conv2(temp)
        x = add([temp, self.conv3(x)])
        if self.pooling:
            x = self.max_pool((2, 2))(x)
        if self.dropout != 0.0:
            x = self.drop_out(self.dropout)(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


class ForkConvNnArchExample(ForkConvNn):
    """ Fully Convolutional Neural Network with total 26 layers, 10 convolution,
        10 transpose convolution, 3 max pooling, and 3 upsampling layers.
    """

    def __init__(self):

        super(ForkConvNnArchExample, self).__init__()
        self.initializers = tf.keras.initializers.GlorotNormal()
        # self.regularizers = regularizers.l2(0.001)
        self.combined_layers = self.CombinedLayers()
        self.mean_layers = self.MeanLayers()
        self.std_deviation_layers = self.StdDeviationlayers()

    class CombinedLayers(Model):
        def __init__(self):
            super().__init__()
            self.regularizers = None #regularizers.l2(0.001)
            self.initializers = None
            # self.initializers = tf.keras.initializers.GlorotUniform()
            self.conv1 = Conv2D(filters=64,
                                kernel_size=4,
                                # activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers,
                                kernel_initializer=self.initializers)

            self.conv2 = Conv2D(filters=64,
                                kernel_size=4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers,
                                kernel_initializer=self.initializers)

            self.max_pool = MaxPool2D((2, 2))
            self.up_sam = UpSampling2D(size=(2, 2))

            self.conv3 = Conv2D(64, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers,
                                kernel_initializer=self.initializers)
            self.norm = tf.keras.layers.BatchNormalization()
            self.act_fun = tf.keras.activations.relu
            self.conv3T = Conv2DTranspose(64, 4,
                                          activation=tf.nn.leaky_relu,
                                          padding='same',
                                          kernel_regularizer=self.regularizers,
                                          kernel_initializer=self.initializers
                                          )
            self.conv2T = Conv2DTranspose(64, 4,
                                          activation=tf.nn.leaky_relu,
                                          padding='same',
                                          kernel_regularizer=self.regularizers,
                                          kernel_initializer=self.initializers
                                          )
            self.conv1T = Conv2DTranspose(64, 4,
                                          activation=tf.nn.leaky_relu,
                                          padding='same',
                                          kernel_regularizer=self.regularizers,
                                          kernel_initializer=self.initializers
                                          )

        @tf.function
        def call(self, x):
            # x = self.InputLayer
            x = self.conv1(x)
            x = self.norm(x)
            x = self.act_fun(x)
            x = self.conv2(x)
            x = self.max_pool(x)
            x= self.conv3(x)
            x= self.conv3T(x)
            x = self.up_sam(x)
            x = self.conv2T(x)
            x = self.conv1T(x)
            return x

    class MeanLayers(Model):
        def __init__(self):
            super().__init__()
            # self.regularizers = regularizers.l2(0.001)
            self.regularizers = None
            self.initializers = None
            # self.initializers = tf.keras.initializers.GlorotUniform()
            self.conv1 = Conv2D(10, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers,
                                kernel_initializer=self.initializers)

            self.max_pool = MaxPool2D((2, 2))
            self.up_sam = UpSampling2D(size=(2, 2))

            self.conv3 = Conv2D(10, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers,
                                kernel_initializer=self.initializers)

            self.conv3T = Conv2DTranspose(10, 4,
                                               activation=tf.nn.leaky_relu,
                                               padding='same',
                                               kernel_regularizer=self.regularizers,
                                               kernel_initializer=self.initializers
                                              )
            self.conv1T = Conv2DTranspose(10, 4,
                                          activation=tf.nn.leaky_relu,
                                          padding='same',
                                          kernel_regularizer=self.regularizers,
                                          kernel_initializer=self.initializers
                                          )

            self.conv_out_mu = Conv2DTranspose(1, 4,
                                               activation=tf.nn.leaky_relu,
                                               padding='same',
                                               kernel_regularizer=self.regularizers,
                                               kernel_initializer=self.initializers
                                              )
        @tf.function
        def call(self, x):
            x = self.conv1(x)
            x = self.max_pool(x)
            x = self.conv3(x)
            x = self.conv3T(x)
            x= self.up_sam(x)
            x = self.conv1T(x)
            mu = self.conv_out_mu(x)

            return mu

    class StdDeviationlayers(Model):
        def __init__(self):
            super().__init__()
            self.regularizers = None #regularizers.l2(0.001)
            self.initializers = None
            # self.initializers = tf.keras.initializers.GlorotUniform()
            self.conv1 = Conv2D(10, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers,
                                kernel_initializer=self.initializers)

            self.max_pool = MaxPool2D((2, 2))
            self.up_sam = UpSampling2D(size=(2, 2))

            self.conv3 = Conv2D(10, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers,
                                kernel_initializer=self.initializers)

            self.conv3T = Conv2DTranspose(10, 4,
                                          activation=tf.nn.leaky_relu,
                                          padding='same',
                                          kernel_regularizer=self.regularizers,
                                          kernel_initializer=self.initializers
                                          )
            self.conv1T = Conv2DTranspose(10, 4,
                                          activation=tf.nn.leaky_relu,
                                          padding='same',
                                          kernel_regularizer=self.regularizers,
                                          kernel_initializer=self.initializers
                                          )
            self.conv_out_sigma = Conv2DTranspose(1, 4,
                                                  activation=lambda x: tf.nn.elu(x) + 1,
                                                  padding='same',
                                                  kernel_regularizer=self.regularizers,
                                                  kernel_initializer=self.initializers)
        @tf.function
        def call(self, x):
            x = self.conv1(x)
            x = self.max_pool(x)
            x = self.conv3(x)
            x = self.conv3T(x)
            x = self.up_sam(x)
            x = self.conv1T(x)
            sigma = self.conv_out_sigma(x)

            return sigma


class CombinedLayers(Model):
    def __init__(self):
        super(CombinedLayers, self).__init__()
        self.regularizers = regularizers.l2(0.001)
        self.conv1 = Conv2D(filters=128,
                            kernel_size=4,
                            activation=tf.nn.leaky_relu,
                            padding='same',
                            kernel_regularizer=self.regularizers)

        self.conv2 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)

        self.conv3 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)

        self.max_pool_1 = MaxPool2D(pool_size=(2, 2))

        self.conv4 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)

        self.conv5 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)

        self.conv6 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)

        self.max_pool_2 = MaxPool2D(pool_size=(2, 2))

    @tf.function
    def call(self, x):
        # x = self.InputLayer
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool_1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max_pool_2(x)
        return x

class MeanLayers(Model):
    def __init__(self):
        super(MeanLayers, self).__init__()
        self.regularizers = regularizers.l2(0.001)
        self.conv1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                      kernel_regularizer=self.regularizers)

        self.conv2T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                      kernel_regularizer=self.regularizers)

        self.conv3T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                      kernel_regularizer=self.regularizers)

        self.up_sampling_1 = UpSampling2D(size=(2, 2))

        self.conv4T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                      kernel_regularizer=self.regularizers)
        self.conv5T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                      kernel_regularizer=self.regularizers)
        self.conv6T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                      kernel_regularizer=self.regularizers)

        self.up_sampling_2 = UpSampling2D(size=(2, 2))

        self.conv7 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv7T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                      kernel_regularizer=self.regularizers)
        self.conv8 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv8T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                      kernel_regularizer=self.regularizers)
        self.conv9 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                            kernel_regularizer=self.regularizers)
        self.conv9T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                      kernel_regularizer=self.regularizers)

        self.max_pool_3 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_3 = UpSampling2D(size=(2, 2))

        self.conv10 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                             kernel_regularizer=self.regularizers)
        self.conv10T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                       kernel_regularizer=self.regularizers)

        self.conv_out_mu = Conv2DTranspose(1, 4,
                                           activation=tf.nn.leaky_relu,
                                           padding='same',
                                           kernel_regularizer=self.regularizers)

    @tf.function
    def call(self, x):
        ## separate branch from this point

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.max_pool_3(x)
        x = self.conv10T(x)
        x = self.up_sampling_3(x)
        x = self.conv9T(x)
        x = self.conv8T(x)
        x = self.conv7T(x)
        x = self.up_sampling_2(x)
        x = self.conv6T(x)
        x = self.conv5T(x)
        x = self.conv4T(x)
        x = self.up_sampling_1(x)
        x = self.conv3T(x)
        x = self.conv2T(x)
        x = self.conv1T(x)

        mu = self.conv_out_mu(x)

        return mu


class StdDeviationlayers(Model):
    def __init__(self):
        super(StdDeviationlayers, self).__init__()
        self.regularizers = regularizers.l2(0.001)
        self.conv1_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                        kernel_regularizer=self.regularizers)

        self.conv2_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                        kernel_regularizer=self.regularizers)

        self.conv3_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                        kernel_regularizer=self.regularizers)

        self.up_sampling_1_1 = UpSampling2D(size=(2, 2))

        self.conv4_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                        kernel_regularizer=self.regularizers)
        self.conv5_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                        kernel_regularizer=self.regularizers)
        self.conv6_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                        kernel_regularizer=self.regularizers)
        self.up_sampling_2_1 = UpSampling2D(size=(2, 2))

        self.conv7_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                              kernel_regularizer=self.regularizers)
        self.conv7_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                        kernel_regularizer=self.regularizers)
        self.conv8_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                              kernel_regularizer=self.regularizers)
        self.conv8_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                        kernel_regularizer=self.regularizers)
        self.conv9_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                              kernel_regularizer=self.regularizers)
        self.conv9_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                        kernel_regularizer=self.regularizers)

        self.max_pool_3_1 = MaxPool2D(pool_size=(2, 2))
        self.up_sampling_3_1 = UpSampling2D(size=(2, 2))

        self.conv10_1 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=self.regularizers)
        self.conv10_1T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                         kernel_regularizer=self.regularizers)

        self.conv_out_sigma = Conv2DTranspose(1, 4,
                                              activation=lambda x: tf.nn.elu(x) + 1,
                                              padding='same',
                                              kernel_regularizer=self.regularizers)

    @tf.function
    def call(self, x):
        ## separate branch from this point
        y = self.conv7_1(x)
        y = self.conv8_1(y)
        y = self.conv9_1(y)
        y = self.conv10_1(y)
        y = self.max_pool_3_1(y)
        y = self.conv10_1T(y)
        y = self.up_sampling_3_1(y)
        y = self.conv9_1T(y)
        y = self.conv8_1T(y)
        y = self.conv7_1T(y)
        y = self.up_sampling_2_1(y)
        y = self.conv6_1T(y)
        y = self.conv5_1T(y)
        y = self.conv4_1T(y)
        y = self.up_sampling_1_1(y)
        y = self.conv3_1T(y)
        y = self.conv2_1T(y)
        y = self.conv1_1T(y)

        sigma = self.conv_out_sigma(y)

        return sigma

class block1(Model):
    def __init__(self):
        super(block1, self).__init__()
        self.combined_layers = Conv2D(filters=2,
                            kernel_size=4,
                            activation=tf.nn.leaky_relu,
                            padding='same',
                            )

        self.mean_layers = Conv2D(filters=1,
                            kernel_size=4,
                            activation=tf.nn.leaky_relu,
                            padding='same',
                            )
        self.std_deviation_layers = Conv2D(filters=1,
                            kernel_size=4,
                            activation=tf.nn.leaky_relu,
                            padding='same',
                            )
    @tf.function
    def call(self, x):
        x = self.combined_layers(x)
        mu = self.mean_layers(x)
        return x


class SkipConvNn(Model):
    def __init__(self,
                 sample_scaling=0.5,
                 meas_output_same=False):
        super().__init__()

        self.sample_scaling = sample_scaling
        self.meas_output_same = meas_output_same

    # a factory method that would instantiate the class given its name:
    @staticmethod
    def get_skip_conv_nn_arch(nn_arch_id):
        return globals()[f'SkipConvNnArch' + nn_arch_id]()

    @tf.function
    def call(self, x):
        """
        Args:
            `x`: 'NCHW' format batch tf dataset.
        Returns:
            `output`: a tensor of format NHWC (C=3) that is formed
                by concatenation of mu, sigma and m_sample_scaling
                along last dimension.
        """
        # Convert the input data into the`NHWC` format that is channels_last
        x = tf.transpose(x, perm=[0, 2, 3, 1])

        # comb_layer_output = self.combined_layers(x)
        mu = self.mean_layers(x)

        sampled_map = x[..., 0][..., None]
        mask = x[..., 1][..., None]
        if self.meas_output_same:
            # Keep the estimated to be the same as measured value at
            # observed locations.
            mu = tf.where(mask == 1, sampled_map, mu)
            # mu[mask == 1] = sampled_map[mask == 1]

        input_to_sigma_layer = concatenate([x, mu], axis=-1)
        # print(input_to_sigma_layer.shape)
        # set_trace()
        sigma = self.std_deviation_layers(input_to_sigma_layer)
        # z = tf.concat((mu, sigma), axis=3)

        m_sample_scaling = tf.where(mask == 1, self.sample_scaling, 1 - self.sample_scaling)
        m_sample_scaling = tf.where(mask == -1, 0.0, m_sample_scaling)

        mean_n_sigma = tf.concat((mu, sigma), axis=-1)
        mean_sigma_n_sample_scaling = tf.concat((mean_n_sigma, m_sample_scaling), axis=-1)

        # in_sample_scaling = tf.where(mask == 1, 1.0, 0.0) # to calculate loss at sample locations
        # mean_sigma_n_sample_scaling = tf.concat((mean_sigma_n_sample_scaling, in_sample_scaling), axis=-1)
        #
        # out_sample_scaling = tf.where(mask == 0, 1.0, 0.0)  # to calculate loss at unobserved locations
        # mean_sigma_n_sample_scaling = tf.concat((mean_sigma_n_sample_scaling, out_sample_scaling), axis=-1)

        mean_sigma_n_sample_scaling = tf.concat((mean_sigma_n_sample_scaling, mask), axis=-1)

        output = tf.transpose(mean_sigma_n_sample_scaling, perm=[0, 3, 1, 2])
        # print("output shape ", output.shape)
        return output  # tf.transpose(z, perm=[0, 3, 1, 2])

    def fit(self, train_dataset=None, l_alpha=[],
            l_epochs=[], test_changes=False, loss_metric=None,
            l_learning_rate=[],
            batch_size=None, validation_dataset=None,
            callbacks=[], nn_arch_id='1',
            save_weight_in_callback='True',
            tensorboard_callback=[],
            weightfolder="./saved_weights/",
            loss_folder="./train_data/",
            b_evaluate=False,
            test_dataset=None):
        """
        Returns:
            dic_history: a dictionary that contians
                    "train_mean_rmse_loss": a list of length len(l_apha)x epochs that contains
                    a concatenated train loss for all entries of l_alpha.
        """

        dict_history = {"train_mean_rmse_loss": [],
                        "train_sigma_rmse_error": [],
                        "val_mean_rmse_loss": [],
                        "val_sigma_rmse_error": [],
                        "alpha_vector": [],
                        "train_loss": [],
                        "val_loss": [],
                        }
        assert l_learning_rate != []

        def loss_function_sigma(y_true, y_predict):
            """
            Args:
                -`y_true`: is N x C x H x W tensor tha contains C= num_sources true maps as a target.
                        if channels are combined then C = 1.

                -`y_predict`: is N x 4 x H x W tensor where
                        y_predict[:,0,...] is an estimated mean power,
                        y_predict[:,1,...] is an estimated std. deviation,
                        y_predict[:,2,...] is a sample_scaling tensor
                        y_predict[:,3,...] is a mask

            """
            # y_true = tf.transpose(y_true, perm=[0, 2, 3, 1])
            # t_delta = tf.abs(y_true[:, 0, ...] - y_predict[:, 0, ...])
            # loss_sigma = tf.square(t_delta - y_predict[:, 1, ...])
            m_sample_scaling = y_predict[:, 2, ...]
            m_mean = y_predict[:, 0, ...]
            m_std = y_predict[:, 1, ...]
            t_delta = tf.abs(y_true[:, 0, ...] - m_mean)  # * m_sample_scaling
            loss_sigma = tf.square(t_delta - m_std) * m_sample_scaling
            # loss_sigma = tf.reduce_mean(tf.square(tf.sqrt(t_delta) - y_predict[:, 1, ...]))
            # loss_sigma = tf.square(t_delta - y_predict[:, 1, ...])
            # reduce sum except for the batch dimension
            loss_sigma = tf.reduce_sum(loss_sigma, axis=-1)
            loss_sigma = tf.reduce_sum(loss_sigma, axis=-1)
            # get normalization factor in_sample and out_sample only.
            # It is equal to scaling factor * no. of meas loc +
            # (1-scaling factor) * no. of loc outside buildings
            sum_m_sample_scaling = tf.reduce_sum(m_sample_scaling, axis=-1)
            sum_m_sample_scaling = tf.reduce_sum(sum_m_sample_scaling, axis=-1)
            loss_sigma = tf.divide(loss_sigma,
                                   tf.math.maximum(sum_m_sample_scaling, tf.constant([1.0])))
            # print(tf.math.maximum(sum_m_sample_scaling, tf.constant([1.0])))

            return loss_sigma

        def loss_function_mean(y_true, y_predict):
            """
            Args:
                -`y_true`: is N x C x H x W tensor tha contains C= num_sources true maps as a target.
                        if channels are combined then C = 1.

                -`y_predict`: is N x 4 x H x W tensor where
                        y_predict[:,0,...] is an estimated mean power,
                        y_predict[:,1,...] is an estimated std. deviation,
                        y_predict[:,2,...] is a sample_scaling tensor
                        y_predict[:,3,...] is a mask

            """
            # y_true = tf.transpose(y_true, perm=[0, 2, 3, 1])
            # t_delta = tf.square(y_true[:, 0, ...] - y_predict[:, 0, ...])
            m_sample_scaling = y_predict[:, 2, ...]
            m_mean = y_predict[:, 0, ...]
            t_delta = tf.square(y_true[:, 0, ...] - m_mean) * m_sample_scaling
            loss_mean = t_delta  # rmse

            # reduce sum except for the batch dimension
            loss_mean = tf.reduce_sum(loss_mean, axis=-1)
            loss_mean = tf.reduce_sum(loss_mean, axis=-1)
            # get normalization factor in_sample and out_sample only.
            # It is equal to scaling factor * no. of meas loc +
            # (1-scaling factor) * no. of loc outside buildings
            sum_m_sample_scaling = tf.reduce_sum(m_sample_scaling, axis=-1)
            sum_m_sample_scaling = tf.reduce_sum(sum_m_sample_scaling, axis=-1)
            loss_mean = tf.divide(loss_mean, tf.math.maximum(sum_m_sample_scaling, tf.constant([1.0])))

            return loss_mean

        def loss_function_keras(y_true, y_predict):
            loss_mean_keras = tf.keras.losses.MSE(y_true[:, 0, ...], y_predict[:, 0, ...])
            return loss_mean_keras

        def loss_mean_in_sample(y_true, y_predict):
            """
            Args:
                -`y_true`: is N x C x H x W tensor tha contains C= num_sources true maps as a target.
                        if channels are combined then C = 1.

                -`y_predict`: is N x 4 x H x W tensor where
                        y_predict[:,0,...] is an estimated mean power,
                        y_predict[:,1,...] is an estimated std. deviation,
                        y_predict[:,2,...] is a sample_scaling tensor
                        y_predict[:,3,...] is a mask

            """
            # y_true = tf.transpose(y_true, perm=[0, 2, 3, 1])
            # t_delta = tf.square(y_true[:, 0, ...] - y_predict[:, 0, ...])
            m_in_sample_scaling = tf.where(y_predict[:, 3, ...] == 1, 1.0, 0.0)
            m_mean = y_predict[:, 0, ...]
            t_delta = tf.square(y_true[:, 0, ...] - m_mean) * m_in_sample_scaling
            loss_mean = t_delta

            # Normalization factor, in this case it is total number of measurement
            # points.
            sum_m_sample_scaling = tf.reduce_sum(m_in_sample_scaling, axis=-1)
            sum_m_sample_scaling = tf.reduce_sum(sum_m_sample_scaling, axis=-1)
            # reduce the sum except the batch dimension
            loss_mean = tf.reduce_sum(loss_mean, axis=-1)
            loss_mean = tf.reduce_sum(loss_mean, axis=-1)
            # average the loss at in sample
            loss_mean = tf.divide(loss_mean, tf.math.maximum(sum_m_sample_scaling, tf.constant([1.0])))

            return loss_mean

        def loss_sigma_in_sample(y_true, y_predict):
            """
            Args:
                -`y_true`: is N x C x H x W tensor tha contains C= num_sources true maps as a target.
                        if channels are combined then C = 1.

                -`y_predict`: is N x 4 x H x W tensor where
                        y_predict[:,0,...] is an estimated mean power,
                        y_predict[:,1,...] is an estimated std. deviation,
                        y_predict[:,2,...] is a sample_scaling tensor
                        y_predict[:,3,...] is a mask

            """
            # y_true = tf.transpose(y_true, perm=[0, 2, 3, 1])
            # t_delta = tf.abs(y_true[:, 0, ...] - y_predict[:, 0, ...])
            # loss_sigma = tf.square(t_delta - y_predict[:, 1, ...])
            m_in_sample_scaling = tf.where(y_predict[:, 3, ...] == 1, 1.0, 0.0)
            m_mean = y_predict[:, 0, ...]
            m_std = y_predict[:, 1, ...]
            t_delta = tf.abs(y_true[:, 0, ...] - m_mean)  # * m_sample_scaling
            loss_sigma = tf.square(t_delta - m_std) * m_in_sample_scaling

            sum_m_sample_scaling = tf.reduce_sum(m_in_sample_scaling, axis=-1)
            sum_m_sample_scaling = tf.reduce_sum(sum_m_sample_scaling, axis=-1)
            # num_in_sample = len(tf.where(y_predict[:, 3, ...] == 1))
            loss_sigma = tf.reduce_sum(loss_sigma, axis=-1)
            loss_sigma = tf.reduce_sum(loss_sigma, axis=-1)
            loss_sigma = tf.divide(loss_sigma, tf.math.maximum(sum_m_sample_scaling, tf.constant([1.0])))

            return loss_sigma

        def loss_mean_out_sample(y_true, y_predict):
            """
            Args:
                -`y_true`: is N x C x H x W tensor tha contains C= num_sources true maps as a target.
                        if channels are combined then C = 1.

                -`y_predict`: is N x 4 x H x W tensor where
                        y_predict[:,0,...] is an estimated mean power,
                        y_predict[:,1,...] is an estimated std. deviation,
                        y_predict[:,2,...] is a sample_scaling tensor
                        y_predict[:,3,...] is a mask

            """
            # y_true = tf.transpose(y_true, perm=[0, 2, 3, 1])
            # t_delta = tf.square(y_true[:, 0, ...] - y_predict[:, 0, ...])
            m_out_sample_scaling = tf.where(y_predict[:, 3, ...] == 0, 1.0, 0.0)
            m_mean = y_predict[:, 0, ...]
            t_delta = tf.square(y_true[:, 0, ...] - m_mean) * m_out_sample_scaling
            loss_mean = t_delta  # rmse

            sum_m_sample_scaling = tf.reduce_sum(m_out_sample_scaling, axis=-1)
            sum_m_sample_scaling = tf.reduce_sum(sum_m_sample_scaling, axis=-1)
            # num_out_sample = len(tf.where(y_predict[:, 3, ...] == 0))
            loss_mean = tf.reduce_sum(loss_mean, axis=-1)
            loss_mean = tf.reduce_sum(loss_mean, axis=-1)
            loss_mean = tf.divide(loss_mean, tf.math.maximum(sum_m_sample_scaling, tf.constant([1.0])))

            return loss_mean

        def loss_sigma_out_sample(y_true, y_predict):
            """
            Args:
                -`y_true`: is N x C x H x W tensor tha contains C= num_sources true maps as a target.
                        if channels are combined then C = 1.

                -`y_predict`: is N x 4 x H x W tensor where
                        y_predict[:,0,...] is an estimated mean power,
                        y_predict[:,1,...] is an estimated std. deviation,
                        y_predict[:,2,...] is a sample_scaling tensor
                        y_predict[:,3,...] is a mask

            """
            # y_true = tf.transpose(y_true, perm=[0, 2, 3, 1])
            # t_delta = tf.abs(y_true[:, 0, ...] - y_predict[:, 0, ...])
            # loss_sigma = tf.square(t_delta - y_predict[:, 1, ...])
            m_out_sample_scaling = tf.where(y_predict[:, 3, ...] == 0, 1.0, 0.0)
            m_mean = y_predict[:, 0, ...]
            m_std = y_predict[:, 1, ...]
            t_delta = tf.abs(y_true[:, 0, ...] - m_mean)  # * m_sample_scaling
            loss_sigma = tf.square(t_delta - m_std) * m_out_sample_scaling

            sum_m_sample_scaling = tf.reduce_sum(m_out_sample_scaling, axis=-1)
            sum_m_sample_scaling = tf.reduce_sum(sum_m_sample_scaling, axis=-1)

            loss_sigma = tf.reduce_sum(loss_sigma, axis=-1)
            loss_sigma = tf.reduce_sum(loss_sigma, axis=-1)
            loss_sigma = tf.divide(loss_sigma, tf.math.maximum(sum_m_sample_scaling, tf.constant([1.0])))

            return loss_sigma

        for alpha, epochs, learning_rate in zip(l_alpha, l_epochs, l_learning_rate):

            # Loss function
            def loss_function(y_true, y_predict):
                """
                Args:
                    -`y_true`: is N x C x H x W tensor tha contains C= num_sources true maps as a target.
                            if channels are combined then C = 1.

                    -`y_predict`: is N x 4 x H x W tensor where
                            y_predict[:,0,...] is an estimated mean power,
                            y_predict[:,1,...] is an estimated std. deviation,
                            y_predict[:,2,...] is a sample_scaling tensor
                            y_predict[:,3,...] is a mask

                """
                # print("Y_true", y_true.shape)
                # y_true = tf.transpose(y_true, perm=[0, 2, 3, 1])
                # print("Y-predict shape", y_predict.shape)
                loss_mean = loss_function_mean(y_true, y_predict)
                loss_sigma = loss_function_sigma(y_true, y_predict)
                loss = alpha * loss_sigma + (1 - alpha) * loss_mean

                return loss

            # loss metric
            if loss_metric == 'Custom':
                assert alpha is not None
                if alpha == 0:
                    # freeze combined lsayers and std. deviation layers
                    # self.combined_layers.trainable = False
                    self.std_deviation_layers.trainable = False
                    self.mean_layers.trainable = True

                elif alpha == 1:
                    # freeze combined layers and mean layers
                    # self.combined_layers.trainable = False
                    self.std_deviation_layers.trainable = True
                    self.mean_layers.trainable = False

                elif alpha == 0.5:
                    # self.combined_layers.trainable = True
                    self.std_deviation_layers.trainable = True
                    self.mean_layers.trainable = True
                else:
                    raise NotImplementedError


            weight_folder = weightfolder + \
                            f"nn_arch_{nn_arch_id}_alpha={alpha}_epochs={epochs}" \
                            f"_samp_scale={self.sample_scaling}" \
                            f"_out_same={self.meas_output_same}"
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weight_folder,
                                                             save_weights_only=save_weight_in_callback,
                                                             verbose=0,
                                                             # mode='min',
                                                             # save_best_only=True
                                                             )

            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            self.compile(optimizer=optimizer, loss=loss_function,
                         # loss_weights=[alpha, 1 - alpha],
                         metrics=[loss_function,
                                  loss_function_mean,
                                  loss_function_sigma,
                                  # loss_function_keras,
                                  ],
                         )

            if test_dataset is not None:
                self.compile(optimizer=optimizer, loss=loss_function,
                             # loss_weights=[alpha, 1 - alpha],
                             metrics=[loss_function,
                                      loss_function_mean,
                                      loss_function_sigma,
                                      # loss_function_keras,
                                      loss_mean_in_sample,
                                      loss_mean_out_sample,
                                      loss_sigma_in_sample,
                                      loss_sigma_out_sample,
                                      ],
                             )

            if test_changes:
                l_initial_weight = [#self.combined_layers.get_weights(),
                                    self.mean_layers.get_weights(),
                                    self.std_deviation_layers.get_weights()]

            if train_dataset is not None:
                history = super().fit(x=train_dataset, epochs=epochs,
                                      validation_data=validation_dataset,
                                      callbacks=[cp_callback, tensorboard_callback], verbose=1)
            if test_dataset is not None:
                assert test_dataset is not None
                history_eval = super().evaluate(x=test_dataset, return_dict=True)
                return history_eval

            vector_alpha = alpha * np.ones(shape=(epochs,))
            vector_alpha.tolist()

            dict_history[f"train_mean_rmse_loss"] += history.history["loss_function_mean"]
            dict_history[f"train_sigma_rmse_error"] += history.history["loss_function_sigma"]
            dict_history[f"val_mean_rmse_loss"] += history.history["val_loss_function_mean"]
            dict_history[f"val_sigma_rmse_error"] += history.history["val_loss_function_sigma"]
            dict_history[f"train_loss"] += history.history["loss"]
            dict_history[f"val_loss"] += history.history["val_loss"]
            dict_history[f"alpha_vector"] += vector_alpha.tolist()

            if test_changes:
                l_final_weight = [#self.combined_layers.get_weights(),
                                  self.mean_layers.get_weights(),
                                  self.std_deviation_layers.get_weights()]

                print("Weight values of Combined layer are the same? ",
                      list_are_close(l_initial_weight[0], l_final_weight[0]))
                print("Weight values of mean layer are the same? ",
                      list_are_close(l_initial_weight[1], l_final_weight[1]))
                print("Weight values of std. layer are the same? ",
                      list_are_close(l_initial_weight[2], l_final_weight[2]))

            if save_weight_in_callback:
                # Save loss metrics for each value of alpha
                lossfolder = loss_folder + \
                             f"nn_arch_{nn_arch_id}_alpha={alpha}_epochs={epochs}" \
                             f"_samp_scale={self.sample_scaling}" \
                             f"_out_same={self.meas_output_same}.pkl"
                outfile = open(lossfolder, 'wb')
                pickle.dump(dict_history, outfile)
                outfile.close()

        return dict_history


class SkipConvNnArch20(SkipConvNn):
    """ Fully Convolutional Neural Network with total 26 layers, 10 convolution,
        10 transpose convolution, 3 max pooling, and 3 upsampling layers.
    """

    def __init__(self, **kwargs):

        super(SkipConvNnArch20, self).__init__(**kwargs)
        self.regularizers = regularizers.l2(0.001)
        # self.combined_layers = self.CombinedLayers()
        self.mean_layers = self.MeanLayers()
        self.std_deviation_layers = self.StdDeviationlayers()

    class MeanLayers(Model):
        def __init__(self):
            super().__init__()
            self.regularizers = None
            # self.regularizers = regularizers.l2(0.001)

            self.conv1 = Conv2D(filters=128,
                                kernel_size=4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv2 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv3 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.max_pool_1 = MaxPool2D(pool_size=(2, 2))

            self.conv4 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv5 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv6 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.max_pool_2 = MaxPool2D(pool_size=(2, 2))

            self.conv1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.conv2T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.conv3T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.up_sampling_1 = UpSampling2D(size=(2, 2))

            self.conv4T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv5T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv6T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.up_sampling_2 = UpSampling2D(size=(2, 2))

            self.conv7 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv7T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv8 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv8T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv9 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv9T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.max_pool_3 = MaxPool2D(pool_size=(2, 2))
            self.up_sampling_3 = UpSampling2D(size=(2, 2))

            self.conv10 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                 kernel_regularizer=self.regularizers)
            self.conv10T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                           kernel_regularizer=self.regularizers)

            self.conv_out_mu = Conv2DTranspose(1, 4,
                                               activation=tf.nn.leaky_relu,
                                               padding='same',
                                               kernel_regularizer=self.regularizers)

        @tf.function
        def call(self, x):
            ## separate branch from this point
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.max_pool_1(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.max_pool_2(x)
            x = self.conv7(x)
            x = self.conv8(x)
            x = self.conv9(x)
            x = self.max_pool_3(x)
            x = self.conv10(x)
            x = self.conv10T(x)
            x = self.up_sampling_3(x)
            x = self.conv9T(x)
            x = self.conv8T(x)
            x = self.conv7T(x)
            x = self.up_sampling_2(x)
            x = self.conv6T(x)
            x = self.conv5T(x)
            x = self.conv4T(x)
            x = self.up_sampling_1(x)
            x = self.conv3T(x)
            x = self.conv2T(x)
            x = self.conv1T(x)

            mu = self.conv_out_mu(x)

            return mu

    class StdDeviationlayers(Model):
        def __init__(self):
            super().__init__()
            self.regularizers = None
            # self.regularizers = regularizers.l2(0.001)
            self.conv1 = Conv2D(filters=128,
                                kernel_size=4,
                                activation=tf.nn.leaky_relu,
                                padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv2 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv3 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.max_pool_1 = MaxPool2D(pool_size=(2, 2))

            self.conv4 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv5 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv6 = Conv2D(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.max_pool_2 = MaxPool2D(pool_size=(2, 2))

            self.conv1_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.conv2_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.conv3_1T = Conv2DTranspose(128, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)

            self.up_sampling_1_1 = UpSampling2D(size=(2, 2))

            self.conv4_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv5_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv6_1T = Conv2DTranspose(256, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.up_sampling_2_1 = UpSampling2D(size=(2, 2))


            self.conv7_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv7_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv8_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv8_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)
            self.conv9_1 = Conv2D(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                kernel_regularizer=self.regularizers)
            self.conv9_1T = Conv2DTranspose(512, 4, activation=tf.nn.leaky_relu, padding='same',
                                          kernel_regularizer=self.regularizers)

            self.max_pool_3_1 = MaxPool2D(pool_size=(2, 2))
            self.up_sampling_3_1 = UpSampling2D(size=(2, 2))

            self.conv10_1 = Conv2D(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                 kernel_regularizer=self.regularizers)
            self.conv10_1T = Conv2DTranspose(1024, 4, activation=tf.nn.leaky_relu, padding='same',
                                           kernel_regularizer=self.regularizers)

            self.conv_out_sigma = Conv2DTranspose(1, 4,
                                                  activation= tf.keras.activations.exponential, #lambda x: tf.nn.elu(x) + 1,
                                                  padding='same',
                                                  kernel_regularizer=self.regularizers)

        @tf.function
        def call(self, y):
            ## separate branch from this point
            y = self.conv1(y)
            y = self.conv2(y)
            y = self.conv3(y)
            y = self.max_pool_1(y)
            y = self.conv4(y)
            y = self.conv5(y)
            y = self.conv6(y)
            y = self.max_pool_2(y)
            y = self.conv7_1(y)
            y = self.conv8_1(y)
            y = self.conv9_1(y)
            y = self.max_pool_3_1(y)
            y = self.conv10_1(y)
            y = self.conv10_1T(y)
            y = self.up_sampling_3_1(y)
            y = self.conv9_1T(y)
            y = self.conv8_1T(y)
            y = self.conv7_1T(y)
            y = self.up_sampling_2_1(y)
            y = self.conv6_1T(y)
            y = self.conv5_1T(y)
            y = self.conv4_1T(y)
            y = self.up_sampling_1_1(y)
            y = self.conv3_1T(y)
            y = self.conv2_1T(y)
            y = self.conv1_1T(y)

            sigma = self.conv_out_sigma(y)

            return sigma