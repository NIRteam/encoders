import math

from keras import Model
from keras.layers import Dense, LeakyReLU, Lambda
import keras.backend as K
from model.AE import AE
from constants import constant
import tensorflow as tf


class VAE(AE):
    def __init__(self, input_shape, input_layers, output_layers, hidden_size, alpha=constant.ALPHA,
                 activation=constant.ACTIVATION):
        super().__init__(input_shape, input_layers, output_layers, hidden_size, alpha, activation)

    def create_encoder_decoder(self):
        # Создаем первый слой энкодера
        encoder_layer = self.create_first_layer()

        # Создаем дополнительные слои энкодера и применяем функцию активации LeakyReLU
        for i in range(self.input_layers - 3):
            self.hidden_size //= 2
            encoder_layer = Dense(int(math.floor(self.hidden_size)))(encoder_layer)
            encoder_layer = LeakyReLU(alpha=self.alpha)(encoder_layer)

        z_mean = Dense(self.hidden_size)(encoder_layer)
        z_log_var = Dense(self.hidden_size)(encoder_layer)
        z = Lambda(self.sampling)([z_mean, z_log_var])

        # Сохраняем выход последнего слоя энкодера
        output_layer = encoder_layer

        # Создаем слои декодера и применяем указанную функцию активации
        for i in range(self.output_layers - 1):
            self.hidden_size *= 2
            output_layer = Dense(self.hidden_size, activation=self.activation)(output_layer)
        # Создаем финальный слой декодера
        output_layer = Dense(self.my_input_shape[0], activation=self.activation)(output_layer)

        model = Model(self.input_tensor, output_layer)

        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        model.add_loss(K.mean(kl_loss) * constant.LAM)

        return model

    def sampling(self, args):
        z_mean, z_log_var = args
        batch_size = tf.shape(z_mean)[0]
        epsilon = tf.random.normal(shape=(batch_size, self.hidden_size), mean=0., stddev=1.)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
