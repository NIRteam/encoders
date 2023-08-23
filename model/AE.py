import math

from keras import Input, Model
from keras.layers import LeakyReLU, Dense
from keras.metrics import Metric
import keras.backend as K
import tensorflow as tf
from constants import constant


class AE(Model):
    def __init__(self, input_shape, input_layers, output_layers, hidden_size, alpha=constant.ALPHA,
                 activation=constant.ACTIVATION):
        super(AE, self).__init__()
        self.my_input_shape = input_shape
        self.input_tensor = Input(shape=input_shape)
        self.input_layers = input_layers
        self.output_layers = output_layers
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.activation = activation
        self.model = self.create_encoder_decoder()

    def create_encoder_decoder(self):
        # Создаем первый слой энкодера
        encoder_layer = self.create_first_layer()

        # Создаем дополнительные слои энкодера и применяем функцию активации LeakyReLU
        for i in range(self.input_layers - 1):
            self.hidden_size //= 2
            encoder_layer = Dense(int(math.floor(self.hidden_size)))(encoder_layer)
            encoder_layer = LeakyReLU(alpha=self.alpha)(encoder_layer)

        # Сохраняем выход последнего слоя энкодера
        output_layer = encoder_layer

        # Создаем слои декодера и применяем указанную функцию активации
        for i in range(self.output_layers - 1):
            self.hidden_size *= 2
            output_layer = Dense(self.hidden_size, activation=self.activation)(output_layer)
        # Создаем финальный слой декодера
        output_layer = Dense(self.my_input_shape[0], activation=self.activation)(output_layer)

        model = Model(self.input_tensor, output_layer)

        return model

    def create_first_layer(self):
        first_layer = Dense(self.hidden_size)(self.input_tensor)
        first_layer = LeakyReLU(alpha=self.alpha)(first_layer)
        return first_layer

    def compile(self, optimizer=constant.OPTIMIZER, loss=constant.LOSS):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy', ErrorsMetric()])

    def summary(self):
        self.model.summary()


class ErrorsMetric(Metric):
    def __init__(self, name='errors', **kwargs):
        super(ErrorsMetric, self).__init__(name=name, **kwargs)
        self.num_diffs = self.add_weight(name='num_diffs', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = K.cast(K.round(y_pred), dtype='int64')
        equal_bool_arr = K.not_equal(y_true, y_pred)
        equal_int_arr = K.cast(equal_bool_arr, dtype='int64')

        res = K.cast(self.num_diffs, dtype='int64')
        for i in equal_int_arr:
            error_count = K.sum(K.cast(i, dtype='int64'))
            if tf.greater(error_count, res):
                res = error_count

        self.num_diffs.assign(K.cast(res, dtype='float32'))

    def result(self):
        return self.num_diffs

    def reset_states(self):
        self.num_diffs.assign(0.0)
