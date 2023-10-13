import keras.backend as K
from model.AE import AE
import tensorflow as tf
from model.ErrorsMetric import ErrorsMetric
from constants import constant


class CAE(AE):
    def __init__(self, input_shape, num_layers, hidden_size, alpha=constant.ALPHA, activation=constant.ACTIVATION,
                 name_model="CAE"):
        super().__init__(input_shape, num_layers, hidden_size, alpha, activation, name_model)

    def compile(self, optimizer=constant.OPTIMIZER, loss=None, metrics=None, loss_weights=None, weighted_metrics=None,
                run_eagerly=None, **kwargs):
        super(AE, self).compile(optimizer=optimizer, loss=self.contractive_loss,
                                metrics=['binary_accuracy', ErrorsMetric()],
                                loss_weights=loss_weights, weighted_metrics=weighted_metrics, run_eagerly=run_eagerly,
                                **kwargs)

    def contractive_loss(self, y_true, y_pred):
        """Вычисляем Contractive Loss для CAE."""
        y_true = tf.cast(y_true, tf.float32)
        mse = K.mean(K.square(y_true - y_pred))
        coefficient = 0.01
        return mse + coefficient

    def jaccard_coef(self, y_true, y_pred, smooth=1e-12):
        inter = K.sum(K.abs(y_true * y_pred), axis=-1)
        union = K.sum(K.square(y_true), axis=-1) + K.sum(K.square(y_pred), axis=-1) - inter
        return K.mean((inter + smooth) / (union + smooth))
