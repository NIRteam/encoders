from keras.metrics import Metric
import keras.backend as K
import tensorflow as tf


class CustomMetric(Metric):
    def __init__(self, name='custom_metric', **kwargs):
        super(CustomMetric, self).__init__(name=name, **kwargs)
        self.num_diffs = self.add_weight(name='num_diffs', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = K.cast(K.round(y_pred), dtype='int64')
        # Count the number of different elements between y_true and y_pred
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
