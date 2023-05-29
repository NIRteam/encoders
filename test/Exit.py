import tensorflow as tf


class ExitCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('binary_accuracy') >= 1.0:
            # if logs.get('loss') <= 0.0001:

            print("\nWe have reached %2.2f%% accuracy, so we will stopping training." % (1.0 * 100))
            self.model.stop_training = True
