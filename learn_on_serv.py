import logging
import model_factory

from tensorflow.python.keras.callbacks import CSVLogger
import tensorflow as tf
from constants.constant import *
from model.iterator import MyIterator
from pathlib import Path


# logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w")


class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, monitor1, monitor2):
        super(CustomEarlyStopping, self).__init__()
        self.monitor1 = monitor1
        self.monitor2 = monitor2
        self.stale_epochs = 0
        self.previous_accuracy = None
        self.threshold = 1e-3

    def on_epoch_end(self, epoch, logs=None):
        errors = logs.get(self.monitor1)
        accuracy = logs.get(self.monitor2)

        if errors == 0:
            self.model.stop_training = True

        if self.previous_accuracy is not None and abs(accuracy - self.previous_accuracy) < self.threshold:
            self.stale_epochs += 1
        else:
            self.stale_epochs = 0

        self.previous_accuracy = accuracy

        if self.stale_epochs >= 50:
            self.model.stop_training = True

        return


if __name__ == "__main__":
    current_directory = Path.cwd()
    try:
        for param in LIST_OF_PARAMS:
            try:
                iterator = MyIterator(param.n, param.k)
                num_samples = 2 ** param.n

                input_shape = (param.n,)
                hidden_size = param.n

                list_of_models = []
                for name_of_model in LIST_OF_NAMES_OF_MODELS:
                    uncompiled_model = model_factory.new_model(
                        name_of_model, input_shape, param.layers, hidden_size
                    )
                    uncompiled_model.compile()
                    list_of_models.append(uncompiled_model)

                try:
                    for model in list_of_models:
                        try:
                            csv_logger = CSVLogger(
                                current_directory / "logs" / f"{model.name_model}-{param.n}-{param.k}",
                                separator=',', append=True)
                            early_stopping = CustomEarlyStopping(monitor1='errors', monitor2='binary_accuracy')

                            model.fit(iterator, steps_per_epoch=num_samples, epochs=100000,
                                      callbacks=[csv_logger, early_stopping])

                            model.save_weights(
                                current_directory / "weights" / f"{model.name_model}-{param.n}-{param.k}.h5")
                        except Exception as err:
                            logging.error(f"Error while processing model = {model}. Reason: {err}")

                except Exception as err:
                    logging.error(f"Error while processing models. Reason: {err}")

            except Exception as err:
                logging.error(f"Error while processing combination n = {param.n}, k = {param.k}. Reason: {err}")

    except Exception as err:
        logging.error(f"Error while whole process. Reason: {err}")
