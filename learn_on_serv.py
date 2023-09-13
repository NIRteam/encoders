import model_factory
from keras.callbacks import CSVLogger, EarlyStopping

from constants.constant import INPUT_LAYERS, OUTPUT_LAYERS, LIST_OF_PARAMS, LIST_OF_NAMES_OF_MODELS, PATH_TO_LOGGER, \
    PATH_TO_WEIGHTS
from model.iterator import MyIterator


class ParamsForBCH:
    def __init__(self, k, n):
        self.k = k
        self.n = n


try:
    for param in LIST_OF_PARAMS:
        iterator = MyIterator(param.k, param.n)
        num_samples = 2 ** param.n

        input_shape = (param.n,)
        hidden_size = param.n + 1

        list_of_models = []
        for name_of_model in LIST_OF_NAMES_OF_MODELS:
            uncompiled_model = model_factory.new_model(
                name_of_model, input_shape, INPUT_LAYERS, OUTPUT_LAYERS, hidden_size
            )
            list_of_models.append(uncompiled_model.compile())

        try:
            for model in list_of_models:
                csv_logger = CSVLogger(PATH_TO_LOGGER, separator=',', append=True)
                early_stopping = EarlyStopping(monitor='errors', mode='auto', min_delta=0, patience=0,
                                               restore_best_weights=True, baseline=0)

                history = model.fit(iterator, steps_per_epoch=num_samples, epochs=1,
                                    callbacks=[csv_logger, early_stopping])

                model.save_weights(PATH_TO_WEIGHTS)
        except Exception as err:
            # TODO: добавить логирование
            pass

except Exception as err:
    # TODO: добавить логирование
    pass
