from tensorflow.python.keras.layers import Dense, LeakyReLU, Input, Masking
from tensorflow.python.keras.models import Model
from model.ErrorsMetric import ErrorsMetric
from constants import constant


class MAE(Model):
    def __init__(self, input_shape, num_layers, hidden_size, alpha=constant.ALPHA, activation=constant.ACTIVATION, name_model="MAE"):
        super(MAE, self).__init__()
        self.my_input_shape = input_shape
        self.input_tensor = Input(shape=input_shape)
        self.num_layers = num_layers
        self.name_model = name_model
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.activation = activation
        self.list_layer = {}
        self.model = self.create_encoder_decoder()

    def create_encoder_decoder(self):
        encoder_layer = self.create_encoder()
        decoder_layer = self.create_decoder(encoder_layer)

        model = Model(self.input_tensor, decoder_layer, name=self.name_model)
        return model

    def create_encoder(self):
        encoder_input = self.input_tensor
        encoder_input = Dense(self.hidden_size)(encoder_input)
        encoder_input = LeakyReLU(alpha=self.alpha)(encoder_input)
        self.list_layer[self.hidden_size] = self.my_input_shape[0] * self.hidden_size + self.hidden_size
        for i in range(1, self.num_layers):
            new_hidden_size = int(self.hidden_size / (2 ** i))
            encoder_input = Dense(new_hidden_size)(encoder_input)
            encoder_input = LeakyReLU(alpha=self.alpha)(encoder_input)
            self.list_layer[new_hidden_size] = self.my_input_shape[0] * new_hidden_size + new_hidden_size
        return encoder_input

    def create_decoder(self, encoder_layer):
        decoder_input = encoder_layer
        reversed_dict = dict(reversed([(key, value) for key, value in self.list_layer.items()]))
        for hidden, _ in reversed_dict.items():
            decoder_input = Dense(units=hidden)(decoder_input)
            decoder_input = LeakyReLU(alpha=self.alpha)(decoder_input)

        # Добавляем слой Masking для игнорирования нулевых значений во входных данных
        masked_input = Masking(mask_value=0.0)(self.input_tensor)
        decoder_output = decoder_input * masked_input  # Подключаем декодер к маске

        return decoder_output

    def compile(self, optimizer=constant.OPTIMIZER, loss=constant.LOSS, metrics=None, loss_weights=None,
                weighted_metrics=None, run_eagerly=None, **kwargs):
        super(MAE, self).compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy', ErrorsMetric()],
                                loss_weights=loss_weights, weighted_metrics=weighted_metrics,
                                run_eagerly=run_eagerly, **kwargs)

    def summary(self):
        self.model.summary()

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)
