import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, LeakyReLU
from constants import constant
from model.ErrorsMetric import ErrorsMetric

class VAE(Model):
    def __init__(self, input_shape, num_layers, hidden_size, alpha=constant.ALPHA, activation=constant.ACTIVATION, name_model="VAE"):
        super(VAE, self).__init__()
        self.my_input_shape = input_shape
        self.input_tensor = Input(shape=input_shape)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.name_model = name_model
        self.alpha = alpha
        self.activation = activation
        self.list_layer = {}
        self.model = self.create_encoder_decoder()

    def create_encoder_decoder(self):
        encoder_output = self.create_encoder()
        decoder_output = self.create_decoder(encoder_output)

        model = Model(self.input_tensor, decoder_output, name=self.name_model)
        return model

    def create_encoder(self):
        encoder_input = self.input_tensor
        encoder_layer = Dense(self.hidden_size)(encoder_input)
        encoder_layer = LeakyReLU(alpha=self.alpha)(encoder_layer)

        for i in range(1, self.num_layers):
            new_hidden_size = int(self.hidden_size / (2 ** i))
            encoder_layer = Dense(new_hidden_size)(encoder_layer)
            encoder_layer = LeakyReLU(alpha=self.alpha)(encoder_layer)

        z_mean = Dense(self.hidden_size, name="z_mean")(encoder_layer)
        z_log_var = Dense(self.hidden_size, name="z_log_var")(encoder_layer)

        return [encoder_input, z_mean, z_log_var]

    def create_decoder(self, encoder_output):
        encoder_input, z_mean, z_log_var = encoder_output
        batch_size = tf.shape(z_mean)[0]
        epsilon = tf.random.normal(shape=(batch_size, self.hidden_size), mean=0., stddev=1.0)
        z_mean = tf.cast(z_mean, dtype=tf.float32)
        z_log_var = tf.cast(z_log_var, dtype=tf.float32)
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        decoder_layer = z

        for i in range(1, self.num_layers):
            new_hidden_size = int(self.hidden_size / (2 ** i))
            decoder_layer = Dense(new_hidden_size)(decoder_layer)
            decoder_layer = LeakyReLU(alpha=self.alpha)(decoder_layer)

        decoder_output = Dense(self.my_input_shape[0], activation=self.activation)(decoder_layer)

        return decoder_output

    def summary(self):
        self.model.summary()

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)

    def compile(self, optimizer=constant.OPTIMIZER, loss=constant.LOSS, metrics=None, loss_weights=None,
                weighted_metrics=None, run_eagerly=None, **kwargs):
        super(VAE, self).compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy', ErrorsMetric()],
                                loss_weights=loss_weights, weighted_metrics=weighted_metrics,
                                run_eagerly=run_eagerly, **kwargs)