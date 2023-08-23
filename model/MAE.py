from keras.layers import Dense, LeakyReLU, Masking

from model.AE import AE
from constants import constant


class MAE(AE):
    def __init__(self, input_shape, input_layers, output_layers, hidden_size, alpha=constant.ALPHA,
                 activation=constant.ACTIVATION):
        super().__init__(input_shape, input_layers - 1, output_layers, hidden_size, alpha, activation)

    def create_first_layer(self):
        masking_layer = Masking(mask_value=0.0)(self.input_tensor)
        first_layer = Dense(self.hidden_size)(masking_layer)
        first_layer = LeakyReLU(alpha=self.alpha)(first_layer)
        return first_layer
