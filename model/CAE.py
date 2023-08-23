import keras.backend as K
from model.AE import AE, ErrorsMetric
from constants import constant


class CAE(AE):
    def __init__(self, input_shape, input_layers, output_layers, hidden_size, alpha=constant.ALPHA,
                 activation=constant.ACTIVATION):
        super().__init__(input_shape, input_layers - 1, output_layers, hidden_size, alpha, activation)

    def compile(self, optimizer=constant.OPTIMIZER):
        self.model.compile(optimizer=optimizer, loss=self.contractive_loss, metrics=['binary_accuracy', ErrorsMetric()])

    def jaccard_coef(self, y_true, y_pred, smooth=1e-12):
        inter = K.sum(K.abs(y_true * y_pred), axis=-1)
        union = K.sum(K.square(y_true), axis=-1) + K.sum(K.square(y_pred), axis=-1) - inter
        return K.mean((inter + smooth) / (union + smooth))

    def get_jacobian(self, model, x):
        """Вычисляем Якобиан выхода относительно входа (для каждого входного образца)."""
        jacobian = K.stack([K.gradients(model.output[:, i], model.input)[0]
                            for i in range(model.output_shape[1])])
        return jacobian

    def contractive_loss(self, y_true, y_pred):
        """Вычисляем Contractive Loss для CAE."""
        mse = K.mean(K.square(y_true - y_pred))
        jacobian = self.get_jacobian(self.model, self.input_tensor)
        jacobian_norm = K.sqrt(K.sum(K.square(jacobian), axis=(2, 3)))
        contractive_loss = constant.CONTRACTIVE_LOSS_WEIGHT * K.mean(K.square(jacobian_norm))
        return mse + contractive_loss
