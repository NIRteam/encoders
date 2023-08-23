from enum import Enum

LOSS = "MSE"
OPTIMIZER = "Adam"
ACTIVATION = "sigmoid"
ALPHA = 0.2
LAM = 1e-4
CONTRACTIVE_LOSS_WEIGHT = 1


class Models(Enum):
    AE = 'ae'
    VAE = 'vae'
    CAE = 'cae'
    MAE = 'mae'
