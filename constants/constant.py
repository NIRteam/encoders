from enum import Enum

from learn_on_serv import ParamsForBCH

LOSS = "MSE"
OPTIMIZER = "Adam"
ACTIVATION = "sigmoid"
ALPHA = 0.2
LAM = 1e-4
CONTRACTIVE_LOSS_WEIGHT = 1

# for server learning
INPUT_LAYERS = 5

OUTPUT_LAYERS = 2

# TODO заполнить табличку для БЧХ
LIST_OF_PARAMS = [ParamsForBCH(4, 7)]

LIST_OF_NAMES_OF_MODELS = ['ae', 'vae', 'cae', 'mae']

PATH_TO_LOGGER = f''

PATH_TO_WEIGHTS = f''


class Models(Enum):
    AE = 'ae'
    VAE = 'vae'
    CAE = 'cae'
    MAE = 'mae'
