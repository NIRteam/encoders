from enum import Enum

LOSS = "MSE"
OPTIMIZER = "Adam"
ACTIVATION = "sigmoid"
ALPHA = 0.2
LAM = 1e-4
CONTRACTIVE_LOSS_WEIGHT = 1

# for server learning
INPUT_LAYERS = 5

OUTPUT_LAYERS = 2


class ParamsForBCH:
    def __init__(self, k, n, layers):
        self.k = k
        self.n = n
        self.layers = layers


LIST_OF_PARAMS = [ParamsForBCH(4, 7, 3), ParamsForBCH(11, 15, 4), ParamsForBCH(7, 15, 4), ParamsForBCH(5, 15, 4),
                  ParamsForBCH(26, 31, 4), ParamsForBCH(21, 31, 4), ParamsForBCH(16, 31, 4), ParamsForBCH(11, 31, 4),
                  ParamsForBCH(6, 31, 4)]
#   ParamsForBCH(57, 63), ParamsForBCH(51, 63), ParamsForBCH(45, 63),
# ParamsForBCH(39, 63), ParamsForBCH(36, 63), ParamsForBCH(30, 63), ParamsForBCH(24, 63),
# ParamsForBCH(18, 63), ParamsForBCH(16, 63), ParamsForBCH(10, 63), ParamsForBCH(7, 63),
# ParamsForBCH(120, 127), ParamsForBCH(113, 127), ParamsForBCH(106, 127), ParamsForBCH(99, 127),
# ParamsForBCH(92, 127), ParamsForBCH(85, 127), ParamsForBCH(78, 127), ParamsForBCH(71, 127),
# ParamsForBCH(64, 127), ParamsForBCH(57, 127), ParamsForBCH(50, 127), ParamsForBCH(43, 127),
# ParamsForBCH(36, 127), ParamsForBCH(29, 127), ParamsForBCH(22, 127), ParamsForBCH(15, 127),
# ParamsForBCH(8, 127), ParamsForBCH(247, 255), ParamsForBCH(239, 255), ParamsForBCH(231, 255),
# ParamsForBCH(223, 255), ParamsForBCH(215, 255), ParamsForBCH(207, 255), ParamsForBCH(199, 255),
# ParamsForBCH(191, 255), ParamsForBCH(187, 255), ParamsForBCH(179, 255), ParamsForBCH(171, 255),
# ParamsForBCH(163, 255), ParamsForBCH(155, 255), ParamsForBCH(147, 255), ParamsForBCH(139, 255),
# ParamsForBCH(131, 255), ParamsForBCH(123, 255), ParamsForBCH(115, 255), ParamsForBCH(107, 255),
# ParamsForBCH(99, 255), ParamsForBCH(91, 255), ParamsForBCH(87, 255), ParamsForBCH(79, 255),
# ParamsForBCH(71, 255), ParamsForBCH(63, 255), ParamsForBCH(55, 255), ParamsForBCH(47, 255),
# ParamsForBCH(45, 255), ParamsForBCH(37, 255), ParamsForBCH(29, 255), ParamsForBCH(21, 255),
# ParamsForBCH(13, 255), ParamsForBCH(9, 255)]

LIST_OF_NAMES_OF_MODELS = ['ae', 'vae', 'cae', 'mae']

PATH_TO_LOGGER = f'C:\\Git\\encoders'

PATH_TO_WEIGHTS = f'C:\\Git\\encoders'


class Models(Enum):
    AE = 'ae'
    VAE = 'vae'
    CAE = 'cae'
    MAE = 'mae'
