from constants.constant import *
from model.AE import AE
from model.CAE import CAE
from model.MAE import MAE
from model.VAE import VAE


def new_model(name, input_shape, num_layers, hidden_size):
    if name == Models.AE.value:
        model = AE(input_shape=input_shape, num_layers=num_layers, hidden_size=hidden_size)
    elif name == Models.VAE.value:
        model = VAE(input_shape=input_shape, num_layers=num_layers, hidden_size=hidden_size)
    elif name == Models.MAE.value:
        model = MAE(input_shape=input_shape, num_layers=num_layers, hidden_size=hidden_size)
    elif name == Models.CAE.value:
        model = CAE(input_shape=input_shape, num_layers=num_layers, hidden_size=hidden_size)

    return model
