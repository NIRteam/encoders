import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, Input

from test.Exit import ExitCallback
from test.er import CustomMetric

ENCODED_LEN = 3
DATA_LEN = 5
NEURONS = 64
BIG_ENCODED_LEN = 9
BIG_ENCODED_OUT = 2


def binary_generator():
    n = DATA_LEN

    while True:
        current = [0] * n
        while current is not None:
            result = current.copy()
            # Увеличение текущего числа на 1
            i = n - 1
            while i >= 0 and current[i] == 1:
                current[i] = 0
                i -= 1
            if i == -1:
                current = None
            else:
                current[i] = 1

            res = np.array(result).reshape((1, -1))
            yield res, res


input_data = Input(shape=(DATA_LEN))
x = Flatten()(input_data)
x = Dense(NEURONS, activation='tanh')(x)
encoded = Dense(ENCODED_LEN, activation='tanh')(x)

input_dec = Input(shape=(ENCODED_LEN))
d = Dense(NEURONS, activation='tanh')(input_dec)
d = Dense(NEURONS * 2, activation='tanh')(d)
d = Dense(NEURONS * 4, activation='tanh')(d)
decoded = Dense(DATA_LEN, activation='tanh')(d)

exitCallback = ExitCallback()
encoder = keras.Model(input_data, encoded, name="encoder")
decoder = keras.Model(input_dec, decoded, name="decoder")

autoencoder = keras.Model(input_data, decoder(encoder(input_data)), name="autoencoder")
autoencoder.compile(optimizer='adam', metrics=['binary_accuracy', CustomMetric()], loss='mse')
autoencoder.summary()

autoencoder.fit(binary_generator(),
                steps_per_epoch=2 ** DATA_LEN,
                epochs=10000)

for a in binary_generator():
    autoencoder.predict(tf.expand_dims(a[0], axis=0))

# encoder.save('model/encoder')
# decoder.save('model/decoder')
# autoencoder.save('model/autoencoder')
