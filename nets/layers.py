from keras.layers import Layer, Dense
from keras import backend as K
from nets.loss_functions import ordlinal_loss, ordinal_activation
import tensorflow as tf

class RoundLayer(Layer):
    def __init__(self, **kwargs):
        super(RoundLayer, self).__init__(**kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        return K.round(X)

    def get_config(self):
        config = {"name": self.__class__.__name__}
        base_config = super(RoundLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AgeBinaryClassifiers(Layer):
    def __init__(self, **kwargs):
        super(AgeBinaryClassifiers, self).__init__(**kwargs)
    def call(self, layer_input):
        arr = tf.constant_initializer()
        for i in range(128):
            tf.concat([arr, Dense(1, activation=ordinal_activation(i, layer_input))(layer_input)], 1)
        return arr   
        