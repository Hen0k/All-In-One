from keras.layers import Layer, Dense, Concatenate
from keras import backend as K
from nets.loss_functions import ordlinal_loss, ordinal_activation
import tensorflow as tf
import numpy as np
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
###########

class AgeBinaryClassifiers(Layer):

    def __init__(self, **kwargs):
        # self.__class__.__name__ = "AgeBinaryClassifiers"
        super(AgeBinaryClassifiers, self).__init__(**kwargs)
        # self.input_tensor = input_tensor

    # def get_output(self, train=False):
    #     X = self.get_input(train)
    #     X = self.input_tensor
    #     arr = [ordinal_activation(i, X) for i in range(128)]
    #     return arr
    def call(self, inputs):
        # l = [ordinal_activation(i, inputs) for i in range(128)]
        
        # arr = Concatenate(axis=-1)(l)
        output = Dense(128, activation="sigmoid")(inputs)
        return output
    def compute_output_shape(self, input_shape):
       
        return [input_shape[0], 128]

    # def get_config(self):
    #     # config = {"name": self.__class__.__name__}
    #     base_config = super(AgeBinaryClassifiers, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))

class SaveCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # serialize model to JSON
        model_json = model.to_json()
        with open("../model/new_model.json", 'w') as json_file:
            json_file.write(model_json)
       # self.model.model.save_weights("../models/"+self.config.large_model_name +  ".h5")
       # mode.save_weights("../models/"+self.config.small_model_name+".h5")
        

# class AgeBinaryClassifiers(Layer):
#     def __init__(self, units, **kwargs):
#         if 'input_shape' not in kwargs and 'input_dim' in kwargs:
#             kwargs['input_shape'] = (kwargs.pop('input_dim'))

#         self.output_dim = units
#         self.activation = ordinal_activation#(ord_numer, self.units)
#         # self.build((1, 128))
#         super(AgeBinaryClassifiers, self).__init__(**kwargs)

#     def build(self, input_shape):
#         self.kernel = self.add_weight(name='kernel',
#                                       shape=(input_shape[1], self.output_dim),
#                                       initializer='uniform',
#                                       trainable=False)

#         super(AgeBinaryClassifiers, self).build(input_shape)

#     def call(self, X):
#         # assert isinstance(X, list), "The input to the layer is not a List"
#         arr = [self.activation(i, X) for i in range(128)]

#         return arr
    
    # def compute_output_shape(self, input_shape):
    #     # assert isinstance(input_shape, list)
    #     shape_a, shape_b = input_shape
    #     return [(shape_a[0], self.output_dim), shape_b[:-1]]
