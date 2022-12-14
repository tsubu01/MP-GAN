from numpy import hstack
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Reshape
from keras.layers import LeakyReLU
from keras.layers import Conv2DTranspose
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Dropout

from tensorflow.keras.optimizers import Adam, SGD
from matplotlib import pyplot
import matplotlib


class DeepModel():
    def __init__(self, n_inputs, layers):
        self.n_inputs = n_inputs
        self.layers = layers
    
    def create_model(self):
        model = Sequential()
        for layer in self.layers:
            
            if layer['type'] == 'Dense':
                model.add(Dense(layer['nodes'],
                                activation=layer['activation'],
                                kernel_initializer='he_uniform',
                                use_bias=layer['use_bias'],
                                input_dim=self.n_inputs))
            
            elif layer['type'] == 'Conv':
                if 'input_shape' in layer:
                    model.add(Conv2D(filters=layer['filters'],
                                      kernel_size=layer['kernel'],
                                      activation = layer['activation'],
                                      strides=layer['strides'],
                                      kernel_regularizer=layer['kernel_reg'],
                                      input_shape=layer['input_shape']))
                else:
                    model.add(Conv2D(filters=layer['filters'],
                                      kernel_size=layer['kernel'],
                                      activation = layer['activation'],
                                      strides=layer['strides'],
                                      padding=layer['padding'],
                                      kernel_regularizer=layer['kernel_reg'],
                                      use_bias=layer['use_bias']))
            
            elif layer['type'] == 'Reshape':
                model.add(Reshape((layer['newshape'])))
            
            elif layer['type'] == 'Deconv':
                model.add(Conv2DTranspose(filters=layer['filters'],
                                          kernel_size=layer['kernel'],
                                          strides=layer['strides'],
                                          padding=layer['padding'],
                                          use_bias=layer['use_bias']))
            
            elif layer['type'] == 'globav':
                model.add(GlobalAveragePooling2D())
            
            elif layer['type'] == 'dropout':
                model.add(Dropout(layer['rate']))
            
            elif layer['type'] == 'batchnorm':
                model.add(BatchNormalization())
                
            elif layer['type'] == 'flatten':
                model.add(Flatten())
            
            elif layer['type'] == 'leakyrelu':
                model.add(LeakyReLU())
            
            elif layer['type'] == 'scaler':
                model.add(tf.keras.layers.Rescaling(layer['scale'], offset=layer['offset']))
                
        return model
    
    
