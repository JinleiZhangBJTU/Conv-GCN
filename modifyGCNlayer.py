from __future__ import print_function
import tensorflow as tf
from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K
from utils import calculate_laplacian

class GraphConvolution1(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self, units, support=1,
                 activation=None,
                 use_bias=True,
                 adj = None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution1, self).__init__(**kwargs)
        self.adj1 = calculate_laplacian(adj)
        self.adj = tf.sparse_tensor_to_dense(self.adj1)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

        self.support = support
        # assert support >= 1



    def compute_output_shape(self, input_shapes):
        # features_shape = input_shapes[0]
        output_shape = (None, input_shapes[1], self.units)
        return output_shape  # (batch_size, output_dim)

    def build(self, input_shapes):
        # features_shape = input_shapes[0]#<tf.Tensor 'input_2:0' shape=(?, 276) dtype=float32>

        input_dim = input_shapes[2] # 14
        self.kernel = self.add_weight(shape=(input_dim * self.support,#（15*1,14）（14*1,1）
                                             self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        # features = inputs[0]
        # basis = inputs[1:]
        supports = list()
        # for i in range(self.support):
        #     supports.append(inputs)
        #     # supports.append(K.dot(basis[i], features))
        # supports = K.concatenate(supports, axis=1)
        # print(self.adj.shape)
        supports = K.dot(self.adj, inputs)
        supports = tf.transpose(supports,perm=[1,0,2])
        output = K.dot(supports, self.kernel)  #(276, 15)*(15*16) (276 * 16)* (16*1)
        if self.bias:
            output += self.bias
        return self.activation(output)

    def get_config(self):
        config = {'units': self.units,
                  'support': self.support,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        base_config = super(GraphConvolution1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))