# Copyright (c) 2018 Yuhang Lu <luyuhang@iie.ac.cn>

from lib import graph
import tensorflow as tf
import scipy.sparse
import numpy as np
from  keras.initializers import RandomUniform
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import MaxPooling1D
from keras import activations

class MyLayer(Layer):

    def __init__(self,output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.w_rlu = self.add_weight(name='w_rlu',shape=([1,1,self.output_dim]),initializer=self.my_init,trainable=True)
        # self.w_rlu = self.add_weight(name='kernel',shape=([3,1], self.output_dim),initializer='uniform',trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        g0 = scipy.sparse.csr_matrix(self.build_graph()).astype(np.float32)
        graphs0 = []
        for i in range(3):
            graphs0.append(g0)
        L = [graph.laplacian(A, normalized=True) for A in graphs0]
        for i in range(2):
            with tf.variable_scope('gcn{}'.format(i + 1)):
                with tf.name_scope('filter'):
                    x = self.my_gcn(x, L[i], str(i))
                with tf.name_scope('bias_relu'):
                    x = self.brelu(x)
                with tf.name_scope('pooling'):
                    x = MaxPooling1D(pool_size=1,padding='same')(x)
        x = K.squeeze(x,2)
        return x
    def my_gcn(self,x,L,th):
        Fout,neibs = 1,3
        _ , M, Fin = x.get_shape()
        M, Fin = int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, -1])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_],0)  # K x M x Fin*N
        if neibs > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, neibs):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [neibs, M, Fin, -1])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [-1, Fin*neibs])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        # W = _weight_variable([Fin*K, Fout])
        # x = tf.matmul(x, W)  # N*M x Fout
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform_initializer(minval=-init_range, maxval=init_range)
        self.w_gcn = self.add_weight(name='w_gcn_'+th,shape=([Fin*neibs,self.output_dim]),initializer=self.my_init,trainable=True)
        x = K.dot(x,self.w_gcn)
        return tf.reshape(x, [-1, M, Fout])  # N x M x Fout

    def my_init(self):
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = RandomUniform(minval=-init_range, maxval=init_range)
        return initial

    def brelu(self,x):
        return activations.relu(x + self.w_rlu)

    def build_graph(self):
        filename = '../data/gcn/txt_graph.txt'
        file = open(filename).read().strip().split('\n')
        graph=np.zeros((len(file),len(file)),dtype=np.int)
        y = 0
        for line in file:
            new_col=np.array(line.split())
            for i in new_col:
                graph[y][int(i)]=1
            y += 1
        return graph
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1])

