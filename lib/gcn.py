# Copyright (c) 2018 Yuhang Lu <luyuhang@iie.ac.cn>

from lib import graph
import tensorflow as tf
import scipy.sparse
import numpy as np

def build(x):
    x = tf.expand_dims(x,2)
    g0 = scipy.sparse.csr_matrix(build_graph()).astype(np.float32)
    graphs0 = []
    for i in range(3):
        graphs0.append(g0)
    L = [graph.laplacian(A, normalized=True) for A in graphs0]
    for i in range(2):
        with tf.variable_scope('gcn{}'.format(i + 1)):
            with tf.name_scope('filter'):
                x = my_gcn(x, L[i])
            with tf.name_scope('bias_relu'):
                x = brelu(x)
            with tf.name_scope('pooling'):
                x = mpool(x,1)
    x = tf.squeeze(x)
    return x

def text_gcn(x,L):

    Fout,K = 1,3
    N, M, Fin = x.get_shape()
    N, M, Fin = int(N), int(M), int(Fin)
    # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
    L = scipy.sparse.csr_matrix(L)
    L = graph.rescale_L(L, lmax=2)
    L = L.tocoo()
    indices = np.column_stack((L.row, L.col))
    L = tf.SparseTensor(indices, L.data, L.shape)
    L = tf.sparse_reorder(L)
    # Transform to Chebyshev basis
    x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
    x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
    x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
    def concat(x, x_):
        x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
        return tf.concat([x, x_],0)  # K x M x Fin*N
    if K > 1:
        x1 = tf.sparse_tensor_dense_matmul(L, x0)
        x = concat(x, x1)
    for k in range(2, K):
        x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
        x = concat(x, x2)
        x0, x1 = x1, x2
    x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
    x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
    x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
    # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
    W = _weight_variable([Fin*K, Fout])
    x = tf.matmul(x, W)  # N*M x Fout
    return tf.reshape(x, [N, M, Fout])  # N x M x Fout

def my_gcn(x,L):
    Fout,K = 1,3
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
    if K > 1:
        x1 = tf.sparse_tensor_dense_matmul(L, x0)
        x = concat(x, x1)
    for k in range(2, K):
        x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
        x = concat(x, x2)
        x0, x1 = x1, x2
    x = tf.reshape(x, [K, M, Fin, -1])  # K x M x Fin x N
    x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
    x = tf.reshape(x, [-1, Fin*K])  # N*M x Fin*K
    # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
    W = _weight_variable([Fin*K, Fout])
    x = tf.matmul(x, W)  # N*M x Fout
    return tf.reshape(x, [-1, M, Fout])  # N x M x Fout


def build_graph():
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

def mpool(x, p):
    """Max pooling of size p. Should be a power of 2."""
    if p > 1:
        x = tf.expand_dims(x, 3)  # N x M x F x 1
        x = tf.nn.max_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
        #tf.maximum
        return tf.squeeze(x, [3])  # N x M/p x F
    else:
        return x

def _weight_variable(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform_initializer(minval=-init_range, maxval=init_range)
    var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
    return var

def _bias_variable(shape):
    #initial = tf.constant_initializer(0.1)
    initial = tf.constant_initializer(0.0)
    var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
    return var

def brelu(x,relu=True):
    """Bias and ReLU (if relu=True). One bias per filter."""
    N, M, F = x.get_shape()
    b = _bias_variable([1, 1, int(F)])
    x = x + b
    return tf.nn.relu(x) if relu else x
