# Copyright (c) 2018 Yuhang Lu <luyuhang@iie.ac.cn>

from lib import gcn_utils,graph,gcn,mAP
import numpy as np
import os,time,datetime
import tensorflow as tf
from keras import backend as K
import keras
from keras.engine.topology import Layer
import keras.layers as layers
from keras.models import Model
from keras.callbacks import ModelCheckpoint,Callback
#GPU 控制
filter_sizes,filters = [2,3,4,5],250
maxlen = 400

def build(txt_shape,img_shape,act_1=None,act_2=None,loss_function=mAP.my_loss):
    K.clear_session()
    input_text = layers.Input(shape=(txt_shape[1],txt_shape[2]))
    input_image = layers.Input(shape=(img_shape,))
    # ------- lstm ----------- 
    text_mat = layers.Lambda(get_topk, output_shape=(maxlen,txt_shape[2]))(input_text)
    print('text_mat',text_mat.get_shape())
    convs = []
    for fsz in filter_sizes:
        conv1 = layers.Convolution1D(filters,kernel_size=fsz,strides=1,activation='tanh')(text_mat)
        pool1 = layers.pooling.MaxPooling1D(maxlen-fsz+1)(conv1)
        pool1 = layers.core.Flatten()(pool1)
        print('pool',fsz,':',pool1.get_shape())
        convs.append(pool1)
    merge = K.concatenate(convs,axis=1)
    print('merge',merge.get_shape())
    # text_out = layers.Lambda(TextCNN, output_shape=(txt_shape[1],))(input_text)
    text_dense = layers.Dense(512,activation=act_1)(merge)
    image_dense = layers.Dense(512,activation=act_1)(input_image)
    mul = layers.Multiply()([text_dense,image_dense])
    pred = layers.Dense(1,activation=act_2)(mul)
    model = Model(inputs=[input_text,input_image], outputs=pred)
    model.compile(loss=loss_function, optimizer='adam', metrics=[mAP.auc])
    model.summary()
    return model
def get_topk(text_in):
    text_idx = tf.nn.top_k(tf.reduce_sum(text_in,2),maxlen)[1]
    print('text_idx',text_idx.get_shape())
    text_mat = tf.squeeze(tf.gather(text_in,text_idx,axis=1),axis=0)
    print('text_mat',text_mat.get_shape())
    return text_mat
# def TextCNN(input):
#     filter_sizes,filters = [2,3,4,5],250
#     maxlen = 400
#     print('input',input.get_shape())
#     text_idx = tf.nn.top_k(tf.reduce_sum(input,2),maxlen)[1]
#     print('text_idx',text_idx.get_shape())
#     text_mat = tf.squeeze(tf.gather(input,text_idx,axis=1),axis=0)
#     print('text_mat',text_mat.get_shape())
#     convs = []
#     for fsz in filter_sizes:
#         conv1 = layers.Conv1D(filters,kernel_size=fsz,activation='tanh')(text_mat)
#         pool1 = layers.pooling.MaxPooling1D(maxlen-fsz+1)(conv1)
#         pool1 = layers.core.Flatten()(pool1)
#         convs.append(pool1)
#     merge = K.concatenate(convs,axis=1)
#     return merge

# class TextCNN(Layer):
#     def __init__(self, **kwargs):
#         super(TextCNN, self).__init__(** kwargs)

#     def build(self, input_shape):
#         assert len(input_shape)==3
#         # W.shape = (time_steps, time_steps)
#         self.W = self.add_weight(name='att_weight', 
#                                  shape=(input_shape[1], input_shape[1]),
#                                  initializer='uniform',
#                                  trainable=True)
#         self.b = self.add_weight(name='att_bias', 
#                                  shape=(input_shape[1],),
#                                  initializer='uniform',
#                                  trainable=True)
#         super(TextCNN, self).build(input_shape)

#     def call(self, inputs):
#         # inputs.shape = (batch_size, time_steps, seq_len)
#         x = K.permute_dimensions(inputs, (0, 2, 1))
#         # x.shape = (batch_size, seq_len, time_steps)
#         a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
#         outputs = K.permute_dimensions(a * x, (0, 2, 1))
#         outputs = K.sum(outputs, axis=1)
#         return outputs

#     def compute_output_shape(self, input_shape):
#         return input_shape[0], input_shape[2]

