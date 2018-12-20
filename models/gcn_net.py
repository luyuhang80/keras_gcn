# Copyright (c) 2018 Yuhang Lu <luyuhang@iie.ac.cn>

from lib import gcn_utils,graph,gcn,mAP
import numpy as np
import os,time,datetime
import tensorflow as tf
from keras import backend as K
import keras
import keras.layers as layers
from keras.models import Model
from keras.callbacks import ModelCheckpoint,Callback
#GPU 控制
def build(txt_shape,img_shape,act_1=None,act_2=None,loss_function=mAP.my_loss):
	K.clear_session()
	input_text = layers.Input(shape=(txt_shape,))
	input_image = layers.Input(shape=(img_shape,))
	text_embedding = gcn.MyLayer(1)(input_text)
	# text_att = AttentionLayer()(text_embedding)
	text_dense = layers.Dense(512,activation=act_1)(text_att)
	image_dense = layers.Dense(512,activation=act_1)(input_image)
	mul = layers.Multiply()([text_dense,image_dense])
	pred = layers.Dense(1,activation=act_2)(mul)
	model = Model(inputs=[input_text,input_image], outputs=pred)
	model.compile(loss=loss_function, optimizer='adam', metrics=[mAP.auc])
	model.summary()
	return model
	
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias', 
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        outputs = K.permute_dimensions(a * x, (0, 2, 1))
        outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]




