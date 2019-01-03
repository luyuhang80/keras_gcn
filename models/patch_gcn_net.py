# Copyright (c) 2018 Yuhang Lu <luyuhang@iie.ac.cn>

from lib import graph,gcn,mAP
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
	input_text = layers.Input(shape=(txt_shape[1],txt_shape[2]))
	input_image = layers.Input(shape=(img_shape[1],img_shape[2],))
	text_embedding = gcn.MyLayer(1)(input_text)
	# text_att = AttentionLayer()(text_embedding)
	text_dense = layers.Dense(512,activation=act_1)(text_embedding)
	text_dense = layers.core.RepeatVector(img_shape[1])(text_dense)
	image_dense = layers.Dense(512,activation=act_1)(input_image)
	mul = layers.Multiply()([text_dense,image_dense])
	mul_dense = layers.Lambda(k_sum)(mul)
	pred = layers.Dense(1,activation=act_2)(mul_dense)
	model = Model(inputs=[input_text,input_image], outputs=pred)
	model.compile(loss=loss_function, optimizer='adam', metrics=[mAP.auc])
	model.summary()
	return model

def k_sum(mul):
	return K.sum(mul,2)




