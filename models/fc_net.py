# Copyright (c) 2018 Yuhang Lu <luyuhang@iie.ac.cn>

from lib import gcn_utils,graph,gcn,mAP
import numpy as np
import os,time,datetime
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
import keras
import keras.layers as layers
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback,EarlyStopping
#GPU 控制
def build(txt_shape,img_shape,act=None,loss_function=mAP.my_loss):
	input_text = layers.Input(shape=(txt_shape,))
	input_image = layers.Input(shape=(img_shape,))
	text_embedding = layers.Dense(2048,activation='relu')(input_text)
	text_dense = layers.Dense(512,activation='relu')(text_embedding)
	image_dense = layers.Dense(512,activation='relu')(input_image)
	mul = layers.Multiply()([text_dense,image_dense])
	pred = layers.Dense(1,activation=act)(mul)
	model = Model(inputs=[input_text,input_image], outputs=pred)
	model.compile(loss=loss_function, optimizer='adam', metrics=[mAP.auc])
	model.summary()
	return model
# print("Saved model to disk")




