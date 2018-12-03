# Copyright (c) 2018 Yuhang Lu <luyuhang@iie.ac.cn>

from lib import model, utils,graph
import numpy as np
import os,time
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
import keras
import keras.layers as layers
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from lib.mAP import my_loss
from keras.callbacks import Callback,EarlyStopping
#GPU 控制
os.environ["CUDA_VISIBLE_DEVICES"] = '0,3'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
# Initialize session
sess = tf.Session()
K.set_session(sess)

train_val = [40000,5000]
print('start prepairing data ...')
# x0_train,x1_train,y0_train,y1_train,y_train,x0_test,x1_test,y0_test,y1_test,y_test = utils.prepair_data(train_val)
# save and load data
# utils.save_data(x0_train,x1_train,y_train,x0_test,x1_test,y_test)
x0_train,x1_train,y_train,x0_test,x1_test,y_test = utils.load_data()
print('data ready ...')
# Now instantiate the elmo model
elmo_model = hub.Module("/Users/yuhanglu/Desktop/myproject/my_module_cache/latest", trainable=True)
# elmo_model = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
print('load model completed ...')
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

# Build our model
def ElmoEmbedding(x):
    return elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

input_text = layers.Input(shape=(1,), dtype=tf.string)
input_image = layers.Input(shape=(x1_train.shape[1],), dtype=tf.float32)
text_embedding = layers.Lambda(ElmoEmbedding, output_shape=(1024,))(input_text)
text_dense = layers.Dense(512, activation='relu')(text_embedding)
image_dense = layers.Dense(512, activation='relu')(input_image)
mul = layers.Multiply()([text_dense,image_dense])
add = layers.Dense(256)(mul)
pred = layers.Dense(1)(mul)
model = Model(inputs=[input_text,input_image], outputs=pred)
model.compile(loss=my_loss, optimizer='adam', metrics=['accuracy'])
checkpoint = ModelCheckpoint(filepath='./checkpoints/',monitor='loss',mode='auto' ,save_best_only='False',period=1)
model.summary()
# histories = Histories()
my_callbacks = [checkpoint]
model.fit([x0_train,x1_train],y_train,validation_data\
	=([x0_test,x1_test],y_test),epochs=30,batch_size=50,callbacks=my_callbacks)
# serialize weights to HDF5
print("Saved model to disk")




