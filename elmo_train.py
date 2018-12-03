# Copyright (c) 2018 Yuhang Lu <luyuhang@iie.ac.cn>

from lib import elmo_utils,mAP
import numpy as np
import os,time,keras
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
import keras.layers as layers
from keras.models import Model
from keras.callbacks import ModelCheckpoint,Callback
#GPU 控制
os.environ["CUDA_VISIBLE_DEVICES"] = '0,3'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
save_dir = os.path.join(os.getcwd(),'checkpoints')
# Initialize session
sess = tf.Session()
K.set_session(sess)

train_val = [40000,5000]
print('start prepairing data ...')
# x0_train,x1_train,y0_train,y1_train,y_train,x0_test,x1_test,y0_test,y1_test,y_test = elmo_utils.prepair_data(train_val)
# save and load data
# elmo_utils.save_data(x0_train,x1_train,y_train,x0_test,x1_test,y_test)
x0_train,x1_train,y_train,x0_test,x1_test,y_test = elmo_utils.load_data()
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

input_text = layers.Input(shape=(1,), dtype=tf.string)
input_image = layers.Input(shape=(x1_train.shape[1],), dtype=tf.float32)
text_embedding = layers.Lambda(ElmoEmbedding, output_shape=(1024,))(input_text)
text_dense = layers.Dense(512, activation=None)(text_embedding)
image_dense = layers.Dense(512, activation=None)(input_image)
mul = layers.Multiply()([text_dense,image_dense])
add = layers.Dense(256,activation=None)(mul)
pred = layers.Dense(1,activation=None)(add)
model = Model(inputs=[input_text,input_image], outputs=pred)
model.compile(loss=mAP.my_loss, optimizer='adam', metrics=[mAP.auc])
filepath = 'model_{epoch:02d}_{val_loss:.2f}.HDF5'
checkpoint = ModelCheckpoint(os.path.join(save_dir,filepath),verbose=1,save_weights_only='True',period=1)
model.summary()
# histories = Histories()
my_callbacks = [checkpoint]
model.fit([x0_train,x1_train],y_train,validation_data\
	=([x0_test,x1_test],y_test),epochs=30,batch_size=50,callbacks=my_callbacks)
# serialize weights to HDF5
print("Saved model to disk")




