# Copyright (c) 2018 Yuhang Lu <luyuhang@iie.ac.cn>

from lib import elmo_utils
import numpy as np
import os,time,random,keras
from scipy import sparse
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
import keras.layers as layers
from keras.models import Model
from sklearn.model_selection import train_test_split
import keras.preprocessing.text as T
from keras.preprocessing.text import Tokenizer
from lib.mAP import get_desc,mAP,my_loss
#GPU 控制
os.environ["CUDA_VISIBLE_DEVICES"] = '0,3'
import threading
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
# Initialize session
sess = tf.Session()
K.set_session(sess)

train_val = [40000,5000]
print('start prepairing data ...')
x0_train,x1_train,y0_train,y1_train,y_train,x0_test,x1_test,y0_test,y1_test,y_test = elmo_utils.prepair_data(train_val)
# save and load data
elmo_utils.save_data(x0_train,x1_train,y_train,x0_test,x1_test,y_test)
x0_train,x1_train,y_train,x0_test,x1_test,y_test = elmo_utils.load_data()
x_x0,x_x1,x_y0,x_y1,c_x0,c_x1,c_y0,c_y1 = elmo_utils.load_test_data()
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
text_dense = layers.Dense(512, activation='relu')(text_embedding)
image_dense = layers.Dense(512, activation='relu')(input_image)
mul = layers.Multiply()([text_dense,image_dense])
pred = layers.Dense(1, activation='sigmoid')(mul)
model = Model(inputs=[input_text,input_image], outputs=pred)
model.compile(loss=my_loss, optimizer='adam', metrics=[auc])
model.summary()
# histories = Histories()
# model.fit([x0_train,x1_train],y_train,validation_data=([x0_test,x1_test],y_test),epochs=1,batch_size=8)
# load weights to HDF5
model.load_weights("model.h5")
# res = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
# res = model.predict([x_x0[:4],x_x1[:4]],batch_size=4)
print("load model from disk")
x_x0,x_x1 = get_desc(x_x0,x_x1,model)
c_x0,c_x1 = get_desc(c_x0,c_x1,model)
# threads = []
# threads.append(threading.Thread(target=mAP,args=(c_x0,x_x1,c_y0,x_y1,model,100,'Text')))
# threads.append(threading.Thread(target=mAP,args=(c_x1,x_x0,c_y1,x_y0,model,100,'Image')))
# for t in threads:
    # t.setDaemon(True)
    # t.start()
# t_m = new_mAP(des_text,des_image,x_y0[:64],x_y1[:64],model,100,'Text')
t_m = mAP(c_x0,x_x1,c_y0,x_y1,model,100,'Text')
print('Text map:',t_m)
# i_m = mAP(c_x1,x_x0,c_y1,x_y0,model,100,'Image')
# print('Image map:',i_m)
# print('Text map:',t_m, 'Image map:',i_m)

