# Copyright (c) 2018 Yuhang Lu <luyuhang@iie.ac.cn>

from lib import gcn_utils,graph,gcn,mAP
import numpy as np
import os,time,datetime
import tensorflow as tf
from keras import backend as K
import keras
import keras.layers as layers
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback,EarlyStopping
#GPU 控制
os.environ["CUDA_VISIBLE_DEVICES"] = '0,3'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
save_dir = os.path.join(os.getcwd(),'checkpoints')
now_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
path = os.path.join(save_dir,now_time)
isExists= os.path.exists(path)
if not isExists:
	os.makedirs(path)
else:
	print(now_time+' path exists, create fail.')
	sys.exit()

data_path = '../data'
BATCH_SIZE = 128
# Initialize session
train_val = [40000,5000]
print('start prepairing data ...')
x0_train,x1_train,y0_train,y1_train,y_train,x0_test,x1_test,y0_test,y1_test,y_test = gcn_utils.prepair_data(train_val,data_path)
# save and load data
gcn_utils.save_data(x0_train,x1_train,y_train,x0_test,x1_test,y_test,data_path)
x0_train,x1_train,y_train,x0_test,x1_test,y_test = gcn_utils.load_data(data_path)
print('data ready ...')
# input_text = layers.Input(batch_shape=(BATCH_SIZE,x0_train.shape[1],x0_train.shape[2],), dtype=tf.float32)
input_text = layers.Input(shape=(x0_train.shape[1],), dtype=tf.float32)
# input_image = layers.Input(batch_shape=(BATCH_SIZE,x1_train.shape[1],), dtype=tf.float32)
input_image = layers.Input(shape=(x1_train.shape[1],), dtype=tf.float32)
text_embedding = layers.Dense(4096,activation='relu')(input_text)
# text_embedding = layers.Lambda(gcn.build,output_shape=(x0_train.shape[1],))(input_text)
text_dense = layers.Dense(512,activation='relu')(text_embedding)
image_dense = layers.Dense(512,activation='relu')(input_image)
mul = layers.Multiply()([text_dense,image_dense])
pred = layers.Dense(1,activation='sigmoid')(mul)
model = Model(inputs=[input_text,input_image], outputs=pred)
model.compile(loss=mAP.my_loss, optimizer='adam', metrics=[mAP.auc])
filepath = 'model_{epoch:02d}_{val_auc:.2f}.HDF5'
checkpoint = ModelCheckpoint(os.path.join(path,filepath),verbose=1,save_weights_only='True',period=1)
model.summary()
# histories = Histories()
# model.load_weights('./checkpoints/model_06_2.21.HDF5')
my_callbacks = [checkpoint]
model.fit([x0_train,x1_train],y_train,validation_data\
	=([x0_test,x1_test],y_test),epochs=30,batch_size=BATCH_SIZE,callbacks=my_callbacks)
# serialize weights to HDF5
print("Saved model to disk")




