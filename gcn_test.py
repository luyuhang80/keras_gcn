# Copyright (c) 2018 Yuhang Lu <luyuhang@iie.ac.cn>

from lib import gcn_utils,graph,gcn,mAP
import numpy as np
import os,time,sys
import tensorflow as tf
import tensorflow_hub as hub
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
now_time = sys.argv[1]
save_dir = os.path.join(save_dir,now_time)

data_path = '../data/'
BATCH_SIZE = 128
# Initialize session
sess = tf.Session()
K.set_session(sess)

train_val = [40000,5000]
print('start prepairing data ...')
# x0_train,x1_train,y0_train,y1_train,y_train,x0_test,x1_test,y0_test,y1_test,y_test = gcn_utils.prepair_data(train_val,data_path)
# save and load data
# gcn_utils.save_data(x0_train,x1_train,y_train,x0_test,x1_test,y_test,data_path)
x0_train,x1_train,y_train,x0_test,x1_test,y_test = gcn_utils.load_data(data_path)
x_x0,x_x1,x_y0,x_y1,c_x0,c_x1,c_y0,c_y1 = gcn_utils.load_test_data(data_path)

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

input_text = layers.Input(shape=(x0_train.shape[1],), dtype=tf.float32)
input_image = layers.Input(shape=(x1_train.shape[1],), dtype=tf.float32)
text_embedding = layers.Dense(4096,activation='relu')(input_text)
# text_embedding = layers.Lambda(gcn.build,output_shape=(x0_train.shape[1],))(input_text)
text_dense = layers.Dense(512,activation='relu')(text_embedding)
image_dense = layers.Dense(512,activation='relu')(input_image)
mul = layers.Multiply()([text_dense,image_dense])
pred = layers.Dense(1,activation='sigmoid')(mul)
model = Model(inputs=[input_text,input_image], outputs=pred)
# model.compile(loss=mAP.my_loss, optimizer='adam', metrics=[mAP.auc])
model.compile(loss=mAP.my_loss, optimizer='adam', metrics=[mAP.auc])
# filepath = 'model_{epoch:02d}_{val_auc:.2f}.HDF5'
# checkpoint = ModelCheckpoint(os.path.join(save_dir,filepath),verbose=1,save_weights_only='True',period=1)
model.summary()
max_ = (0,'None')
for file in os.listdir(save_dir):
	if os.path.splitext(file)[1] == '.HDF5':
		filename = '\r{}: '.format(file)
		string = filename
		print(string,end='',flush=True)
		model.load_weights(os.path.join(save_dir,file))
		string += 'Descriptors building ... '
		print(string,end ='',flush=True)
		des_x_x0,des_x_x1 = mAP.get_desc(x_x0,x_x1,model)
		des_c_x0,des_c_x1 = mAP.get_desc(c_x0,c_x1,model)
		string += 'got .'
		print(string,end='',flush=True)
		t_m = mAP.mAP(des_c_x0,des_x_x1,c_y0,x_y1,model,100,'Text')
		i_m = mAP.mAP(des_c_x1,des_x_x0,c_y1,x_y0,model,100,'Image')
		string = filename + 'Avg:{:.2f}, Txt:{:.2f}, Img:{:.2f}.'.format((t_m+i_m)/2,t_m,i_m)
		if (t_m+i_m)/2 > max_[0]:
			max_ = ((t_m+i_m)/2,filename)
		print(string)
print('The best model is:{}, avg map :{:.2f}'.format(max_[1],max_[0]))



