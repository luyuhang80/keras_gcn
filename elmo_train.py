# Copyright (c) 2018 Yuhang Lu <luyuhang@iie.ac.cn>

from lib import elmo_utils,mAP
import numpy as np
import os,time,datetime
import tensorflow as tf
from keras import backend as K
from models import gcn_net as net
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
# x0_train,x1_train,y0_train,y1_train,y_train,x0_test,x1_test,y0_test,y1_test,y_test = elmo_utils.prepair_data(train_val)
# save and load data
# elmo_utils.save_data(x0_train,x1_train,y_train,x0_test,x1_test,y_test)
x0_train,x1_train,y_train,x0_test,x1_test,y_test = elmo_utils.load_data()
model = net.build(x0_train.shape[1],x1_train.shape[1],act=None,loss_function=mAP.my_loss)
filepath = 'model_{epoch:02d}_{val_auc:.2f}.HDF5'
checkpoint = ModelCheckpoint(os.path.join(path,filepath),verbose=1,save_weights_only='True',period=1)
my_callbacks = [checkpoint]
model.fit([x0_train,x1_train],y_train,validation_data=\
	([x0_test,x1_test],y_test),epochs=30,batch_size=BATCH_SIZE,callbacks=my_callbacks)




