# Copyright (c) 2018 Yuhang Lu <luyuhang@iie.ac.cn>

from lib import rn_utils,graph,gcn,mAP
from lib.h5_data_pair_loader import PosNegLoader
import numpy as np
import os,time,datetime
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
from models import rn_net as net
import keras,h5py
import keras.layers as layers
from keras.models import Model
from keras.callbacks import ModelCheckpoint,Callback

#GPU 控制
K.clear_session()
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
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
data_path = '/Users/yuhanglu/Desktop/myproject/data'
BATCH_SIZE = 128
# Initialize session
train_val = [2000,500]
print('start prepairing data ...')

# initialize data loader
text_h5_path = "/home1/yul/yzq/data/cmplaces/text_bow_unified.h5"
image_h5_path = "/home1/yul/yzq/data/cmplaces/natural50k_onr.h5"
adjmat_path = '/home1/yul/yzq/data/cmplaces/txt_graph_knn_unified.txt'
n_classes = 205
label_start_with_zero = True
n_train = 819200
n_val = 8192

train_loader = PosNegLoader(text_h5_path, image_h5_path, "train", "train", n_train, n_train,batch_size=BATCH_SIZE, n_classes=n_classes, shuffle=True, whole_batches=True)
val_loader = PosNegLoader(text_h5_path, image_h5_path, "val", "val", n_val, n_val,batch_size=BATCH_SIZE, n_classes=n_classes, shuffle=True, whole_batches=True)

x0_train,x1_train,y0_train,y1_train,y_train,x0_test,x1_test,y0_test,y1_test,y_test = rn_utils.prepair_data(train_val,data_path,train_loader,val_loader)
# save and load data
rn_utils.save_data(x0_train,x1_train,y_train,x0_test,x1_test,y_test,data_path)
x0_train,x1_train,y_train,x0_test,x1_test,y_test = rn_utils.load_data(data_path)
# x1_train = x1_train[:,:,4:]
# x1_test = x1_test[:,:,4:]
model = net.build(x0_train.shape,x1_train.shape,loss_function='binary_crossentropy')
# model = net.build(x0_train.shape[1],x1_train.shape[1],act_1=None,act_2=None,loss_function=mAP.my_loss)
filepath = 'model_{epoch:02d}_{val_auc:.2f}.HDF5'
checkpoint = ModelCheckpoint(os.path.join(path,filepath),verbose=1,save_weights_only='True',period=1)
my_callbacks = [checkpoint]

model.fit([x0_train,x1_train],y_train,validation_data=\
	([x0_test,x1_test],y_test),epochs=30,batch_size=BATCH_SIZE,callbacks=my_callbacks)




