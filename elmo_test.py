# Copyright (c) 2018 Yuhang Lu <luyuhang@iie.ac.cn>

from lib import gcn_utils,graph,gcn,mAP
import numpy as np
import os,time,datetime,sys
import tensorflow as tf
from keras import backend as K
import keras
import keras.layers as layers
from models import elmo_net as net
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback,EarlyStopping
#GPU 控制
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
save_dir = os.path.join(os.getcwd(),'checkpoints')
now_time = sys.argv[1]
save_dir = os.path.join(save_dir,now_time)
data_path = '../data/'
BATCH_SIZE = 128

train_val = [40000,5000]
print('start prepairing data ...')
x0_train,x1_train,y0_train,y1_train,y_train,x0_test,x1_test,y0_test,y1_test,y_test = elmo_utils.prepair_data(train_val)
# save and load data
elmo_utils.save_data(x0_train,x1_train,y_train,x0_test,x1_test,y_test)
x0_train,x1_train,y_train,x0_test,x1_test,y_test = elmo_utils.load_data()
x_x0,x_x1,x_y0,x_y1,c_x0,c_x1,c_y0,c_y1 = elmo_utils.load_test_data()
print('data ready ...')
model = net.build(x0_train.shape[1],x1_train.shape[1],act=None,loss_function=mAP.my_loss)
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
			max_ = ((t_m+i_m)/2,filename[1:-3])
		print(string)
print('The best model is:{}, avg map :{:.2f}'.format(max_[1],max_[0]))




