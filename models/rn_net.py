# Copyright (c) 2018 Yuhang Lu <luyuhang@iie.ac.cn>

from lib import gcn_utils,graph,gcn,mAP
import numpy as np
import os,time,datetime
import tensorflow as tf
from keras import backend as K
import keras
import keras.layers as layers
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint,Callback
#GPU 控制
def build(txt,img,loss_function=mAP.my_loss):
    K.clear_session()
    # K.set_learning_phase(1)
    input_text = layers.Input(shape=(txt[1],))
    input_image = layers.Input(shape=(img[1],img[2]))
    # txt
    text_dense = gcn.MyLayer(1)(input_text)
    text_dense = layers.Dense(512,activation='relu')(text_dense)
    # img
    image_dense = RNet()([text_dense,input_image])
    # image_dense = layers.GlobalAveragePooling1D()(input_image)
    image_dense = layers.Dense(512,activation='relu')(image_dense)

    mul = layers.Multiply()([text_dense,image_dense])
    pred = layers.Dense(1,activation='sigmoid')(mul)
    model = Model(inputs=[input_text,input_image], outputs=pred)
    model.compile(loss=loss_function, optimizer='adam', metrics=[mAP.auc])
    model.summary()
    return model

class RNet(layers.Layer):

    def __init__(self, conv_channels=256, out_dim=512, relation_glimpse=1, dropout_ratio=.5, **kwargs):
        self.out_dim = out_dim
        self.conv_channels = conv_channels
        self.relation_glimpse = relation_glimpse
        self.dropout_ratio = dropout_ratio
        super(RNet, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)

        self.v_prj = layers.Dense(self.conv_channels)
        self.q_prj = layers.Dense(self.conv_channels)
        out_channel1 = int(self.conv_channels/2)
        out_channel2 = int(self.conv_channels/4)
        self.r_conv01 = layers.Conv2D(filters=out_channel1, kernel_size=1)
        self.r_conv02 = layers.Conv2D(filters=out_channel2, kernel_size=1)
        self.r_conv03 = layers.Conv2D(filters=self.relation_glimpse, kernel_size=1)
        self.r_conv1 = layers.Conv2D(filters=out_channel1, kernel_size=1, dilation_rate=(1,1),padding='valid')
        self.r_conv2 = layers.Conv2D(filters=out_channel2, kernel_size=1, dilation_rate=(1,2),padding='valid')
        self.r_conv3 = layers.Conv2D(filters=self.relation_glimpse, kernel_size=1, dilation_rate=(1,4),padding='valid')
        self.relu = layers.ReLU()
        self.drop = layers.Dropout(self.dropout_ratio)

        # Create a trainable weight variable for this layer.
        # self.kernel = self.add_weight(name='kernel',
                                      # shape=(input_shape[0][1], self.output_dim),
                                      # initializer='uniform',
                                      # trainable=True)
        super(RNet, self).build(input_shape)  # Be sure to call this at the end
    def call(self, x):
        assert isinstance(x, list)
        '''
        :param X: [batch_size, Nr, in_dim]
        :return: relation map:[batch_size, relation_glimpse, Nr, Nr]
        relational_x: [bs, Nr, in_dim]
        Nr = Nr
        '''
        Q, X = x
        X_ = X
        # # img part 
        bs, Nr, in_dim = X.get_shape() 
        # print('bs',bs,'Nr',Nr, 'in_dim',in_dim)
        # project the visual features and get the relation map
        X = self.v_prj(X) #[bs, Nr, subspace_dim]
        Q = K.expand_dims(self.q_prj(Q),1)#[bs, 1, subspace_dim]
        # print('X',X.get_shape())
        # print('Q',Q.get_shape())
        X = X + Q
        Xi = K.tile(K.expand_dims(X,1),[1,Nr,1,1])#[bs, Nr, Nr, subspace_dim]
        Xj = K.tile(K.expand_dims(X,2),[1,1,Nr,1])#[bs, Nr, Nr, subspace_dim]
        X = Xi * Xj #[bs, Nr, Nr, subspace_dim]
        # X = K.permute_dimensions(X,[0, 3, 1, 2])#[bs, subspace_dim, Nr, Nr]
        # X0 = keras.activations.relu(self.r_conv01(X))
        X0 = self.drop(self.relu(self.r_conv01(X)))
        X0 = self.drop(self.relu(self.r_conv02(X0)))
        X0 = self.drop(self.relu(self.r_conv03(X0)))

        relation_map0 = X0 + K.permute_dimensions(X0,(0,2,1,3))  # [128,1,49,49]
        relation_map0 = K.reshape(relation_map0,(-1,self.relation_glimpse,int(Nr*Nr)))

        relation_map0 = K.softmax(relation_map0)
        relation_map0 = K.reshape(relation_map0,[-1,self.relation_glimpse,Nr,Nr])# [128,1,49,49*49]

        X1 = self.drop(self.relu(self.r_conv1(X)))#[bs, subspace_dim, Nr, Nr]
        X1 = self.drop(self.relu(self.r_conv2(X1)))  # [bs, subspace_dim, Nr, Nr]
        X1 = self.drop(self.relu(self.r_conv3(X1)))  # [bs, relation_glimpse, Nr, Nr]
        # 将矩阵上下三角对应位置相加，合并相同patch关系的推理结果
        relation_map1 = X1 + K.permute_dimensions(X1,(0,2,1,3))
        # 将Nr*Nr拉直为一维特征，进行softmax，再还原为Nr*Nr二维特征
        # view（）函数： 变换数据的维度，但数据量和值不变，根据-1的位置，推测-1所在维度的值
        relation_map1 = K.reshape(relation_map1,[-1,self.relation_glimpse,Nr*Nr])
        relation_map1 = K.softmax(relation_map1)
        relation_map1 = K.reshape(relation_map1,[-1,self.relation_glimpse,Nr,Nr])# [128,1,49,49*49]
        relational_X = K.zeros_like(X_)
        for g in range(self.relation_glimpse):
            relational_X = relational_X + K.batch_dot(relation_map1[:,g,:,:], X_) + K.batch_dot(relation_map0[:,g,:,:], X_)
        relational_X = relational_X/(2*self.relation_glimpse) #(relational_X/self.relation_glimpse + self.nonlinear(X_))/2
        _ , f1, f2 = relational_X.get_shape()
        relational_X = K.reshape(relational_X,[-1,f1*f2])

        return [relational_X]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_b[0],shape_b[1]*shape_b[2])]



class MyLayer(layers.Layer):

    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.fc1 = layers.Dense(512)
        self.fc2 = layers.Dense(512)
        self.fc3 = layers.Dense(1)
        # Create a trainable weight variable for this layer.
        # self.kernel = self.add_weight(name='kernel',
                                      # shape=(input_shape[0][1], self.output_dim),
                                      # initializer='uniform',
                                      # trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        a, b = x
        _,N1,N2 = b.get_shape()
        b = K.reshape(b,[-1,N1*N2])
        a = self.fc1(a)
        b = self.fc2(b)
        c = self.fc3(a*b)
        return [c]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],1)]
