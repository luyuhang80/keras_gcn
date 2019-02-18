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
    # image_mul = layers.GlobalAveragePooling1D()(input_image)
    # image_rn = RNet()([text_dense,input_image])
    image_att = BilinearAttentionLayer()([text_dense,input_image])
    image_mul = layers.Multiply()([input_image,image_att])
    image_mul = layers.GlobalAveragePooling1D()(image_mul)
    image_dense = layers.Dense(512,activation='relu')(image_mul)

    mul = layers.Multiply()([text_dense,image_dense])
    pred = layers.Dense(1,activation='sigmoid')(mul)
    model = Model(inputs=[input_text,input_image], outputs=pred)
    model.compile(loss=loss_function, optimizer='adam', metrics=[mAP.auc])
    model.summary()
    return model

class RNet(layers.Layer):

    def __init__(self, conv_channels=256, out_dim=512, relation_glimpse=1, dropout_ratio=.2, **kwargs):
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
    def call(self, inputs):
        assert isinstance(inputs, list)
        '''
        :param X: [batch_size, Nr, in_dim]
        :return: relation map:[batch_size, relation_glimpse, Nr, Nr]
        relational_x: [bs, Nr, in_dim]
        Nr = Nr
        '''
        Q, X = inputs
        X_ = X
        x = X[:,:,4:]
        b = X[:,:,:4]
        pos = b
        # pos = self.pos_encoding(b)
        X = K.concatenate([X,pos],2)
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
        # relational_X = K.reshape(relational_X,[-1,f1*f2])
        relational_X = K.sum(relational_X,1)

        return [relational_X]

    def pos_encoding(self,b):
        bs, vlocs, bdim = b.get_shape()
        print('b',b.get_shape())
        x = b[:,:,0]
        y = b[:,:,1]
        w = b[:,:,2]
        h = b[:,:,3]
        xi = K.tile(K.expand_dims(x,1),[1,vlocs,1])
        xj = K.tile(K.expand_dims(x,2),[1,1,vlocs])
        print('xi',xi.get_shape())
        print('xj',xj.get_shape())
        x_delta = K.abs(xi-xj)
        yi = K.tile(K.expand_dims(y,1),[1,vlocs,1])
        yj = K.tile(K.expand_dims(y,2),[1,1,vlocs])
        y_delta = K.abs(yi-yj)
        wi = K.tile(K.expand_dims(w,1),[1,vlocs,1])
        wj = K.tile(K.expand_dims(w,2),[1,1,vlocs])
        hi = K.tile(K.expand_dims(h,1),[1,vlocs,1])
        hj = K.tile(K.expand_dims(h,2),[1,1,vlocs])
        g1 = x_delta/(wj+1e-5*K.ones_like(wj))
        g1 = K.expand_dims(g1,3)
        g2 = x_delta/(hj+1e-5*K.ones_like(hj))
        g2 = K.expand_dims(g2,3)
        g3 = K.expand_dims(K.abs(wi/wj),3)
        g4 = K.expand_dims(K.abs(hi/hj),3)
        g = K.concatenate([g1,g2,g3,g4],axis=3)
        return g
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        # return [(shape_b[0],shape_b[1],shape_b[2])]
        return [(shape_b[0],shape_b[2])]



class BilinearAttentionLayer(layers.Layer):

    def __init__(self, num_hid=1, dropout=0.5, **kwargs):
        self.num_hid = num_hid
        self.dropout = dropout
        super(BilinearAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)

        self.v_proj = layers.Dense(self.num_hid)
        self.q_proj = layers.Dense(self.num_hid)
        self.drop_out = layers.Dropout(self.dropout)
        # self.h_mat = K.placeholder(shape=(1,1,self.num_hid))
        self.h_mat = self.add_weight(name='h_mat',shape=([1,1,self.num_hid]),initializer='normal',trainable=True)
        self.h_bias = self.add_weight(name='h_bias',shape=([1,1,1]),initializer='normal',trainable=True)
        # self.h_mat = nn.Parameter(torch.Tensor(1, 1, num_hid).normal_())
        # self.h_bias = nn.Parameter(torch.Tensor(1, 1, 1).normal_())
        super(BilinearAttentionLayer, self).build(input_shape)  # Be sure to call this at the end
    
    def logits(self, v, q):
        batch, k, _ = v.get_shape()
        v_proj = self.drop_out(self.v_proj(v)) # [batch, k, num_hid]
        q_proj = K.permute_dimensions(K.expand_dims(self.drop_out(self.q_proj(q)),1),(0,2,1))# [batch, num_hid, 1]
        v_proj = v_proj * self.h_mat
        logits = K.batch_dot(v_proj, q_proj) + self.h_bias #[batch, k, 1]
        return logits

    def call(self, x):
        assert isinstance(x, list)
        q,v = x
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        # if prev_logits is None:
        logits = self.logits(v, q) #[batch, k, 1]
        # else:
            # logits = self.logits(v, q) + prev_logits
        w = K.softmax(logits, 1) #[batch, k, 1]
        v = w * v  #[batch, k, vdim]
        v = K.sum(v,1)
        # return v, logits, w
        return [v]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],1)]
