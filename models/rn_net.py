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
    # text_gcn = gcn.MyLayer(1)(input_text)
    text_dense = layers.Dense(2048,activation='relu')(input_text)
    text_dense = layers.Dense(1024,activation='relu')(text_dense)
    # img
    # image_mul = layers.GlobalAveragePooling1D()(input_image)
    image_rn = RNet()([input_image])
    image_avp = layers.GlobalAveragePooling1D()(image_rn)
    # image_att = BilinearAttentionLayer()([text_dense,input_image])
    # image_mul = layers.Multiply()([input_image,image_att])
    # image_mul = layers.GlobalAveragePooling1D()(image_mul)
    image_dense = layers.Dense(1024,activation='relu')(image_avp)

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
        #assert isinstance(input_shape, list)

        self.v_prj = layers.Dense(self.conv_channels)
        self.q_prj = layers.Dense(self.conv_channels)
        out_channel1 = int(self.conv_channels/2)
        out_channel2 = int(self.conv_channels/4)
        self.r_conv01 = layers.Conv2D(filters=out_channel1, kernel_size=1)
        self.r_conv02 = layers.Conv2D(filters=out_channel2, kernel_size=1)
        self.r_conv03 = layers.Conv2D(filters=self.relation_glimpse, kernel_size=1)
        self.r_conv1 = layers.Conv2D(filters=out_channel1, kernel_size=3, dilation_rate=1,padding='same')
        self.r_conv2 = layers.Conv2D(filters=out_channel2, kernel_size=3, dilation_rate=2,padding='same')
        self.r_conv3 = layers.Conv2D(filters=self.relation_glimpse, kernel_size=3, dilation_rate=4,padding='same')
        self.relu = layers.ReLU()
        self.drop = layers.Dropout(self.dropout_ratio)

        # Create a trainable weight variable for this layer.
        # self.kernel = self.add_weight(name='kernel',
                                      # shape=(input_shape[0][1], self.output_dim),
                                      # initializer='uniform',
                                      # trainable=True)
        super(RNet, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        # assert isinstance(inputs, list)
        '''
        :param X: [batch_size, Nr, in_dim]
        :return: relation map:[batch_size, relation_glimpse, Nr, Nr]
        relational_x: [bs, Nr, in_dim]
        Nr = Nr
        '''
        # Q, X = inputs
        ph_x = inputs[0]
        subsapce_dim = self.conv_channels
        relation_glimpse = self.relation_glimpse
        # # img part 
        bs, N, in_dim = ph_x.get_shape() 
        # print('bs',bs,'Nr',Nr, 'in_dim',in_dim)
        # project the visual features and get the relation map
        ph_reshape = tf.reshape(ph_x, [-1, in_dim])
        ph_subdim, _ = self.fc(ph_reshape, subsapce_dim, activation_fn=None)
        ph_subdim = tf.reshape(ph_subdim, [int(B), int(N), int(subsapce_dim)])
        ph_exp1 = tf.expand_dims(ph_subdim, 1)
        ph_exp1 = tf.tile(ph_exp1, [1, N, 1, 1])
        ph_exp2 = tf.expand_dims(ph_subdim, 2)
        ph_exp2 = tf.tile(ph_exp2, [1, 1, N, 1])
        ph_input = ph_exp1 * ph_exp2  # [bs,N,N, ph_subdim]

        X0 = tf.nn.dropout(tf.nn.relu(tf.layers.conv2d(inputs=ph_input,filters=int(subsapce_dim/2),kernel_size=1)),self.dropout_ratio)
        X0 = tf.nn.dropout(tf.nn.relu(tf.layers.conv2d(inputs=X0,filters=int(subsapce_dim/4),kernel_size=1)),self.dropout_ratio)
        X0 = tf.nn.dropout(tf.nn.relu(tf.layers.conv2d(inputs=X0,filters=relation_glimpse,kernel_size=1)), self.dropout_ratio)
        # (256,36,36,1)
        rel_map0 = X0 + tf.transpose(X0,[0,2,1,3])
        # print('rel_map0_bt',rel_map0.get_shape())   # (256,36,36,1)
        rel_map0 = tf.transpose(rel_map0,[0,3,2,1])
        # print('rel_map0_at',rel_map0.get_shape())   #(256,1,36,36)

        rel_map0 = tf.reshape(rel_map0,[self.batch_size,relation_glimpse,-1])
        # print('rel_map0_shape1',rel_map0.get_shape())  #(256,1,1296)
        rel_map0 = tf.nn.softmax(rel_map0,axis=2)
        rel_map0 = tf.reshape(rel_map0,[self.batch_size,relation_glimpse,N,-1])
        print('rel_map0',rel_map0.get_shape())  #(256,1,36,36)

        X1 = tf.nn.dropout(tf.nn.relu(tf.layers.conv2d(inputs=ph_input,filters=int(subsapce_dim/2),kernel_size=3,dilation_rate=(1,1),padding='same')),dropout_ratio)
        X1 = tf.nn.dropout(tf.nn.relu(tf.layers.conv2d(inputs=X1,filters=int(subsapce_dim/4),kernel_size=3,dilation_rate=(1,2),padding='same')),dropout_ratio)
        X1 = tf.nn.dropout(tf.nn.relu(tf.layers.conv2d(inputs=X1,filters=relation_glimpse,kernel_size=3,dilation_rate=(1,4),padding='same')),dropout_ratio)
        rel_map1 = X1 + tf.transpose(X1,[0,2,1,3])
        rel_map1 = tf.transpose(rel_map1,[0,3,2,1])
        rel_map1 = tf.reshape(rel_map1,[self.batch_size,relation_glimpse,-1])
        rel_map1 = tf.nn.softmax(rel_map1,2)
        rel_map1 = tf.reshape(rel_map1,[self.batch_size,relation_glimpse,N,-1])
        print('rel_map1',rel_map1.get_shape())  # (256,1,36,36)
        print('ph_x',ph_x.get_shape())   # (256,36,2048)

        rel_x = tf.zeros_like(ph_x)
        for g in range(relation_glimpse):
            rel_x = rel_x + tf.matmul(rel_map1[:,g,:,:], ph_x) + tf.matmul(rel_map0[:,g,:,:], ph_x)
        rel_x = rel_x/(2 * relation_glimpse)

        rn_out = tf.reshape(rel_x,[self.batch_size,-1])

        return [rn_out]

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
        g2 = y_delta/(hj+1e-5*K.ones_like(hj))
        g2 = K.expand_dims(g2,3)
        g3 = K.expand_dims(K.abs(wi/wj),3)
        g4 = K.expand_dims(K.abs(hi/hj),3)
        g = K.concatenate([g1,g2,g3,g4],axis=3)
        return g
    def compute_output_shape(self, input_shape):
        #assert isinstance(input_shape, list)
        # shape_a, shape_b = input_shape
        # return [(shape_b[0],shape_b[1],shape_b[2])]
        # return [(shape_b[0],shape_b[2])]
        return [input_shape]



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
        return [(shape_a[0],shape_b[1],1)]
