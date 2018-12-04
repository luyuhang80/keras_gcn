import os,time,random
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model

def mAP(c_x0,x_x1,c_y0,x_y1,model,k,s):
	metric = K.function(inputs=[model.layers[-4].output,model.layers[-3].output],outputs=[model.layers[-1].output])
	retrieval_size, test_size  = x_x1.shape[0], c_x0.shape[0]
	t_map_list,t_map_sum = [],0
	for i in range(test_size):
		if s=='Text':
			tmp_text = np.tile(c_x0[i],(retrieval_size,1))
			tmp_y0 = np.repeat(c_y0[i],retrieval_size)
			tmp_image = x_x1
			tmp_y1 = x_y1
		else:
			tmp_image = np.tile(c_x0[i],(retrieval_size,1))
			tmp_y1 = np.repeat(c_y0[i],retrieval_size)
			tmp_text = x_x1
			tmp_y0 = x_y1
		tmp_y= np.ones([retrieval_size])
		tmp_y[tmp_y0!=tmp_y1]=0
		pred = np.squeeze(metric([tmp_text,tmp_image]))
		match_list = tmp_y[np.argsort(pred)[-k:]]
		tmp_map = compute_list(match_list)
		t_map_list.append(tmp_map)
		t_map_sum += tmp_map
		# string = '\r%s retrievaling, %d / %d mAP: %.2f, total mAP: %.2f.' % (s,i+1,test_size,tmp_map,t_map_sum/len(t_map_list))
		# print(string,end='',flush=True)
	return t_map_sum/len(t_map_list)

def my_loss(y_true,y_pred):
	lamda,mu = 0.35,0.8
    with tf.name_scope('loss'):
	    with tf.name_scope('var_loss'):
	        labels = tf.cast(self.ph_labels, tf.float32)
	        shape = labels.get_shape()
	        same_class = tf.boolean_mask(self.logits, tf.cast(y_true ,  tf.bool))
	        diff_class = tf.boolean_mask(self.logits, tf.cast(1-y_true, tf.bool))
	        same_mean, same_var = tf.nn.moments(same_class, [0])
	        diff_mean, diff_var = tf.nn.moments(diff_class, [0])
	        var_loss = same_var + diff_var
	    with tf.name_scope('mean_loss'):
	        mean_loss = lamda * tf.where(
	            tf.greater(mu - (same_mean - diff_mean), 0),
	            mu - (same_mean - diff_mean), 0)
	    self.loss = (1) * var_loss + (1) * mean_loss
	return loss


def get_desc(text,image,model):
	# descriptor = K.function(inputs=[model.get_layer('input_1').input,model.get_layer('input_2').input],outputs=[model.get_layer('dense_1').output,model.get_layer('dense_2').output])
	descriptor = Model(inputs=model.input,outputs=[model.layers[-4].output,model.layers[-3].output])
	text,image = descriptor.predict([text,image])
	return text,image

def compute_list(match_list):
	count,pres = 0,0
	for i,t in enumerate(match_list):
		if t==1:
			count += 1
			pres += 1.0 * count/(i+1)
	if count>0: return 1.0 * pres/count
	return 0

def auc(y_true, y_pred):
    y_pred = tf.nn.sigmoid(y_pred)
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def get_time(seconds):
	m, s = divmod(seconds, 60)
	h, m = divmod(m, 60)
	if h>0:
		string = "%02d:%02d:%02d" % (h,m,s)
	elif m>0:
		string = "%02d:%02d" % (m,s)
	else:
		string = "%d" % (s)
	return string