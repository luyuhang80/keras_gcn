import os
import numpy as np
import time
import random
import keras,h5py
from random import choice
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from scipy import sparse

# from sklearn.cross_validation import train_test_split


def load_wl(filename):
    dic,now = {},0
    for line in open('./data/'+filename):
        dic[line.strip().split()[0]] = now
        now += 1
    return dic

def get_list(filename):
    res = []
    for line in open(filename):
        tmp = line.strip()
        if tmp:
            res.append(tmp)
    return res

def build_text():
    data = []
    for line in open(data_path+'/total_txt_img_cat.list','r'):
        tmp = line.strip().split('\t')[0]
        file = open(data_path+'/texts_content/'+tmp+'.xml').read()
        data.append(file)
    return np.array(data)
    
def make_one_hot(data1):
    return (np.arange(10)==data1[:,None]).astype(np.integer)

def load_bow(data_path):
    now,data = 1,[]
    for line in open(data_path+'/total_txt_img_cat.list','r'):
        tmp,lab = line.strip().split('\t')[0],line.strip().split('\t')[2]
        s = str(now)
        while len(s)<4:
            s = '0' + s
        arr = np.load('/Users/yuhanglu/Desktop/GIT/yul_features/text_8874/text_'+s+'_'+lab.strip()+'.npy')
        data.append(arr)
        now += 1
    return np.array(data)

def prepair_data(train_val,data_path):

    data_path1 = '/home1/yul/yzq/data/cmplaces'
    f = h5py.File(data_path1 + '/natural_v_objs.h5','r')
    rois = f['data']['rois'].value
    objs = f['data']['v_objs'].value
    image = np.concatenate((rois,objs),2)
    image_label = f['labels'].value
    f.close() 
    
    t = h5py.File(data_path1+'/text_bow_unified.h5','r')
    text = t['data']['features'].value
    text_label = t['labels'].value

    x_x0,c_x0,x_y0,c_y0 = \
    train_test_split(text,text_label,test_size=0.1, random_state=0)
    x_x1,c_x1,x_y1,c_y1 = \
    train_test_split(image,image_label,test_size=0.1,random_state=0)

    save_test_data(x_x0,x_x1,x_y0,x_y1,c_x0,c_x1,c_y0,c_y1,data_path)
    
    train_index,test_index = make_index(train_val,x_y0,x_y1)

    train_index=index_shuffle(train_index)
    test_index=index_shuffle(test_index)

    x0_train=x_x0[train_index[0],:]
    x1_train=x_x1[train_index[1],:,:]
    y0_train=x_y0[train_index[0]]
    y1_train=x_y1[train_index[1]]
    y_train= np.ones([len(train_index[0])])
    y_train[x_y0[train_index[0]]!=x_y1[train_index[1]]]=0
    x0_test=c_x0[test_index[0],:]
    x1_test=c_x1[test_index[1],:,:]
    y0_test=c_y0[test_index[0]]
    y1_test=c_y1[test_index[1]]
    y_test= np.ones([len(test_index[0])])
    y_test[c_y0[test_index[0]]!=c_y1[test_index[1]]]=0

    return x0_train,x1_train,make_one_hot(y0_train),make_one_hot(y1_train),y_train,\
     x0_test,x1_test, make_one_hot(y0_test),make_one_hot(y1_test),y_test

    #return np.array(x0_train),np.array(x1_train),make_one_hot(np.array(y0_train)),make_one_hot(np.array(y1_train)),np.array(y_train),\
    # np.array(x0_test),np.array(x1_test), make_one_hot(np.array(y0_test)),make_one_hot(np.array(y1_test)),np.array(y_test)

def save_data(x0_train,x1_train,y_train,x0_test,x1_test,y_test,data_path):
    np.save(data_path + '/gcn/x0_train.npy',x0_train)
    np.save(data_path + '/gcn/x1_train.npy',x1_train)
    np.save(data_path + '/gcn/y_train.npy',y_train)
    np.save(data_path + '/gcn/x0_test.npy',x0_test)
    np.save(data_path + '/gcn/x1_test.npy',x1_test)
    np.save(data_path + '/gcn/y_test.npy',y_test)
    
def load_data(data_path):
    x0_train = np.load(data_path+'/gcn/x0_train.npy')
    x1_train = np.load(data_path+'/gcn/x1_train.npy')
    y_train = np.load(data_path+'/gcn/y_train.npy')
    x0_test = np.load(data_path+'/gcn/x0_test.npy')
    x1_test = np.load(data_path+'/gcn/x1_test.npy')
    y_test = np.load(data_path+'/gcn/y_test.npy')
    return x0_train,x1_train,y_train,x0_test,x1_test,y_test

def mAP(match_list):
    n = len(match_list)
    tp_counter = 0
    cumulate_precision = 0
    for i in range(0,n):
        if match_list[i] == True:
            tp_counter += 1
            cumulate_precision += (float(tp_counter)/float(i+1))
    if tp_counter != 0:
        av_precision = cumulate_precision/float(tp_counter)
        return av_precision
    return 0

def make_index(train_val,text_label,image_label):
    txt_dic,img_dic = {},{}
    for i in range(len(text_label)):
        if text_label[i] not in txt_dic:
            txt_dic[text_label[i]] = [i]
        else:
            txt_dic[text_label[i]].append(i)

        if image_label[i] not in img_dic:
            img_dic[image_label[i]] = [i]
        else:
            img_dic[image_label[i]].append(i)

    n1 = int(train_val[0]/len(txt_dic))
    n2 = int(train_val[1]/len(txt_dic))
    idx1,idx2 = [],[]
    for i in range(len(txt_dic)):
        for j in range(n1):
            idx1.append(choice(txt_dic[i]))
            idx2.append(choice(img_dic[i]))
            tmp1 = choice(range(len(txt_dic)))
            tmp2 = choice(range(len(txt_dic)))
            idx1.append(choice(txt_dic[tmp1]))
            idx2.append(choice(txt_dic[str(tmp2)]))
    ind1 = []
    ind1.append(idx1)
    ind1.append(idx2)

    ids1,ids2 = [],[]
    for i in range(len(txt_dic)):
        for j in range(n2):
            ids1.append(choice(txt_dic[i]))
            ids2.append(choice(img_dic[i]))
            tmp1 = choice(range(len(txt_dic)))
            tmp2 = choice(range(len(txt_dic)))
            ids1.append(choice(txt_dic[tmp1]))
            ids2.append(choice(txt_dic[tmp2]))
    ind2 = []
    ind2.append(ids1)
    ind2.append(ids2)

    return np.array(ind1),np.array(ind2)  

def get_label(filename):
    file=open(filename,'r+')
    label=[[],[],[],[],[],[],[],[],[],[]]
    row=0
    line=file.readline()
    while line:
        cla=int(line.strip('\n').split('\t')[2])
        label[cla-1].append(row)
        row+=1
        line=file.readline()
    return label
    
def index_shuffle(index):
    N,M=index.shape
    list_all=[]
    for i in range(0,int(M)):
        list1=[]
        list1.append(index[0][i])
        list1.append(index[1][i])
        list_all.append(list1)
    random.shuffle(list_all)
    list_re,fir,sec=[],[],[]
    for y in list_all:
        fir.append(y[0])
        sec.append(y[1])
    list_re.append(fir)
    list_re.append(sec)
    return np.array(list_re)

def load_test_data(data_path):
    x_x0,x_x1,x_y0,x_y1,c_x0,c_x1,c_y0,c_y1 = \
    np.load(data_path+'/gcn/x_x0.npy'),\
    np.load(data_path+'/gcn/x_x1.npy'),\
    np.load(data_path+'/gcn/x_y0.npy'),\
    np.load(data_path+'/gcn/x_y1.npy'),\
    np.load(data_path+'/gcn/c_x0.npy'),\
    np.load(data_path+'/gcn/c_x1.npy'),\
    np.load(data_path+'/gcn/c_y0.npy'),\
    np.load(data_path+'/gcn/c_y1.npy')
    return x_x0,x_x1,x_y0,x_y1,c_x0,c_x1,c_y0,c_y1
def save_test_data(x_x0,x_x1,x_y0,x_y1,c_x0,c_x1,c_y0,c_y1,data_path):
    np.save(data_path+'/gcn/x_x0.npy',x_x0)
    np.save(data_path+'/gcn/x_x1.npy',x_x1)
    np.save(data_path+'/gcn/x_y0.npy',x_y0)
    np.save(data_path+'/gcn/x_y1.npy',x_y1)
    np.save(data_path+'/gcn/c_x0.npy',c_x0)
    np.save(data_path+'/gcn/c_x1.npy',c_x1)
    np.save(data_path+'/gcn/c_y0.npy',c_y0)
    np.save(data_path+'/gcn/c_y1.npy',c_y1)
