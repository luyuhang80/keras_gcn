import os
import numpy as np
import time
import random
import keras,h5py
from numpy.random import choice
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
    train = f['splits']['train'].value
    val = f['splits']['val'].value
    test = f['splits']['test'].value
    x_x1,c_x1,v_x1,x_y1,c_y1,v_y1 = image[train],image[test],image[val],image_label[train],image_label[test],image_label[val]
    f.close() 

    f = h5py.File(data_path1 + '/text_bow_unified.h5','r')
    text = f['data']['features'].value
    text_label = f['labels'].value
    train2 = f['splits']['train'].value
    val2 = f['splits']['val'].value
    test2 = f['splits']['test'].value
    x_x0,c_x0,v_x0,x_y0,c_y0,v_y0 = text[train2],text[test2],text[val2],text_label[train2],text_label[test2],text_label[val2]
    f.close()
    
    save_test_data(x_x0,x_x1,x_y0,x_y1,c_x0,c_x1,c_y0,c_y1,data_path)

    print('origins train, test, val text shape:',x_x0.shape,c_x0.shape,v_x0.shape)
    print('origins train, test, val image shape:',x_x1.shape,c_x1.shape,v_x1.shape)
    
    train_index = make_index(train_val[0],x_y0,x_y1)
    test_index = make_index(train_val[1],v_y0,v_y1)

    train_index=index_shuffle(train_index)
    test_index=index_shuffle(test_index)

    x0_train=x_x0[train_index[0],:]
    x1_train=x_x1[train_index[1],:,:]
    y0_train=x_y0[train_index[0]]
    y1_train=x_y1[train_index[1]]
    y_train= np.ones([len(train_index[0])])
    y_train[y0_train!=y1_train]=0
    x0_test=v_x0[test_index[0],:]
    x1_test=v_x1[test_index[1],:,:]
    y0_test=v_y0[test_index[0]]
    y1_test=v_y1[test_index[1]]
    y_test= np.ones([len(test_index[0])])
    y_test[y0_test!=y1_test]=0

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

def make_index(num,text_label,image_label):
    txt_dic,img_dic = {},{}

    for i,t in enumerate(text_label):
        if t not in txt_dic:
            txt_dic[t] = [i]
        else:
            txt_dic[t].append(i)
    for j,m in enumerate(image_label):
        if m not in img_dic:
            img_dic[m] = [i]
        else:
            img_dic[m].append(i)

    for i in txt_dic:
        print(i,'-txt-',len(txt_dic[i]))
    for g in img_dic:
        print(g,'-img-',len(img_dic[g]))

    n1 = int(num/len(txt_dic))
    idx1,idx2 = [],[]
    a = txt_dic.keys()
    b = img_dic.keys()
    intersection = list(set(a).intersection(set(b)))
    for i in intersection:
        for j in range(n1):
            idx1.append(choice(txt_dic[i]))
            idx2.append(choice(img_dic[i]))
            tmp1 = choice(list(a))
            tmp2 = choice(list(b))
            idx1.append(choice(txt_dic[tmp1]))
            idx2.append(choice(img_dic[tmp2]))
    ind1 = []
    ind1.append(idx1)
    ind1.append(idx2)

    return np.array(ind1)  

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
