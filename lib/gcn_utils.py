import os
import numpy as np
import time
import random
from random import choice
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


def prepair_data(train_val,data_path):

    text = np.squeeze(np.load(data_path + '/load_nx0.npy').astype(float))
    text_label = np.load(data_path + '/load_ny0.npy').astype(int) - 1
    image = np.squeeze(np.load(data_path + '/load_nx1.npy').astype(float))
    image_label = np.load(data_path + '/load_ny1.npy').astype(int) - 1

    x_x0=text[693:,:]
    x_x1=image[693:,:]
    x_y0=text_label[693:]
    x_y1=image_label[693:]
    c_x0=text[:693,:]
    c_x1=image[:693,:]
    c_y0=text_label[:693]
    c_y1=image_label[:693]
    save_test_data(x_x0,x_x1,x_y0,x_y1,c_x0,c_x1,c_y0,c_y1)
    # make pos and neg examples
    train_index=make_index(train_val[0],train_val[0],0,data_path)
    test_index=make_index(train_val[1],train_val[1],1,data_path)
    train_index=index_shuffle(train_index)
    test_index=index_shuffle(test_index)
    x0_train=x_x0[train_index[0],:]
    x1_train=x_x1[train_index[1],:]
    y0_train=x_y0[train_index[0]]
    y1_train=x_y1[train_index[1]]
    y_train= np.ones([len(train_index[0])])
    y_train[x_y0[train_index[0]]!=x_y1[train_index[1]]]=0
    x0_test=c_x0[test_index[0],:]
    x1_test=c_x1[test_index[1],:]
    y0_test=c_y0[test_index[0]]
    y1_test=c_y1[test_index[1]]
    y_test= np.ones([len(test_index[0])])
    y_test[c_y0[test_index[0]]!=c_y1[test_index[1]]]=0

    return x0_train,x1_train,make_one_hot(y0_train),make_one_hot(y1_train),y_train,\
     x0_test,x1_test, make_one_hot(y0_test),make_one_hot(y1_test),y_test

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

def make_index(n1,n2,dest,data_path):
    n2=(n2*11)/10
    if dest==0:
        n1=n1-2000
        filename=data_path+'/trainset_txt_img_cat.list'
        r1=[i for i in range(2173)]
        r2=[i for i in range(2173)]
    if dest==1:
        n1=n1-1000
        filename=data_path+'/testset_txt_img_cat.list'
        r1=[i for i in range(693)]
        r2=[i for i in range(693)]
    list=get_label(filename)
    ind=[]
    for i in range(10):
        # print(choice(list[i]))
        r1+=[choice(list[i]) for _ in range(int(n1/10))]
        r2+=[choice(list[i]) for _ in range(int(n1/10))]
    for p in range(10):
        for q in range(10):
            if p!=q:
                r1+=[choice(list[p]) for _ in range(int(n2/100))]
                r2+=[choice(list[q]) for _ in range(int(n2/100))]
    ind.append(r1)
    ind.append(r2)
    arr=np.array(ind)
    return arr  

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
