# -*- coding: utf-8 -*-
''' @brief: 进行一对一的人脸比对，前提是人脸已经统一对齐过了。 @author: Riwei Chen <riwei.chen@outlook.com> '''
import matplotlib.pyplot as plt
import numpy as np
import skimage 
import sys
import os
import glob
import numpy.linalg as LA
caffe_root = '/home/crw/caffe-master/'
caffe_root = '/media/crw/MyBook/Caffe/caffe-triplet/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import sklearn
import sklearn.metrics.pairwise as pw
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,roc_auc_score
from skimage import transform as tf

caffe.set_mode_cpu()
# 训练数据中，每个通道的平均值
averageImg = [129.1863,104.7624,93.5940]

# 全局使用到的一些数据，保留在全局变量
#=====================================
metric='cosine'
model_define='model_maxout/deploy.prototxt'
model_weight='model_maxout/small_maxout2__iter_1360000.caffemodel'
#model_weight='/media/crw/MyBook/Model/FaceRecognition/Softmax/try6_7/small_maxout100x100__iter_1400000.caffemodel'

feature_layer='eltwise10'
#feature_layer='l2_norm'
image_formats =['jpg','png','bmp']
feature_len = 256
data_w = 128
data_h =  128
#feature_len = 128
#data_w = 256
#data_h = 256
data_as_gray = True
sub_mean = False
scale = 1
#scale =255
net = caffe.Classifier(model_define, model_weight)
#====================================
def read_image(filename,w=128,h=128,as_grey=False):
    ''' @brief: 读取一个图片，返回矩阵 @param：w，h：保留的图像大小 '''
    if as_grey == True:
        X=np.empty((1,1,w,h))
    else:
        X=np.empty((1,3,w,h))
    image=skimage.io.imread(filename,as_grey=as_grey)
    image=tf.resize(image,(w,h))*scale
    if as_grey == True:
        X[0,0,:,:]=image[:,:]
    else:
        # 注意通道的一致性
        if sub_mean == True:
            X[0,2,:,:]=image[:,:,0]-averageImg[0]
            X[0,1,:,:]=image[:,:,1]-averageImg[1]
            X[0,0,:,:]=image[:,:,2]-averageImg[2]       
        else:
            X[0,2,:,:]=image[:,:,0]
            X[0,1,:,:]=image[:,:,1]
            X[0,0,:,:]=image[:,:,2]
    return X

def get_image_feature(filename):
    ''' @brief：获取特征 @param： 图像的文件 @return：feature，提取到的人脸特征 '''
    X=read_image(filename,w=data_w,h=data_h,as_grey=data_as_gray)    
    out = net.forward_all(data=X)                             
    feature = np.float64(out[feature_layer])
    feature=np.reshape(feature,(1,feature_len))
    return feature

def consia_distance(feature1, feature2):
    ''' @brief: 计算两个向量的余炫距离。 '''     
    cx = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)), 2)
    consia=cx(feature1,feature2)
    result = 0.5+0.5*consia
    return result

def evaluate_by_distance(feature1,feature2):    
    ''' @brief：计算提到的特征之间的距离 @param：feature1 特征1 @param：feature2 特征2 '''
    if metric == 'cosine':
        consia_dist = consia_distance(feature1,feature2)
        return consia_dist
    else:
        mt=pw.pairwise_distances(feature1, feature2, metric)
        distance=mt[0][0] 
        return distance



image_formats = ['jpg','png']

feature_dict = dict()
def evaluate_kaggle_test(filepath,filename,resultfile='submit.csv'):
    ''' @brief: 测试evaluate kaggle 数据集合 '''
    fid = open(filename)
    fid.readline()
    lines = fid.readlines()
    fid.close()
    fid =open(resultfile,'w')
    fid.write("Id,Prediction"+'\n')
    result = np.zeros((len(lines),))
    i = 0
    for line in lines:
        word = line.split(',')
        filename1 = os.path.join(filepath,word[1].strip())
        filename2 = os.path.join(filepath,word[2].strip())
        if feature_dict.has_key(filename1):
            feature1 = feature_dict[filename1]
        else:
            feature1 =get_image_feature(filename1)
            feature_dict[filename1] = feature1
        if feature_dict.has_key(filename2):
            feature2 = feature_dict[filename2]
        else:
            feature2 =get_image_feature(filename2)
            feature_dict[filename2] = feature2           
        distance = evaluate_by_distance(feature1,feature2)
        result[i] = distance
        i=i+1
    d_max = np.max(result)
    d_min =np.min(result)
    print d_max,d_min
    i=0
    for line in lines:
        word = line.split(',')
        fid.write(word[0]+','+str((result[i]-d_min)/(d_max-d_min))+'\n')   
        i=i+1
    fid.close()

if __name__ == '__main__':
    filepath = '/media/crw/MyBook/TestData/kaggle_Face_verification_challenge/train_dlib'
    #evaluate_kaggle_train(filepath) 
    filepath = '/media/crw/MyBook/TestData/kaggle_Face_verification_challenge/test_dlib_crop'    
    filename = '/media/crw/MyBook/TestData/kaggle_Face_verification_challenge/pairs.csv'    
    resultfile='submission.csv'    
    evaluate_kaggle_test(filepath,filename,resultfile)