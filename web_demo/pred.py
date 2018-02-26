import caffe
import os
import scipy
import skimage.transform
import numpy as np
import numpy.random as rand
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.misc as scimc
import scipy.io as sio
import time
# import visualization

def colorEncode(grayLab,colorcode):

    [h,w]= np.shape(grayLab)
    rgbLab = np.zeros((h,w,3))
    idx_unique = np.unique(grayLab)
    for idx in idx_unique:
        if idx == 0:
            continue
        rgbLab = rgbLab + np.array(grayLab == idx, dtype=np.uint8)[:,:,None] * np.reshape(colorcode[idx,:],[1,1,3])
    return rgbLab


def riverLabel(grayLab,img,colorcode):
    imgOg = img.copy()
    river = [1]
    [h,w]= np.shape(grayLab)
    rgbLab = np.zeros((h,w,3))
    idx_unique = np.unique(grayLab)
    for idx in idx_unique:
        if idx in river:
            rgbLab = rgbLab + np.array(grayLab == idx, dtype=np.uint8)[:,:,None] * np.reshape([11,200,200],[1,1,3])
        else:
            continue

    for i in range(h):
        for j in range(w):
            for k in range(3):
                if rgbLab[i][j][k] != 0:
                    imgOg[i][j][k] = rgbLab[i][j][k]
    return imgOg

def rescale_scores(scores,new_w,new_h):
    old_w = len(scores)
    old_h = len(scores[0])
    ret = np.zeros((new_w,new_h,len(scores[0,0])),dtype=np.float32)
    for x in range(new_w):
        for y in range(new_h):
            old_x = int(np.floor(x * (old_w / float(new_w))))
            old_y = int(np.floor(y * (old_h / float(new_h))))
            ret[x,y] = scores[old_x,old_y]
    return ret

def predImg(fileName,prototxtFile,modelFile):
    
    # initialize the network
    network = caffe.Net(prototxtFile, modelFile, caffe.TEST)
    
    # load the original image
    im = scimc.imread(fileName)

    # resize image to fit model description
    im_inp = skimage.transform.resize(im,(512,910),preserve_range=True)
   
    # change RGB to BGR
    im_inp = im_inp[:,:,::-1]

    # substract mean and transpose
    im_inp = np.stack((im_inp[:,:,0]-104.00699,im_inp[:,:,1]-116.66877,im_inp[:,:,2]-122.67892),axis=2)
    im_inp = np.transpose(im_inp,axes=[2,0,1])

    # load pre-defined colors
    colorMat = sio.loadmat('./visualizationCode/color150.mat')
    colors = colorMat['colors']
    
    starttime = time.time()
    # obtain predicted image and resize to original size
    starttime = time.time()
    network.blobs['data'].data[...] = np.array([im_inp])
    score = network.forward()['score']
    imPred = np.argmax(rescale_scores(np.transpose(score[0],[1,2,0]),len(im),len(im[0])),axis=2)
    endtime = time.time()

    # color encoding
    rgbPred = colorEncode(imPred, colors)
    inputImg = im
    RiverLabeled = riverLabel(imPred,inputImg, colors)
    
    return inputImg, rgbPred, RiverLabeled,imPred, '%.3f' % (endtime - starttime)
