#-*- coding:utf-8 -*-
from PIL import Image
import os.path
import os
import numpy as np
from scipy.misc import imsave
path = "D:/pix2pix-tensorflow/cifar16/res/"
savepath = "D:/pix2pix-tensorflow/cifar32/img/"
filelist = os.listdir(path)
#filelist.sort(key=lambda x:int(x[:-4]))
i = 0
for file in sorted(filelist):
    
    filepath = path + file
    img = Image.open(filepath)
    img = np.array(img)
    pad = np.zeros([32,32,3])
    pad [8:24,8:24,:] = img[:,:,:]
    #pad = np.zeros([256,256,3])
    #pad [64:192,64:192,:] = img[:,:,:]
    #pad = np.zeros([384,384,3])
    #pad [64:320,64:320,:] = img[:,:,:]
    #pad = np.zeros([512,512,3])
    #pad [64:448,64:448,:] = img[:,:,:]
    #pad = np.zeros([512,512,3])
    #pad [128:384,128:384,:] = img[:,:,:]
    imsave(savepath+file,pad)
    #i += 1
    '''
    if "input" in file or "target" in file:
         os.remove(path+file)
    '''
