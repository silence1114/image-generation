#-*- coding:utf-8 -*-
from PIL import Image
import os.path
import numpy as np
from scipy.misc import imsave
path = "d:/pcgan/preprocess/train_data/origin_data/"
savepath = [''] * 8
savepath[0] = "d:/pcgan/preprocess/train_data/crop_data/face/"
savepath[1] = "d:/pcgan/preprocess/train_data/crop_data/512/"
savepath[2] = "d:/pcgan/preprocess/train_data/crop_data/384/"
savepath[3] = "d:/pcgan/preprocess/train_data/crop_data/384pad512/"
savepath[4] = "d:/pcgan/preprocess/train_data/crop_data/256/"
savepath[5] = "d:/pcgan/preprocess/train_data/crop_data/256pad384/"
savepath[6] ="d:/pcgan/preprocess/train_data/crop_data/128/"
savepath[7] = "d:/pcgan/preprocess/train_data/crop_data/128pad256/"

for path_ in savepath:
    if not os.path.exists(path_):
        os.makedirs(path_)

filelist = os.listdir(path)
for file in sorted(filelist):
    filepath = path + file
    img = Image.open(filepath)
    img = np.array(img)
    file = file.split('.jpg.jpg')[0] + '.jpg'
 
    crop = img[100:980,155:870,:] 
    imsave(savepath[0]+file,crop)

    im = Image.open(savepath[0]+file)
    crop512 = im.resize((512,512),Image.BILINEAR) #512x512
    imsave(savepath[1]+file,crop512) 
    crop512 = np.array(crop512)
	
    crop384 = crop512[64:448,64:448,:]
    imsave(savepath[2]+file,crop384)
    pad384_512 = np.zeros([512,512,3])
    pad384_512[64:448,64:448,:] = crop512[64:448,64:448,:]
    imsave(savepath[3]+file,pad384_512)

    crop256 = crop384[64:320,64:320,:]
    imsave(savepath[4]+file,crop256)
    pad256_384 = np.zeros([384,384,3])
    pad256_384[64:320,64:320,:] = crop384[64:320,64:320,:]
    imsave(savepath[5]+file,pad256_384)


    crop128 = crop256[64:192,64:192,:]
    imsave(savepath[6]+file,crop128)
    pad128_256 = np.zeros([256,256,3])
    pad128_256[64:192,64:192,:] = crop256[64:192,64:192,:]
    imsave(savepath[7]+file,pad128_256)
  
   


