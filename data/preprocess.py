import numpy as np
import os
import glob
import h5py
from PIL import Image


IMG_SIZE = 64


def load_Img(fname):
    img = Image.open(fname)
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    img = np.asarray(img)/255.
    
    if len(img.shape) < 3:
        rgb_img = np.zeros([IMG_SIZE, IMG_SIZE, 3], dtype=np.float32)
        rgb_img[:, :, 0] = rgb_img[:, :, 1] = rgb_img[:, :, 2] = img
        return rgb_img

    return img


file_list = glob.glob('./data/celebA/*.jpg')
IMG_NUM = len(file_list)

rand_idx = np.arange(IMG_NUM)
np.random.shuffle(rand_idx)

HDF5_FILE_WRITE = 'celeba.hdf5'
fw = h5py.File(HDF5_FILE_WRITE, 'w')
images = fw.create_dataset('images', (IMG_NUM, IMG_SIZE, IMG_SIZE, 3), dtype='float32')

for i in xrange(IMG_NUM):
    idx = rand_idx[i]
    images[i] = load_Img(file_list[idx])
    
    if i % 1000 == 0:
        print '%.1f %% preprocessed.' % (100.*i/IMG_NUM)
    
fw.close()

