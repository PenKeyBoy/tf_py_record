# coding=utf-8
import tensorflow as tf
import numpy as np
import optparse
import os
import codecs
import glob
from tensorflow.python.platform import gfile

from random import random

pic_extensions = ["jpg","JPG","jpeg","JPEG"]
class PhotoDataDeal():
    def __init__(self,args):
        self.image_size = args.image_size
        self.train_percent = args.split_tr
        self.valid_percent = args.split_va
        #self.test_percent = args.split_te
        self.photo_path = args.photo_path
        self.write2fpath = args.write2fpath
    @staticmethod
    def get_fileList(path):
        file_list = []
        #print("path:",path)
        for sub in os.listdir(path):
            subpath = os.path.join(path,sub)
            if os.path.isfile(subpath) and sub.split('.')[-1] in pic_extensions:
                    #file_name = os.path.join(path,sub)
                    file_list.extend(glob.glob(subpath))
            if not file_list:
                continue
        #print("file list:\n",file_list)
        return file_list
    
    def gen_data(self,sess,shuffle=True):
        current_level = 0
        train_images = []
        train_labels = []
        valid_images = []
        valid_labels = []
        test_images = []
        test_labels = []
        for subdir in os.listdir(self.photo_path):
            #print("subdir:",subdir)
            subpath = os.path.join(self.photo_path,subdir)
            if os.path.isdir(subpath):

                file_list = self.get_fileList(subpath)
        
                for file in file_list:
                    #print("file:",file)
                    raw_data = gfile.FastGFile(file,'rb').read()
                    image_raw = tf.image.decode_jpeg(raw_data)
                    if image_raw.dtype != tf.float32:
                        #print("image_raw type:",image_raw.dtype)
                        image_date = tf.image.convert_image_dtype(image_raw,tf.float32)
                        image = tf.image.resize_images(image_date,[self.image_size,self.image_size])
                        image_value = sess.run(image)
                    random_num = random()
                    if random_num <= self.train_percent:
                        train_images.append(image_value)
                        train_labels.append(current_level)
                    elif random_num > self.train_percent and random_num <= self.valid_percent:
                        valid_images.append(image_value)
                        valid_labels.append(current_level)
                    else:
                        test_images.append(image_value)
                        test_labels.append(current_level)
            current_level += 1
        if shuffle:
            random_state = np.random.get_state()
            np.random.shuffle(train_images)
            np.random.set_state(random_state)
            np.random.shuffle(train_labels)
        dataToSave = np.asarray([train_images,train_labels,valid_images,valid_labels,test_images,test_labels])
        np.save(self.write2fpath,dataToSave)
        return dataToSave

if __name__ == '__main__':
    usAge = "UsAge:%prog [options]"
    description = "deal photo data and save"
    opts = optparse.OptionParser(usage=usAge,description=description)
    opts.add_option('-i','--image_size',dest='image_size',help="resize image pixel",default=299)
    opts.add_option('-t','--train_split',dest='split_tr',help="train split size",default=0.8)
    opts.add_option('-v','--valid_split',dest="split_va",help="valid split size",default=0.9)
    opts.add_option('-p','--photo_path',dest="photo_path",help="photo path to read",default='photo_data/flower_photos')
    opts.add_option('-w','--write2file',dest='write2fpath',help="write data to file")
    opt,args = opts.parse_args()
    photoData = PhotoDataDeal(opt)
    if not os.path.exists('image_data'):
        os.mkdir('image_data')
    with tf.Session() as sess:
        res = photoData.gen_data(sess)
        print("res first element:\n",res[0])