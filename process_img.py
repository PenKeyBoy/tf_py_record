# coding=utf-8
from matplotlib import pyplot as plt
import tensorflow as tf
import os
import random
#from tensorflow.python.platform import gfile

img_path = 'photo_data/flower_photos/daisy'

def plt_first_img(path):
    for i,img in enumerate(os.listdir(img_path)):
        img_read = plt.imread(os.path.join(img_path,img))
        plt.imshow(img_read)
        if i > 0:
            break
        i += 1

def decode_encode_jpg(img_fpath):
    img_raw = tf.gfile.FastGFile(img_fpath,'rb').read()
    pic_save = tf.gfile.FastGFile('figure/img_deal.jpg','wb')
    with tf.Session() as sess:
        img_data = tf.image.decode_jpeg(img_raw)
        print("img_data value:",img_data.eval())
        plt.imshow(img_data.eval())
        plt.show()
        img_data = tf.image.encode_jpeg(img_data)
        pic_save.write(img_data.eval())

def disorted_jpg(image_data,disorted_order):

    if disorted_order == 0:
        img_data = tf.image.random_brightness(image_data,max_delta=35. / 255.)
        img_data = tf.image.random_hue(img_data,max_delta=0.5)
        img_data = tf.image.random_contrast(img_data,0.5,1.5)
        img_data = tf.image.random_saturation(img_data,lower=0.5,upper=1.5)
    elif disorted_order == 1:
        img_data = tf.image.random_saturation(image_data,lower=0.5,upper=1.5)
        img_data = tf.image.random_brightness(img_data,max_delta=35. / 255.)
        img_data = tf.image.random_hue(img_data,max_delta=0.5)
        img_data = tf.image.random_contrast(img_data,0.5,1.5)
    elif disorted_order == 2:
        img_data = tf.image.random_contrast(image_data,0.5,1.5)
        img_data = tf.image.random_saturation(img_data,lower=0.5,upper=1.5)
        img_data = tf.image.random_brightness(img_data,max_delta=35. / 255.)
        img_data = tf.image.random_hue(img_data,max_delta=0.5)
    elif disorted_order == 3:
        img_data = tf.image.random_hue(image_data,max_delta=0.5)
        img_data = tf.image.random_contrast(img_data,0.5,1.5)
        img_data = tf.image.random_saturation(img_data,lower=0.5,upper=1.5)
        img_data = tf.image.random_brightness(img_data,max_delta=35. / 255.)
    else:
        img_data = image_data
    return tf.clip_by_value(img_data,0.0,1.0)

def proccess_img(img_data,height,width,boxes):
    if boxes is None:
        boxes = tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4])
    # convert image data type
    if img_data.dtype != tf.float32:
        img_data = tf.image.convert_image_dtype(img_data,dtype=tf.float32)
    # slice image
    bbox_begain,bbox_size,bbox_draw = tf.image.sample_distorted_bounding_box(tf.shape(img_data),bounding_boxes=boxes,min_object_covered=0.4)
    distorted_img = tf.slice(img_data,bbox_begain,bbox_size)
    distorted_img = tf.image.resize_images(distorted_img,[height,width],method=random.randint(0,3))
    distorted_img = tf.image.random_flip_up_down(distorted_img)
    distorted_img = tf.image.random_flip_left_right(distorted_img)
    distorted_img = disorted_jpg(distorted_img,random.randint(0,3))
    return distorted_img

def main(img_path):
    img_raw = tf.gfile.FastGFile(img_fpath,'rb').read()
    img_data = tf.image.decode_jpeg(img_raw)
    boxes = tf.constant([[[0.1,0.1,0.7,0.9],[0.25,0.4,0.8,0.75]]])
    with tf.Session() as sess:
        for i in range(6):
            distorted_img = proccess_img(img_data,299,299,boxes)
            plt.imshow(distorted_img.eval())
            plt.show()


if __name__=='__main__':
    #plt_first_img(img_path)
    img_fpath = 'photo_data/flower_photos/daisy/2511306240_9047015f2d_n.jpg'
    #decode_encode_jpg(img_fpath)
    main(img_fpath)