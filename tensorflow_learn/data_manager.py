# coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
from numpy.random import random

class DataManager():
    def __init__(self,mnist_path,config={}):
        self.image = config.get("image")
        self.image_size = None
        self.channel = None
        self.mnist = input_data.read_data_sets(mnist_path,one_hot=True)
        self.validation_labels = self.mnist.validation.labels
        self.test_labels = self.mnist.test.labels
        if not self.image:
            self.validation_images = self.mnist.validation.images
            self.test_images = self.mnist.test.images
        else:
            self.image_size = config['image_size']
            self.channel = config['channel']
            valid_size = self.mnist.validation.images.shape[0]
            test_size = self.mnist.test.images.shape[0]
            self.validation_images = self.mnist.validation.images.reshape([valid_size,self.image_size,self.image_size,self.channel])
            self.test_images = self.mnist.test.images.reshape([test_size,self.image_size,self.image_size,self.channel])

    def iterbatch(self,batch_size):
        batch_ls = [self.mnist.train.next_batch(batch_size) for _ in random(100)]
        for images,labels in batch_ls:
            if not self.image:
                yield images,labels
            else:
                if self.image_size == 28:
                    images_rs = images.reshape([batch_size,self.image_size,self.image_size,self.channel])
                    yield images_rs,labels
                else:
                    raise ValueError("no suitable image_size for train_data reshape") 
