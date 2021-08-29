# coding=utf-8
#from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os

from data_manager import DataManager

class TFRecordData(DataManager):
    def __init__(self,mnist_path,write2path):
        DataManager.__init__(self,mnist_path)
        self.num_sample = len(self.validation_images)
        self.pixels = self.validation_images.shape[1]       #dimension=784
        self.writer = tf.python_io.TFRecordWriter(write2path)
    
        self.tfRecordData = write2path
        for index in range(self.num_sample):
            image_string = self.validation_images[index].tostring()
            tfData = tf.train.Example(features=tf.train.Features(
                feature={
                    "labels":self._int64Tofeature(np.argmax(self.validation_labels[index])),
                    "pixels":self._int64Tofeature(self.pixels),
                    "image_raw":self._byte2feature(image_string)
                }
            ))
            self.writer.write(tfData.SerializeToString())
        self.writer.close()
    def _int64Tofeature(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _byte2feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def readTFrecord(self,sess,iter=10):
        reader = tf.TFRecordReader()
        filename_queue = tf.train.string_input_producer([self.tfRecordData])
        _,serialize_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialize_example,features={
            "pixels":tf.FixedLenFeature([],tf.int64),
            "labels":tf.FixedLenFeature([],tf.int64),
            "image_raw":tf.FixedLenFeature([],tf.string)})
        image = tf.decode_raw(features["image_raw"],tf.uint8)
        labels = tf.cast(features["labels"],tf.int32)
        pixels = tf.cast(features["pixels"],tf.int32)
        coordinate = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coordinate)
        for i in range(iter):
            print (sess.run([image,labels,pixels]))
        coordinate.join(threads)


if __name__=='__main__':
    if not os.path.exists('TFRecordData'):
        os.mkdir('TFRecordData')
    tfRecordData = TFRecordData('data','TFRecordData/out.TFRecord')
    with tf.Session() as sess:

        tfRecordData.readTFrecord(sess)


        




