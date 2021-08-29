# coding=utf-8
import tensorflow as tf
import time
import random
import threading

def loop_coordinate(coord,work_id):
    #coord = tf.train.Coordinator()
    while not coord.should_stop():
        if random.random() < 0.01:
            print("stop from work id:",work_id)
            coord.request_stop()
        else:
            print("current word id:",work_id)
        time.sleep(1)

if __name__=='__main__':
    coord = tf.train.Coordinator()
    threads = []
    for i in range(5):
        print("creat thread %d" %i)
        thread = threading.Thread(target=loop_coordinate,args=(coord,i,))
        thread.start()
        threads.append(thread)
    # for thread in threads:
    #     thread.start()
    coord.join(threads)
