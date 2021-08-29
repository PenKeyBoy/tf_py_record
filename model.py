# coding=utf-8
import os
import tensorflow as tf
from data_manager import DataManager
from tensorflow.contrib.layers.python.layers import initializers



class MinistLearn(object):
    def __init__(self,config):
        #config is the configuration info of superparmaters with format python dict
        self.input_dimension = config['input_dimension']
        self.num_tags = config['num_tags']
        self.input_x = tf.placeholder(dtype=tf.float32,shape=[None,config['input_dimension']],name='input_image')
        self.tag = tf.placeholder(dtype=tf.float32,shape=[None,config['num_tags']],name='tag_number')

        self.learning_rate = config['lr']
        self.decay_rate = config['dr']
        self.decay_steps = config['ds']
        self.ema_decay_rate = config['edr']
        self.lambda_param = config['lp']
        self.batch_size = config['bs']
        self.units = {"layer1":config['layer1_units'],"layer2":config['layer2_units']}
        # params for Lnet5
        self.image_size = config['image_size']
        self.channel = config['channel']
        self.conv1 = config['conv1']
        self.conv1_deep = config['cd1']
        self.conv2 = config['conv2']
        self.conv2_deep = config['cd2']
        self.full_connect = config['fc']
        # reshape the input image data with three dimensions
        self.new_input = tf.placeholder(dtype=tf.float32,shape=[None,self.image_size,self.image_size,self.channel],name='input')
        
        
        #define global_step to record learning step
        self.global_step = tf.Variable(0,dtype=tf.int32,name='global_step',trainable=False)
        self.initializer = initializers.xavier_initializer()
        #average the params for forward porperation
        moving_average_op = tf.train.ExponentialMovingAverage(self.ema_decay_rate,self.global_step)
        if not config["image"]:
            self.logits = self.logitLayer(self.input_x,self.units,None,name='logits')
            #compute average
            moving_average_var = moving_average_op.apply(tf.trainable_variables())
            self.logits_avg = self.logitLayer(self.input_x,self.units,moving_average_op,name='logits_avg')
        else:
            self.logits = self.inference(self.new_input,None,name='logits')
            #compute average
            moving_average_var = moving_average_op.apply(tf.trainable_variables())
            self.logits_avg = self.inference(self.new_input,moving_average_op,name='logits_avg')
        self.loss = self.lossLayer(self.logits,self.tag,self.lambda_param,name='losses')
        # evaluate
        equal_boolean = tf.equal(tf.argmax(self.logits_avg,axis=1),tf.argmax(self.tag,axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(equal_boolean,tf.float32))
        
        if config['opt'] == 'Adam':
            self.opt = tf.train.AdamOptimizer(self.learning_rate)
        else:
            learning_rate = tf.train.exponential_decay(self.learning_rate,self.global_step,decay_steps=self.decay_steps,decay_rate=self.decay_rate,name='learning_rate_decay')
            self.opt = tf.train.GradientDescentOptimizer(learning_rate)
        grad_var = self.opt.compute_gradients(self.loss)
        clip_grad = [[tf.clip_by_value(g,-config['clip'],config['clip']),v] for g,v in grad_var]
        train_op = self.opt.apply_gradients(clip_grad,self.global_step,name='update_params')
        
        self.train_op = tf.group(train_op,moving_average_var)
        self.saver = tf.train.Saver(tf.global_variables())

    def logitLayer(self,input_x,hidden_units,avg_class=None,name='logits'):
        self.W = []
        #input_dimension = tf.shape(input_x)[-1]
        with tf.variable_scope('params_layer1',reuse=tf.AUTO_REUSE):
            W = tf.get_variable('weight_layer1',dtype=tf.float32,shape=[self.input_dimension,hidden_units["layer1"]],initializer=self.initializer)
            b = tf.get_variable('bias_layer1',dtype=tf.float32,shape=[hidden_units["layer1"]],initializer=self.initializer)
            if not avg_class:
                layer_out = tf.nn.relu(tf.matmul(input_x,W) + b)
            else:
                layer_out = tf.nn.relu(tf.matmul(input_x,avg_class.average(W)) + avg_class.average(b))
        
        with tf.variable_scope('params_layer2',reuse=tf.AUTO_REUSE):
            W = tf.get_variable('weight_layer2',dtype=tf.float32,shape=[hidden_units["layer1"],hidden_units["layer2"]],initializer=self.initializer)
            b = tf.get_variable('bias_layer2',dtype=tf.float32,shape=[hidden_units["layer2"]],initializer=self.initializer)
            if not avg_class:
                logit = tf.matmul(layer_out,W) + b
            else:
                logit = tf.matmul(layer_out,avg_class.average(W)) + avg_class.average(b)
        return logit
        """
        with tf.variable_scope('params_hidden',reuse=tf.AUTO_REUSE):
            input_dimension = self.input_dimension
            count = 0;layer_collection = []
            for layer in hidden_units:
                W = tf.get_variable('wight_params',dtype=tf.float32,shape=[input_dimension,hidden_units[layer]],initializer=self.initializer)
                b = tf.get_variable('bias_params',dtype=tf.float32,shape=[hidden_units[layer]],initializer=self.initializer)
    
                if not avg_class:
                    layer_out = tf.nn.relu(tf.matmul(input_x,W) + b)
                else:
                    layer_out = tf.nn.relu(tf.matmul(input_x,avg_class.average(W)) + avg_class.average(b))
                input_dimension = hidden_units[layer]
                input_x = layer_out
                count += 1
                layer_collection.append(layer)
                self.W.append(W)
                if count >= len(hidden_units) - 1:
                    break
        with tf.variable_scope('params_hidden_last',reuse=tf.AUTO_REUSE):
            for layer in hidden_units:
                if layer not in layer_collection:
                    output_dimension = hidden_units[layer]
                    break
            W = tf.get_variable('weight_param',dtype=tf.float32,shape=[input_dimension,output_dimension],initializer=self.initializer)
            b = tf.get_variable('bias_param',dtype=tf.float32,shape=[output_dimension],initializer=self.initializer)
            self.W.append(W)
            if not avg_class:
                logits = tf.matmul(layer_out,W) + b
            else:
                logits = tf.matmul(layer_out,avg_class.average(W)) + avg_class.average(b)
        return logits
        """

    def inference(self,input,avg_class=None,name='Lenet5-logit'):
        with tf.variable_scope('conv2D_1',reuse=tf.AUTO_REUSE):
            conv1_w1 = tf.get_variable('conv1_w1',dtype=tf.float32,shape=[self.conv1,self.conv1,self.channel,self.conv1_deep],initializer=self.initializer)
            conv1 = tf.nn.conv2d(input,conv1_w1,strides=[1,1,1,1],padding='SAME')
            conv1_bias = tf.get_variable('conv1_bias',dtype=tf.float32,shape=[self.conv1_deep],initializer=self.initializer)
            conv1_out = tf.nn.relu(conv1 + conv1_bias,name='conv1_out')
        with tf.variable_scope('pool2D_1',reuse=tf.AUTO_REUSE):
            pool1 = tf.nn.max_pool(conv1_out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pool1')
        #conv2 and pool2
        with tf.variable_scope('conv2D_2',reuse=tf.AUTO_REUSE):
            conv2_w2 = tf.get_variable('conv2_w2',dtype=tf.float32,shape=[self.conv2,self.conv2,self.conv1_deep,self.conv2_deep],initializer=self.initializer)
            conv2 = tf.nn.conv2d(pool1,conv2_w2,strides=[1,1,1,1],padding='SAME')
            conv2_bias = tf.get_variable('conv2_bias',dtype=tf.float32,shape=[self.conv2_deep],initializer=self.initializer)
            conv2_out = tf.nn.relu(conv2 + conv2_bias,name='conv2_out')
        with tf.variable_scope('pool2D_2',reuse=tf.AUTO_REUSE):
            pool2 = tf.nn.max_pool(conv2_out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pool2')
        
        #flattern
        with tf.variable_scope('flatten'):
            reshape_pool2 = pool2.get_shape().as_list()
            flatten_size = reshape_pool2[1]*reshape_pool2[2]*reshape_pool2[3]
            flatten_out = tf.reshape(pool2,[-1,flatten_size],name='flatten')
            
        #full_connect1
        self.W = []
        with tf.variable_scope('fc1',reuse=tf.AUTO_REUSE):
            full_weight1 = tf.get_variable('full_connect_w1',shape=[flatten_size,self.full_connect],initializer=self.initializer)
            full_bias1 = tf.get_variable('full_connect_b1',shape=[self.full_connect],initializer=self.initializer)
            if not avg_class:

                full_out1 = tf.nn.relu(tf.matmul(flatten_out,full_weight1)+full_bias1)
            else:
                full_out1 = tf.nn.relu(tf.matmul(flatten_out,avg_class.average(full_weight1))+avg_class.average(full_bias1))
            self.W.append(full_weight1)
        #dropout
        drop_out1 = tf.nn.dropout(full_out1,keep_prob=0.5,name='dropout1')
        #full_connect2
        with tf.variable_scope('fc2',reuse=tf.AUTO_REUSE):
            full_weight2 = tf.get_variable('full_connect_w2',shape=[self.full_connect,self.num_tags],initializer=self.initializer)
            full_bias2 = tf.get_variable('full_connect_b2',shape=[self.num_tags],initializer=self.initializer)
            if avg_class:
                logits = tf.matmul(drop_out1,avg_class.average(full_weight2)) + avg_class.average(full_bias2)
            else:
                logits = tf.matmul(drop_out1,full_weight2) + full_bias2
            self.W.append(full_weight2)
        return logits

    def lossLayer(self,logits,real_tags,lambda_param,name='losses'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=tf.argmax(real_tags,axis=-1))
        cross_entropy_loss = tf.reduce_mean(cross_entropy)
        #add regular to collection
        for w in self.W:
            tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lambda_param)(w))
        tf.add_to_collection('losses',cross_entropy_loss)
        loss = tf.add_n(tf.get_collection('losses'))
        return loss
    def evaluate(self,preds,real_tags,name='evaluate'):
        pass
        
    



                

        

    
         
        
