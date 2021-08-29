# coding=utf-8
import tensorflow as tf
from numpy.random import RandomState

def playGame(layers,lr=0.02,batch_size=100,loss_type=0):
    x_ = tf.placeholder(dtype=tf.float32,shape=[None,2],name="input_x")
    y_ = tf.placeholder(tf.float32,[None,1],name="input_y")
    w_dict = {};b_dict = {}
    input_shape = 2
    output_shape = 1
    for layer in layers:
        dimension_shape = [input_shape,layers[layer]]
        w_dict[layer],b_dict[layer] = get_var(dimension_shape,0.01)
        #w_dict[layer] = tf.Variable(tf.random_normal([input_shape,layers[layer]]),name='w_' + layer)
        #b_dict[layer] = tf.Variable(tf.zeros(layers[layer],name='b_' + layer))
        input_shape = layers[layer]
    hidden_out = hidden(x_,w_dict,b_dict)
    w_dict['output'] = tf.Variable(tf.random_normal([input_shape,output_shape]),name='w_output')
    b_dict['bias'] = tf.Variable(tf.zeros(output_shape),name='b_output')
    output = tf.nn.sigmoid(tf.matmul(hidden_out,w_dict['output'])+b_dict['bias'])
    if loss_type == 0:  #0:represent the base cross_entroy loss
        loss = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(output,0.001,0.999)) + (1-y_) * tf.log(1-tf.clip_by_value(output,0.001,0.999)))
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    else:
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=output)
        global_step = tf.Variable(0.0,name='global_step')
        #cross by the decay expotienal to change learning_rate with global_step zoom up
        learning_rate = tf.train.exponential_decay(lr,global_step,decay_steps=100,decay_rate=0.99)
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    tf.add_to_collection('losses',loss)
    loss_all = tf.add_n(tf.get_collection('losses'))


    rdm = RandomState(1)
    X = rdm.randn(1000,2)
    Y = [[float(int(x1 + x2 +rdm.randint(-10,10)/batch_size < 2))] for x1,x2 in X]
    init = tf.global_variables_initializer()
    with tf.Session(graph=tf.get_default_graph()) as sess:
        sess.run(init)
        for i in range(100):
            batch_start = i*batch_size % X.shape[0]
            batch_end = min(batch_start + batch_size,X.shape[0])
            w_res,b_res,loss_value,_ = sess.run([w_dict,b_dict,loss_all,train_op],feed_dict={x_:X[batch_start:batch_end],y_:Y[batch_start:batch_end]})
            print("iteration {} loss={}".format(i,loss_value))
            for layer in w_res:
                print("layer {} weight update:\n{}".format(layer,w_res[layer]))


def hidden(input_x,w,b):
    input_layer = input_x
    for layer in w:
        input_layer = tf.nn.relu(tf.matmul(input_layer,w[layer])+b[layer])
    return input_layer

def get_var(dimension_shape,lambda_param):
    w = tf.Variable(tf.random_normal(shape=dimension_shape))
    b = tf.Variable(tf.zeros(dimension_shape[-1]))
    tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(lambda_param)(w))
    return w,b

def move_averge(sess,decay,init,name='move_avg'):
    var = tf.Variable(0,dtype=tf.int32,name=name)
    step = tf.Variable(0,trainable=False)
    exp_avg = tf.train.ExponentialMovingAverage(decay,num_updates=step)
    move_average_op = exp_avg.apply([var])
    sess.run(init)
    sess.run(move_average_op)
    step,v,exp_avg_val = sess.run([step,var,exp_avg.average(var)])
    print("initial current step {} variable value={} and exp_avg value={}".format(step,v,exp_avg_val))
    sess.run([tf.assign(step,100),tf.assign(var,5)])
    sess.run(move_average_op)
    step,v,exp_avg_val = sess.run([step,var,exp_avg.average(var)])
    print("update current step {} variable value={} and exp_avg value={}".format(step,v,exp_avg_val))



if __name__ == '__main__':
    hidden_layers = {"layer1":2,"layer2":3,"layer3":3}
    playGame(hidden_layers)


        