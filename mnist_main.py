# coding=utf-8
import logging
import os
import json
import codecs
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

from data_manager import DataManager
from model import MinistLearn
from ani_plot import AnimationPlot

flags = tf.app.flags

flags.DEFINE_float('lr',0.02,'learning rate')
flags.DEFINE_float('dr',0.96,'decay rate')
flags.DEFINE_integer('ds',100,'decay steps')
flags.DEFINE_float('edr',0.99,'exponential moving average decay rate')
flags.DEFINE_float('lp',0.3,'l2_regularizer lambda param')
flags.DEFINE_integer('clip',5,'avoid the gradient ecplision')
flags.DEFINE_integer('bs',100,'batch size')
flags.DEFINE_integer('units1',50,'hidden layer1 units number')
flags.DEFINE_integer('units2',10,'hidden layer2 units number')
flags.DEFINE_integer('image_size',28,'size of image')
flags.DEFINE_integer('channel',1,'RGB of image')
flags.DEFINE_integer('conv1',5,'size of conv1')
flags.DEFINE_integer('conv1_deep',50,'filters of conv1')
flags.DEFINE_integer('conv2_deep',100,'filters of conv2')
flags.DEFINE_integer('conv2',5,'size of conv2')
flags.DEFINE_integer('fc',150,'nodes of full connection layer1')
flags.DEFINE_integer('input_dimension',784,'input dimension')
flags.DEFINE_integer('num_tags',10,'tags number')
flags.DEFINE_integer('epoch',20,'epoch for train')
flags.DEFINE_string('opt','Adam','optimial algorithm')
flags.DEFINE_string('data_path','data','path to store mnist data')
flags.DEFINE_string('ckpt_path','ckpt','path to save model')
flags.DEFINE_string('config_file','config.cfg','file to save config')
flags.DEFINE_string('log_file',os.path.join('log','train.log'),'log file for train')
flags.DEFINE_string('figure_path',os.path.join('figure','accuracy.png'),'file to save figures')
flags.DEFINE_boolean('image',False,'specify the model')

FLAGS = tf.app.flags.FLAGS

def config_model():
    config = {}
    config['lr'] = FLAGS.lr
    config['dr'] = FLAGS.dr
    config['ds'] = FLAGS.ds
    config['edr'] = FLAGS.edr
    config['lp'] = FLAGS.lp
    config['bs'] = FLAGS.bs
    config['clip'] = FLAGS.clip
    config['epoch'] = FLAGS.epoch
    config['layer1_units'] = FLAGS.units1
    config['layer2_units'] = FLAGS.units2
    config['image_size'] = FLAGS.image_size
    config['channel'] = FLAGS.channel
    config['conv1'] = FLAGS.conv1
    config['cd1'] = FLAGS.conv1_deep
    config['conv2'] = FLAGS.conv2
    config['cd2'] = FLAGS.conv2_deep
    config['fc'] = FLAGS.fc
    config['input_dimension'] = FLAGS.input_dimension
    config['num_tags'] = FLAGS.num_tags
    config['opt'] = FLAGS.opt
    config['image'] = FLAGS.image
    return config

def get_log(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

def save_config(config,config_file):
    #config_json = json.dumps(config)
    with codecs.open(config_file,mode='w',encoding='utf-8') as fw:
        json.dump(config,fw,ensure_ascii=False,indent=4)

def load_config(config_file):
    with open(config_file,'r',encoding='utf-8') as fr:
        return json.load(fr)

def print_config(config,logger):
        for k,v in config.items():
            logger.info('{}:\t{}'.format(k.ljust(15),v))

def make_path(FLAGS):
    #if not os.path.exists(FLAGS.data_path):
    #    os.makedirs(FLAGS.data_path)
    if not os.path.exists(FLAGS.ckpt_path):
        os.makedirs(FLAGS.ckpt_path)
    if not os.path.exists("log"):
        os.makedirs("log")
    if not os.path.exists("figure"):
        os.makedirs("figure")


def plot_figure(x,y,xlab,ylab,xlim,ylim,yticks,title,font_size=14,figure_size=[12,10]):
    #mpl.rcParams['font.family'] = ['sans-serif']
    #plt.ion()
    #plt.clf()
    font_props = mpl.font_manager.FontProperties(fname='font-prop/SIMSUN.TTC')
    mpl.rcParams['font.size'] = font_size
    mpl.rcParams['figure.figsize'] = figure_size
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['axes.grid.axis'] = 'y'
    mpl.rcParams['grid.color'] = 'grey'
    mpl.rcParams['grid.linestyle'] = '--'
    mpl.rcParams['legend.edgecolor'] = 'blue'
    mpl.rcParams['legend.fancybox'] = True
    mpl.rcParams['legend.loc'] = 'best'
    mpl.rcParams['legend.shadow'] = True
    #ax.set_title(title)
    plt.title(title)
    plt.xlim(xlim[0],xlim[1])
    plt.ylim(ylim[0],ylim[1])
    plt.yticks(yticks)
    plt.xlabel(xlab)
    #ax.set_xlabel(xlab)
    #ax.set_ylabel(ylab)
    plt.ylabel(ylab)
    plt.plot(x,y,'--o')
    #plt.legend(loc='best',bbox_to_anchor=(0.5,0.5),prop=font_props)
    #plt.draw()
    return plt

def train(model_class,config,data_path,ckpt_path,logger):
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    model = model_class(config)
    dataManager = DataManager(data_path,config)
    maximum = 0
    with tf.Session() as sess:
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            model.saver.restore(sess,ckpt.model_checkpoint_path) 
        else:
            sess.run(tf.global_variables_initializer())
            xlabel = "steps per epoch"
            ylabel = "loss value"
            data_list = [];accuracy_ls = []
            max_loss = 0.0
            for e in range(config['epoch']):
                steps_per_epoch = 0
                loss_ls = []
                title = "loss value perform from steps at %d epoch" %e
                for image_tr,label_tr in dataManager.iterbatch(model.batch_size):
                    global_step,loss,_ = sess.run([model.global_step,model.loss,model.train_op],feed_dict={model.input_x:image_tr,model.tag:label_tr})
                    steps_per_epoch += 1
                    if e % 10 == 0:
                        logger.info("iteration %d: global_step=%d; loss_value=%.6f" %(e,global_step,loss))
                    loss_ls.append(loss)
                #plot_figure(range(steps_per_epoch),loss_ls,'steps per epoch','loss value','loss value perform from steps at %d_th epoch' %e)
                #logits_avg = sess.run(model.logits_avg,feed_dict={model.input_x:dataManager.validation_images})
                #logger.info("steps_per_epoch={} and forecast result shape:\n{}".format(steps_per_epoch,np.array(logits_avg).shape))
                #accuracy = model.accuracy
                accuracy_value = sess.run(model.accuracy,feed_dict={model.input_x:dataManager.validation_images,model.tag:dataManager.validation_labels})
                accuracy_ls.append(accuracy_value)
                data_list.append([range(steps_per_epoch),loss_ls,title])
                max_loss_epoch = max(loss_ls)
                if max_loss_epoch > max_loss:
                    max_loss = max_loss_epoch 
                if accuracy_value > maximum:
                    maximum = accuracy_value
                    logger.info("saving model ...")
                    model.saver.save(sess,os.path.join(FLAGS.ckpt_path,'mnist.ckpt'),global_step=global_step)
                    #logits_avg = sess.run(model.logits_avg,feed_dict={model.input_x:dataManager.test_images})
                
                    test_accuracy_value = sess.run(model.accuracy,feed_dict={model.input_x:dataManager.test_images,model.tag:dataManager.test_labels})
                    logger.info("iteration %d: accuracy=%9.3f%%" %(e,test_accuracy_value*100))    #9:tab quantities
            xlim = (0,config['epoch']);ylim = (0,1)
            yticks = np.arange(0,1,0.05)
            figure = plot_figure(range(config['epoch']),accuracy_ls,"epoch_numbers","accuracy",xlim,ylim,yticks,"the validation accuracy perform belong to the epoch numbers")
            figure.savefig(FLAGS.figure_path)
            xlim=(0,steps_per_epoch);ylim = (0.0,max_loss)
            AnimationPlot(xlim,ylim,xlabel,ylabel,data_list,title)
def main(_):
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model()
        save_config(config,FLAGS.config_file)
    logger = get_log(FLAGS.log_file)
    print_config(config,logger)
    #mnist_data = 'path/to/mnist_data'
    train(MinistLearn,config,FLAGS.data_path,FLAGS.ckpt_path,logger)

if __name__ == '__main__':
    tf.app.run()
