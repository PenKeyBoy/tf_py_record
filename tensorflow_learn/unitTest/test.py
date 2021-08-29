# coding=utf-8
from numpy.random import rand
import sys
import os
from matplotlib import pyplot as plt
sys.path.append(os.getcwd())
#print(sys.path)
import time

#from mnist_main import plot_figure
from ani_plot import AnimationPlot


if __name__=='__main__':
    #plt.ion()
    xlab = 'rand input'
    ylab = 'square out'
    data_list = []
    #fig,ax=plt.subplots()

    for e in range(100):
        #plt.clf()
        title = "the %d epoch now present" %e
        x = range(10)
        f = lambda a:a**2
        y = [f(i)/(e+1) for i in x]
        data_list.append([x,y,title])
        title = 'test graph work or not'
        #fig = plt.figure(figsize=(14,10))
        #plt.plot(x,y,'--o')
        #plt.draw()
        #plt.show()
        #plot_figure(plt,x,y,xlab,ylab,title)
        #fig.savefig('figure/test.jpg')
        #time.sleep(0.1)
    AnimationPlot((0,10),(0,100),xlab,ylab,data_list,title)
    AnimationPlot.save_fig('figure/test.jpg')
    #AnimationPlot.close_fig()