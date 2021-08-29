# coding=utf-8
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

font_size = 12
figure_size = (10,8)
class AnimationPlot():
    def __init__(self,xlim,ylim,xlabel,ylabel,frames,title):
        self.xlim,self.ylim = xlim,ylim
        self.xlabel,self.ylabel = xlabel,ylabel
        self.title = title
        self.frames = self.gen_data(frames)
        self.fig,self.ax = plt.subplots()

        self.lines, = self.ax.plot([],[],'g--',linewidth=2,animated=False)
        self.ani = FuncAnimation(self.fig,func=self.update,frames=self.frames,init_func=self.init_graph,blit=True)
        plt.show()
    def init_graph(self):
        self.font_prop = mpl.font_manager.FontProperties(fname='font-prop/SIMSUN.TTC')
        plt.rcParams['font.size'] = font_size
        plt.rcParams['figure.figsize'] = figure_size
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.grid.axis'] = 'y'
        plt.rcParams['grid.color'] = 'grey'
        
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        
        self.ax.set_xlim(self.xlim[0],self.xlim[1])
        self.ax.set_ylim(self.ylim[0],self.ylim[1])
        self.ax.set_yticks(list(range(0,100,10)))
        self.ax.set_title(self.title)
        return self.lines,

    def gen_data(self,data_list):
        for data_x,data_y,title in data_list:
            yield data_x,data_y,title

    def update(self,frame):
        self.ax.set_title(frame[-1])
        self.lines.set_data(frame[0],frame[1])
        return self.lines,
    
    @classmethod
    def save_fig(cls,path):
        plt.savefig(path)

    @classmethod
    def close_fig(cls):
        plt.close()