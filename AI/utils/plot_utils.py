import os
import itertools
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib
import mpl_toolkits.axisartist.axislines as axislines

def plot_error_curve(error_train,error_test,xlabel='Epoch',ylabel='Accuary(%)',title='',legend_label=['train', 'test'],save_dir='results/figures',save_filename='Accuary_curve',fontsize=13):
    n=len(error_train)
    x = np.arange(1, n+1, 1)
    fig = plt.figure()
    plt.plot(x,error_train)
    plt.plot(x,error_test)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    if len(title)>0:
        plt.title(title, fontsize=fontsize+5)
    label =legend_label
    plt.legend(label, loc=0, ncol=2)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig.savefig(os.path.join(save_dir,save_filename+'.jpg'),dpi=200)
    fig.savefig(os.path.join(save_dir,save_filename+'.png'),dpi=200)
    fig.savefig(os.path.join(save_dir,save_filename+'.pdf'))
    fig.savefig(os.path.join(save_dir,save_filename+'.eps'))
    plt.close(fig)
    print("Plot error curve finished")


