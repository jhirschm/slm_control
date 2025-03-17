from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import numpy as np
import time
import threading
from os import getcwd, path, makedirs

def plot_creation(x_data, y_data, live = True, rescale_x = True, rescale_y = True, hide = False, save_path = None, fig = None, x_lim = None, y_lim = None, x_label = None, y_label = None, title = None):
    if fig is None:
        fig = plt.figure(figsize=(16/2,12/2))        
        ax = fig.add_subplot(1,1,1)    

    if title is not None:
        pass
        # fig.set_window_title(title)

    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        ax.set_ylabel(y_label)
        ax.axes.grid()

    if x_lim is not None:
        ax.set_xlim(x_lim)
    else:
        ax.set_xlim((x_data.min(), x_data.max()))

    if y_lim is not None:
        ax.set_ylim(y_lim)
    else:
        ax.set_ylim((y_data.min(), y_data.max()))

    line, = ax.plot(x_data, y_data,'r',lw=3)

    if save_path is not None:
        plot_save(dir=save_path)

    if not hide:
        conn = Queue()
        keep_alive = Process(target=plot_show, args=(fig,ax,line,conn,rescale_x,rescale_y,live))
        keep_alive.start()
        
    return conn, fig, keep_alive

def plot_update(conn, x_data, y_data, update = True, save_path = None):
    time.sleep(0.05)
    if not conn.empty():
        x_data, y_data, update = conn.get(timeout=0.3)
        update = False
    if update:
        conn.put([x_data, y_data, update])
    if save_path is not None:
        plot_save(dir=save_path)
    return update
    

def plot_show(fig,ax,line,conn,rescale_x,rescale_y,live):
    if live:
        plot = plot_maintain(fig,ax,line,conn,rescale_x,rescale_y)
        plot.start()
    plt.show()

def plot_save(dir=None):
    if dir == None:
        dir = getcwd()+'/plots'
    plt.savefig(dir)

class plot_maintain(threading.Thread):
    def __init__(self,fig,ax,line,conn,rescale_x,rescale_y):
        threading.Thread.__init__(self)
        self.line = line
        self.conn = conn
        self.fig = fig
        self.ax = ax
        self.rescale_x = rescale_x
        self.rescale_y = rescale_y
        self.update = True

    def run(self):
        while plt.fignum_exists(self.fig.number) and self.update:
            if not self.conn.empty():
                x_data, y_data, self.update = self.conn.get(timeout=0.3)
                self.line.set_xdata(x_data)
                self.line.set_ydata(y_data)
                if self.rescale_x:
                    self.ax.set_xlim((x_data.min(), x_data.max()))
                if self.rescale_y:
                    self.ax.set_ylim((y_data.min(), y_data.max()))
                plt.draw()
            time.sleep(0.01)
        if self.update:
            if not self.conn.empty():
                x_data, y_data, self.update = self.conn.get(timeout=0.3)
                self.conn.put([x_data, y_data, False])

    def stop(self):
        self.update = False

def plot_make_data(phase,lim):
    x = np.linspace(1,7*lim/10,100)
    y = np.sin(x+phase*7/20)
    return x, y

def main():
    # sample routines
    folder = getcwd() + '\\plots\\'
    if not path.exists(folder):
        makedirs(folder)

    x, y = plot_make_data(0,10)
    conn, fig, keep_alive_1 = plot_creation(x, y)
    for k in range(1,21):
        x, y = plot_make_data(k,10)
        plot_update(conn,x,y)
        if k == 10:
            plot_save(folder + '1')
        time.sleep(0.1)
    plot_update(conn,x,y,False)

    x, y = plot_make_data(1,0)
    conn, fig, keep_alive_2 = plot_creation(x, y)
    for k in range(1,10):
        x, y = plot_make_data(0,k)
        plot_update(conn,x,y)
        time.sleep(0.1)
    plot_update(conn,x,y,False)
    plot_save(folder + '2')

    plot_creation(x,y*y,live=False)
    plot_save(folder + '3')
 

if __name__ == '__main__':
    main()