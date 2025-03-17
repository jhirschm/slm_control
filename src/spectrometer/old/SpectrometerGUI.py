"""
spectrometer GUI

@author: Slawa

todo:
    memory usage
    add power indicator
"""

from PyQt5.QtWidgets import *
from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import pyqtgraph as pg
import pandas as pd

import os
import sys
Path=os.path.dirname((os.path.abspath(__file__)))
sys.path.append(Path)

SP=Path.split("\\")
i=0
while i<len(SP) and SP[i].find('python')<0:
    i+=1
import sys
Pypath='\\'.join(SP[:i+1])
sys.path.append(Pypath)

from Spectrometer_class import Spectrometer
import classes.error_class as ER
import numpy as np
import time

import inspect

#class for table construction
class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            # See below for the nested-list data structure.
            # .row() indexes into the outer list,
            # .column() indexes into the sub-list
            return self._data[index.row()][index.column()]

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._data)

    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        return len(self._data[0])

class SpecGUI(QMainWindow):
# class SpecGUI(QDialog):
    def __init__(self,show=True):
        self.app=QApplication(sys.argv)
        super(SpecGUI, self).__init__()
        uic.loadUi(Path+"\\Qt\\spectrometer.ui", self)
        # uic.loadUi(Path+"\\Qt\\spectrometer_dialog.ui", self)
        if show:
            self.show()
        self.connectwindow=ConnectWindow()
        
        self.connected=False
        self.spectrometer=Spectrometer()
        self.spectrum=[]
        self.SubstractBkg=False
        self.sampling=False
        
        self.connectBt.clicked.connect(self.showconnect)
        self.saveBt.clicked.connect(self.save)
        self.configBt.clicked.connect(self.config)
        # self.connectBt.clicked.connect(self.Open)
        self.TakeBkgBt.clicked.connect(self.takebkg)
        self.BkgBt.clicked.connect(self.bkgmode)
        self.startBt.clicked.connect(self.run)
        self.Tint.valueChanged.connect(self.setAcquisition)
        self.Averaging.valueChanged.connect(self.setAcquisition)
        self.ScaleVerBt.clicked.connect(self.scaleV)
        self.ScaleHorBt.clicked.connect(self.scaleH)
        
        
        self.connectwindow.cancelBt.clicked.connect(self.cancel_connection)
        self.connectwindow.connectBt.clicked.connect(self.connect)
        
        self.Msleeptime=0.05 #sleep time to check inputs while taking the data
        
        self.scaleV()
        self.scaleH()
        
        # #timer to check inputs
        # self.Dtimer = QtCore.QTimer(self)
        # self.Dtimer.setInterval(200) #.2 seconds
        # self.Dtimer.timeout.connect(self.checkinputs)
        # self.Dtimer.start()
    
    def checkinputs(self):
        self.app.processEvents()
    
    def showconnect(self):
        """open connection interface"""
        connected=self.connectBt.isChecked()
        if not connected:
            self.disconnect()
        else:
            self.connectwindow.show()
            data=self.spectrometer.devices4GUI
            print(data)
            self.model = TableModel(data)
            self.connectwindow.table.setModel(self.model)
        
    def connect(self):
        """open connection interface"""
        try:
            index = self.connectwindow.table.selectedIndexes()[0].row()
            self.spectrometer.connect(index)
            # print("SN: ",self.spectrometer.SN)
        except ER.SL_exception as error:
            self.showerror(error)
            self.connectBt.setChecked(False)
        else:
            self.connected=True
            self.showstatus('connected')
            self.bkg=np.zeros(len(self.spectrometer.lam))
        self.connectwindow.hide()
        self.setAcquisition()
        # print(self.spectrometer)
        # print(self.spectrometer.lam)
                
    def cancel_connection(self):
        self.connectwindow.hide()
        self.connectBt.setChecked(False)
        self.connected=False
        # raise ER.SL_exception("connection canceled")
            
    def disconnect(self):
        """disconnect the motor"""
        self.connected=False
        self.showstatus('disconnected')
        try:
            self.spectrometer.disconnect()
        except ER.SL_exception as error:
            self.showerror(error)
        
    def save(self):
        """save spectrum"""
        pass
    
    def Open(self):
        """open saved spectrum"""
        pass
        
    def config(self):
        """configurate the spectrometer"""
        pass
        
    def showerror(self,error):
        self.status.setPlaceholderText(error.Message) 
        
        #make red text
        # self.error_message.setTextColor(QColor(255, 0, 0))
        # self.error_message.setAcceptRichText(True)
        # print('<p style="color: red">'+error.Message+'</p>')
        # self.error_message.setPlaceholderText('<p style="color: red">'+error.Message+'</p>')    
    
    def showstatus(self,text):
        self.status.setPlaceholderText(text)
    
    def setAcquisition(self):
        """change the integration time"""
        Tint=self.Tint.value()
        self.Tintegration=Tint
        Nav=int(self.Averaging.value())
        self.Naverage=Nav
        self.spectrometer.config_measure(Tintegration=Tint,Naverage=Nav)
    
    def run(self):
        """run or stop the spectrometer"""
        if self.startBt.isChecked():
            self.sampling=True
            self.start()
        else:
            self.sampling=False
            self.stop()
            
    def start(self):
        """start measuring and displaying spectra"""
        while self.sampling:
            #start the measurement
            self.getspectrum()
                
    def getspectrum(self):
        # print(inspect.getmembers(self.spectrometer,predicate=inspect.ismethod))
        self.spectrometer.spectrometer.start_measure()
        #wait for the measurement to be done
        while (not self.spectrometer.spectrometer.isdataready()) and self.sampling:
            time.sleep(self.Msleeptime)
            self.app.processEvents()
        if self.sampling:
            self.spec=np.array(self.spectrometer.spectrometer.read_data()[0])[:len(self.spectrometer.lam)]
            self.ShowPlot()
    
    def stop(self):
        """stop acquisition"""
        self.spectrometer.stop_measure()
        
    def takebkg(self):
        """take current spectrum as background"""
        if len(self.spec)>0:
            self.bkg=self.spec.copy()
            
    def bkgmode(self):
        if self.BkgBt.isChecked():
            self.SubstractBkg=True
        else:
            self.SubstractBkg=False
        
    def ShowPlot(self):
        """show the last spectrum"""
        X=self.spectrometer.lam
        if self.SubstractBkg:
            Y=self.spec-self.bkg
        else:
            Y=self.spec
        
        View=self.SpecView
        scene = QGraphicsScene(self)
        plot=pg.PlotWidget(labels={'left': 'Intensity a.u.', 'bottom': 'Wavelength nm'})
        plot.plot(X,Y, pen=pg.mkPen((0,0,255), width=2.5))
        plot.showGrid(x = True, y = True, alpha = 0.7) #alpha (0.0-1.0) Opacity of the grid
        
        if self.ShowLam.isChecked():
            Y1=Y.copy()
            ind=Y1<Y1.max()*0.05
            Y1[ind]=0
            lamC=np.sum(X*Y1)/np.sum(Y1)
            line=pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('g', width=2),label=str("%.1f" % lamC))
            line.setPos(lamC)
            plot.addItem(line, ignoreBounds=True)
            
        if self.ShowWidth.isChecked():
            M=Y.max()
            N1=np.argwhere(Y>=M*0.5)[0][0]
            N2=np.argwhere(Y>=M*0.5)[-1][0]
            X1=X[N1]
            X2=X[N2]
            W=X2-X1
            Xmin=X.min()
            Xmax=X.max()
            Expansion=1.0045
            line=pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('g', width=2),label=str("%.1f" % W)
                                  ,span=((X1*Expansion-Xmin/Expansion)/(Xmax*Expansion-Xmin/Expansion),
                                         (X2*Expansion-Xmin/Expansion)/(Xmax*Expansion-Xmin/Expansion)))
            #,bounds=[X1,X2]
            line.setPos(M/2)
            # line=plot.plot([X1,X2],[M/2,M/2],pen=pg.mkPen('g', width=2))
            plot.addItem(line, ignoreBounds=True)
        
        if not self.AutoScaleV:
            plot.setYRange(self.SetYmin.value(),self.SetYmax.value())
        else:
            plot.setYRange(Y.min(),Y.max())
            self.SetYmin.setValue(Y.min())
            self.SetYmax.setValue(Y.max())
        
        if not self.AutoScaleH:
            plot.setXRange(self.SetXmin.value(),self.SetXmax.value())
        else:
            plot.setXRange(X.min(),X.max())
            self.SetXmin.setValue(X.min())
            self.SetXmax.setValue(X.max())
            
        plot.resize(View.size()*0.99)
        scene.addWidget(plot)
        View.setScene(scene)
    
    def scaleV(self):
        """change the vertical scaling behaiviour (fixed or adjusting)"""
        self.AutoScaleV=self.ScaleVerBt.isChecked()
    
    def scaleH(self):
        """change the vertical scaling behaiviour (fixed or adjusting)"""
        self.AutoScaleH=self.ScaleHorBt.isChecked()
    
class ConnectWindow(QDialog):
    def __init__(self):
        super(ConnectWindow, self).__init__()
        uic.loadUi(Path+"\\Qt\\connect.ui", self)
        
        
        
if __name__ == '__main__':
    pg.setConfigOption('background', 'w')
    app = QApplication([])
    window = SpecGUI()

    app.exec_()