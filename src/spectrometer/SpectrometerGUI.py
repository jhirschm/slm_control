"""
spectrometer GUI

@author: Slawa
add
single shot
"""

import csv
from tkinter import filedialog

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
import error_class as ER
import numpy as np
import time

import inspect

from classes.Pulse_class import width

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


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(idx)

class SpecGUI(QMainWindow):
# class SpecGUI(QDialog):
    def __init__(self,show=True,DisableDisconnect=False):
        self.app=QApplication(sys.argv)
        super(SpecGUI, self).__init__()
        uic.loadUi(Path+"\\Qt\\spectrometer.ui", self)
        # uic.loadUi(Path+"\\Qt\\spectrometer_dialog.ui", self)
        if show:
            self.show()
        self.connectwindow=ConnectWindow()
        
        self.DisableDisconnect=DisableDisconnect
        self.connected=False
        self.spectrometer=Spectrometer()
        self.SubstractBkg=False
        self.sampling=False
        self.folder = None
        self.roi = False
        self.plot = None
        self.spec=[]
        
        self.connectBt.clicked.connect(self.showconnect)
        self.saveBt.clicked.connect(self.save)
        self.configBt.clicked.connect(self.config)
        # self.connectBt.clicked.connect(self.Open)
        self.TakeBkgBt.clicked.connect(self.takebkg)
        self.BkgBt.clicked.connect(self.bkgmode)
        self.startBt.clicked.connect(self.run)
        self.Tint.valueChanged.connect(self.setAcquisition)
        self.Averaging.valueChanged.connect(self.setAcquisition)
        self.ExtTrig.clicked.connect(self.setAcquisition)
        self.ICalib.clicked.connect(self.setAcquisition)
        self.ScaleVerBt.clicked.connect(self.scaleV)
        self.ScaleHorBt.clicked.connect(self.scaleH)
        self.VerCursor.toggled.connect(self.v_cursor)
        self.HorizCursor.toggled.connect(self.h_cursor)
        self.logBtn.clicked.connect(self.start_log)
        self.roiBtn.toggled.connect(self.show_roi)
        self.log_f.textChanged.connect(self.up_log)
        self.openBt.clicked.connect(self.Open)
        self.clipBt.clicked.connect(self.clip)
        self.CleanBt.clicked.connect(self.clear)
        
        
        self.connectwindow.cancelBt.clicked.connect(self.cancel_connection)
        self.connectwindow.connectBt.clicked.connect(self.connect)
        
        self.Msleeptime=0.05 #sleep time to check inputs while taking the data
        
        self.scaleV()
        self.scaleH()
        self.vLine = None
        self.hLine = None

        self.Dtimer = QtCore.QTimer(self)
        self.Dtimer.setInterval(200) #.2 seconds
        self.Dtimer.timeout.connect(self.displayPosition)
        self.Dtimer.start()
        # #timer to check inputs
        # self.Dtimer = QtCore.QTimer(self)
        # self.Dtimer.setInterval(200) #.2 seconds
        # self.Dtimer.timeout.connect(self.checkinputs)
        # self.Dtimer.start()
        
        self.DataBackground=[] #buffer for background files to show

    def choose_folder(self):
        self.folder = filedialog.asksaveasfilename()

    def up_log(self):
        self.LogTimer.setInterval(int(self.log_f.value()))

    def show_roi(self):
        self.roi = self.roiBtn.isChecked()
        if self.roi:
            self.roiMax = pg.InfiniteLine(angle=90, movable=True, pen=(0, 0, 0), pos=(self.spectrometer.lam.mean() + 10))
            self.roiMin = pg.InfiniteLine(angle=90, movable=True, pen=(0, 0, 0), pos=(self.spectrometer.lam.mean() - 10))
            self.plot.addItem(self.roiMin)
            self.plot.addItem(self.roiMax)
        else:
            self.plot.removeItem(self.roiMin)
            self.plot.removeItem(self.roiMax)





    def v_cursor(self):
        if self.VerCursor.isChecked():
            if not self.vLine: self.vLine = pg.InfiniteLine(angle=90, movable=True, pen=(0,0,0), pos=self.spectrometer.lam.mean())
            self.plot.addItem(self.vLine, ignoreBounds=True)
        else:
            self.plot.removeItem(self.vLine)

    def h_cursor(self):
        if self.HorizCursor.isChecked():
            if self.SubstractBkg:
                Y = self.spec - self.bkg
            else:
                Y = self.spec
            if not self.hLine: self.hLine = pg.InfiniteLine(angle=0, movable=True, pen=(0,0,0), pos=Y.mean())
            self.plot.addItem(self.hLine, ignoreBounds=True)
        else:
            self.plot.removeItem(self.hLine)

    def checkinputs(self):
        self.app.processEvents()
        
    def displayPosition(self):
        if self.vLine:
            self.vpos.display(self.vLine.value())
        if self.hLine:
            self.hpos.display(self.hLine.value())

    
    def showconnect(self):
        """open connection interface"""
        connected=self.connectBt.isChecked()
        if not connected:
            self.disconnect()
        else:
            self.connectwindow.show()
            data=self.spectrometer.devices
            print('----------------------')
            print(self.spectrometer.devices)
            for devdd in self.spectrometer.devices4GUI:
                print(devdd)
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
        
        #show one spectrum
        self.plot = pg.PlotWidget()
        self.plot.getAxis('bottom').setLabel(text='Wavelength nm', font_weight='bold')
        self.plot.getAxis('left').setLabel(text='Intensity a.u.', font_weight='bold')
        self.plot.showGrid(x=True, y=True, alpha=0.7)
        self.plot.resize(self.SpecView.size()*0.99)
        scene = QGraphicsScene(self)
        scene.addWidget(self.plot)
        self.SpecView.setScene(scene)
        
        self.sampling=True
        self.startBt.setChecked(True)
        self.getspectrum()
        self.startBt.setChecked(False)
        self.sampling=False
                
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
        
    def save(self, file=None):
        """save spectrum"""
        if file == None or file == False:
            file = filedialog.asksaveasfilename()
        if file != '':
            if len(self.spec) > 2:
                X=self.spectrometer.lam
                if self.SubstractBkg:
                    Y=self.spec-self.bkg
                else:
                    Y=self.spec
                file=file.split('.spec')[0]
                np.savetxt(file+'.spec', np.concatenate((np.array(X).reshape((-1, 1)),
                                                        np.array(Y).reshape((-1, 1))), axis=1),
                           header='# wavelength nm\t intensity a.u. ' +
                           '<>\t' + str(self.spectrometer.config_parameters['vendor'])+'<>'+
                           ' SN'+str(self.spectrometer.config_parameters['SN'])+ '\t' +'<>'+
                           'Tintegration ' + str(self.Tintegration)+'<>'+
                           ' Naverage '+ str(self.Naverage)+ '<>'+
                           '\t' + str(time.asctime(time.gmtime(time.time()))),
                           delimiter='\t', comments='')
    
    def Open(self):
        """open saved spectrum"""
        file = filedialog.askopenfilename(filetypes=[("Spec", ".spec")])
        if file != '':
            data=np.loadtxt(file,skiprows=1)
            lam=data[:,0]
            Int=data[:,1]
            # print(lam,Int)
            self.DataBackground.append([lam,Int])
        self.ShowPlot()
    
    def clip(self):
        if self.SubstractBkg:
            y = self.spec-self.bkg
        else:
            y = self.spec
        self.DataBackground.append([self.spectrometer.lam.copy(),y])
        self.ShowPlot()
        
    def clear(self):
        self.DataBackground=[]
        self.ShowPlot()
    
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
        self.status.setPlainText(text)
    
    def addshowstatus(self, text):
        Ctext=self.status.toPlainText()
        self.status.setPlainText(Ctext + text)
    
    def setAcquisition(self):
        """change the integration time"""
        Tint=self.Tint.value()
        self.Tintegration=Tint
        Nav=int(self.Averaging.value())
        self.Naverage=Nav
        self.ExternalTrig=self.ExtTrig.isChecked()
        self.Icalib=self.ICalib.isChecked()
        self.spectrometer.config_measure(Tintegration=Tint,Naverage=Nav,ExtTrig=self.ExternalTrig,IntensityCalibration=self.Icalib)
        
        if self.ExternalTrig and self.spectrometer.spectrometer.Type!= 'Avantes':
            self.showstatus('external trigger works only for Avantes')
        if self.Icalib and self.spectrometer.spectrometer.Type!= 'Qmini':
            self.addshowstatus('\n I calibration works only for Qmini')
            
    
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

        
        # if self.spectrometer.config_parameters['vendor']=='Avantes':
        #     self.spectrometer.spectrometer.start_measure()

        while self.sampling:
            #start the measurement
            self.app.processEvents()
            self.getspectrum()
            time.sleep(self.Msleeptime)
        #     self.sampling=False
        # self.sampling=True
        # self.getspectrum()
        # time.sleep(self.Msleeptime)
        # self.getspectrum()
                
    def getspectrum(self):
        # print(inspect.getmembers(self.spectrometer,predicate=inspect.ismethod))
        # if not self.spectrometer.config_parameters['vendor']=='Avantes':
        #     self.spectrometer.spectrometer.start_measure()
        self.spectrometer.spectrometer.start_measure()
        self.Tstart=time.time()
        self.WaitBar.setMaximum(int(self.Tintegration))
        self.WaitBar.setMinimum(0)
        self.app.processEvents()
        #wait for the measurement to be done

        while (not self.spectrometer.spectrometer.isdataready()) and self.sampling:
            time.sleep(self.Msleeptime)
            t=(time.time()-self.Tstart)*1000
            self.WaitBar.setValue(int(t))
            self.app.processEvents()
            # print('waiting in GUI')
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
            
    def start_log(self,Log_file=None):
        if self.logBtn.isChecked():
            if Log_file:
                self.Log_file = filedialog.asksaveasfilename()
            self.LogTimer = QtCore.QTimer(self)
            self.LogTimer.setInterval(int(self.log_f.value()))
            # print(int(self.log_f.value()))
            self.LogTimer.timeout.connect(self.log_spec)
            self.LogTimer.start()
            
            self.firstLog=True
        else:
            self.LogTimer.stop()
            self.Log_file = None
            
    def log_spec(self):
        if self.Log_file != False and self.Log_file != '':
            x = self.spectrometer.lam
            if self.SubstractBkg:
                y = self.spec-self.bkg
            else:
                y = self.spec
    
            if self.roi:
                min = find_nearest(x, self.roiMin.value())
                max = find_nearest(x, self.roiMax.value())
                x = x[(min + 1):max]
                y = y[(min + 1):max]
            
                
            if self.firstLog:
                self.firstLog=False
                
        
                # if os.path.isfile(self.Log_file):
                #     ref = np.loadtxt(self.Log_file, delimiter='\t')
                # else:
                ref = np.insert(x, 0, np.nan)
                now = time.localtime(time.time())
                self.Ttstart=float(time.time())
                y = np.insert(y, 0, 0)
                data = np.row_stack((ref, y))
                #fmt='%f',
                
                np.savetxt(self.Log_file+'.logspec', data, 
                           header='# start time year:month:day:hh:mm:ss    {}:{}:{}:{}:{}:{}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour,
                                                                  now.tm_min, now.tm_sec)+'<>' +  str(self.spectrometer.config_parameters['vendor'])+'<>'+
                                                                  ' SN'+str(self.spectrometer.config_parameters['SN'])+ '\t' +'<>'+
                                                                  'Tintegration ' + str(self.Tintegration)+'<>'+
                                                                  ' Naverage '+ str(self.Naverage),
                           delimiter='\t', comments='')
                
            else:
                # print(y)
                y = np.insert(y, 0, time.time()-self.Ttstart)
                # print(self.Ttstart,time.time(),time.time()-self.Ttstart)
                with open(self.Log_file+'.logspec', "a") as f:
                    np.savetxt(f, y.reshape(1, -1),delimiter='\t', comments='')
                
        # with open(self.file, 'a', newline='') as tsvfile:
        #     writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        #     writer.writerow(['X', 'Y'])
        #     for k, l in zip(x, y):
        #         if not self.roi or (k > self.roiMin.value() and k < self.roiMax.value()):
        #             writer.writerow([k, l])
        
    def ShowPlot(self):
        """show the last spectrum"""
        plot = self.plot
        plot.clear()
        
        if len(self.DataBackground) > 0:
            LINECOLORS = ['r', 'g', 'c', 'm', 'y', 'k']
            Colors=LINECOLORS*10
            for i in range(len(self.DataBackground)):
                Xb=self.DataBackground[i][0]
                Yb=self.DataBackground[i][1]
                plot.plot(Xb,Yb,pen=pg.mkPen(Colors[i], width=2.5))
                
        X=self.spectrometer.lam
        if self.SubstractBkg:
            Y=self.spec.copy()-self.bkg.copy()
        else:
            Y=self.spec.copy()
        plot.plot(X,Y, pen=pg.mkPen((0,0,255), width=2.5))

        if self.ShowLam.isChecked():
            Y1=Y.copy()
            ind=Y1<Y1.max()*0.05
            Y1[ind]=0
            lamC=np.sum(X*Y1)/np.sum(Y1)
            line=pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('g', width=2),label=str("%.1f" % lamC))
            line.setPos(lamC)
            plot.addItem(line, ignoreBounds=True)
            self.lam_cent.display(lamC)
            
        if self.ShowWidth.isChecked():
            Wmethod=self.width_type.currentText()
            W=width(X,Y,method=Wmethod)
            self.lam_width.display(W)
            
            if Wmethod == 'FWHM':
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

        if self.VerCursor.isChecked():
            plot.addItem(self.vLine, ignoreBounds=True)
        if self.HorizCursor.isChecked():
            plot.addItem(self.hLine, ignoreBounds=True)

        if self.roiBtn.isChecked():
            plot.addItem(self.roiMin)
            plot.addItem(self.roiMax)
        
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
            
    
    def scaleV(self):
        """change the vertical scaling behaiviour (fixed or adjusting)"""
        self.AutoScaleV=self.ScaleVerBt.isChecked()
    
    def scaleH(self):
        """change the vertical scaling behaiviour (fixed or adjusting)"""
        self.AutoScaleH=self.ScaleHorBt.isChecked()
        
    def closeEvent(self, event):
        """call at closing"""
        if not self.DisableDisconnect:
            self.stop()
            self.disconnect()
    
class ConnectWindow(QDialog):
    def __init__(self):
        super(ConnectWindow, self).__init__()
        uic.loadUi(Path+"\\Qt\\connect.ui", self)
        
        

if __name__ == '__main__':

    pg.setConfigOption('background', 'w')
    app = QApplication([])
    window = SpecGUI()

    app.exec_()