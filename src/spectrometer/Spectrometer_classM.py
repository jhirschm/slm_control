"""
Created on Tue Oct 11 20:51:07 2022

@author: Slawa
"""

import os
import sys
Path=os.path.dirname((os.path.abspath(__file__)))
sys.path.append(Path)
SP=Path.split("\\")
i=0
while i<len(SP) and SP[i].find('python')<0:
    i+=1
Pypath='\\'.join(SP[:i+1])
sys.path.append(Pypath)

from avantes.Avantes_spec import AvaSpec
from oceanoptics.OOptSpec import OOptSpec
from broadcom.Qmini import Qmini
from emulator_spec import SpecEmulator
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
# from avantes.avaspec import *

class Spectrometer():
    def __init__(self):
        self.find()
        self.connected=False
        
        self.CF0={'vendor' : float('nan'),
                'SN' : float('nan'),
                'range' : float('nan'),
                'NexusID': float('nan'),
                'userID': float('nan')}
        
    def find(self,autoconnect=False):
        """find spectometers"""
        #avantes
        self.avantes=AvaSpec()
        self.ocean=OOptSpec()
        self.qmini=Qmini()
        self.devices=[]
        self.devices4GUI=[]
        for i in range(self.avantes.Ndev):
            # print(i)
            d=self.avantes.devconfig[i]
            self.devices.append(["Avantes",d[0].decode("utf-8"),d[1]]) #[Spectrometer type, serial number, device ID]
            self.devices4GUI.append(["Avantes",d[0].decode("utf-8")])#,np.array2string(np.array(d[2]),precision=1)])
            
        for i in range(self.ocean.Ndev):
            # print(i)
            d=self.ocean.devconfig[i]
            self.devices.append(["Ocean Optics", d[0], d[2]]) #[Spectrometer type, serial number, model]
            self.devices4GUI.append(["Ocean Optics", d[0]])
            
        for i in range(self.qmini.Ndev):
            # print(i)
            d=self.qmini.devconfig[i]
            self.devices.append(["Qmini", d[0], d[2]]) #[Spectrometer type, serial number, model]
            self.devices4GUI.append(["Qmini", d[0]])
                
        #emulator
        self.devices.append(["Emulator",1,1]) #[Spectrometer type, serial number, device ID]
        self.devices4GUI.append(["Emulator",1])
        if len(self.devices)==1:
            self.devices.append(["Emulator",2,2])
            self.devices4GUI.append(["Emulator",2])
            self.devices.append(["Emulator",3,3])
            self.devices4GUI.append(["Emulator",3])
        if autoconnect:
            self.connect()
        
    def connect(self, DeviceN=[0]):
        """connects to the device number DeviceN in the list of found devices"""
        self.config_parameters=[copy.deepcopy(self.CF0) for n in range(len(DeviceN))]
        self.spectrometer=[None for n in range(len(DeviceN))]
        self.Sconnected=[False for n in range(len(DeviceN))]
        self.SN=[None for n in range(len(DeviceN))]
        self.lam=[None for n in range(len(DeviceN))]
        i=0
        for DN in DeviceN:
            if DN < len(self.devices):
                if self.devices[DN][0]=="Avantes":
                    self.spectrometer[i]=copy.deepcopy(self.avantes)
                    self.spectrometer[i].connect(deviceID=self.devices[DN][2])
                    self.dev_handle=self.spectrometer[i].dev_handle
                    self.lam[i]=self.spectrometer[i].lam #wavelenghts
                    self.connected=True
                    self.Sconnected[i]=True
                    self.SN[i]=self.spectrometer[i].SN
                elif self.devices[DN][0]=="Ocean Optics":
                    self.spectrometer[i]=copy.copy(self.ocean)
                    self.spectrometer[i].connect(SN=self.devices[DN][1])
                    self.dev_handle=self.spectrometer[i].dev_handle
                    self.lam[i]=self.spectrometer[i].lam
                    self.connected=True
                    self.Sconnected[i]=True
                    self.SN[i]=self.spectrometer[i].SN
                elif self.devices[DN][0]=="Qmini":
                    self.spectrometer[i]=copy.copy(self.qmini)
                    self.spectrometer[i].connect(SN=self.devices[DN][1])
                    self.dev_handle=self.spectrometer[i].dev_handle
                    self.lam[i]=self.spectrometer[i].lam
                    self.connected=True
                    self.Sconnected[i]=True
                    self.SN[i]=self.spectrometer[i].SN
                elif self.devices[DN][0]=="Emulator":
                    self.spectrometer[i]=SpecEmulator()
                    self.connected=True
                    self.Sconnected[i]=True
                    self.lam[i]=self.spectrometer[i].lam
                    self.SN[i]=self.devices[DN][1]
                    self.spectrometer[i].SN=self.devices[DN][1]
                
                # print(self.config_parameters[i])
                self.config_parameters[i]['vendor']=self.spectrometer[i].Type
                self.config_parameters[i]['SN']=self.SN[i]
                self.config_parameters[i]['range']=[self.lam[i].min(),self.lam[i].max()]
            else:
                print("Device number is out of range")
            i+=1
        print(self.Sconnected)
    
    def config_measure(self,Tintegration=0.01,Naverage=1,HighRes=True,IntensityCalibration=False,ExtTrig=False):
        """configurates spectrometer
        Tintegration is the integration time in ms
        Naverage is the number of spectra to average
        
        Avantes:
            HighRes True enables 16 bit resolution (65535 max value), 
            false uses 14 bit resolution (16383 max value)"""
        if self.connected:
            for i in range(len(self.spectrometer)):
                if self.Sconnected[i]:
                    self.spectrometer[i].config_measure(Tintegration,Naverage,HighRes,IntensityCalibration,ExtTrig) 
            self.measurement_configed=True
        else:
            print("no connection established yet")
        
    def measure(self,Nspec=1):
        """take data
        Nspec is the number of spectra to measure"""
        if self.connected:
            self.spectrum=[None for n in range(len(self.spectrometer))]
            for i in range(len(self.spectrometer)):
                if self.Sconnected[i]:
                    self.spectrometer[i].measure(Nspec=Nspec)
                    self.spectrum[i]=self.spectrometer[i].spec
        else:
            print("no connection established yet")
    
    def show_spec(self,title=None):
        if self.connected:
            X=self.lam
            Y=self.spectrum[0]
            
            plt.plot(X,Y,linewidth=3)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel('wavelength ($\mu$m)',fontsize=18)
            plt.ylabel('',fontsize=18)
            if not title == None:
                plt.title(title,fontsize=18)
        else:
            print("no connection established yet")
    
    def start_measure(self,Nspec=1):
        """start measure but dont wait for ending it"""
        for i in range(len(self.spectrometer)):
            if self.Sconnected[i]:
                self.spectrometer[i].start_measure(Nspec)
    
    def stop_measure(self):
        """stop measurement"""
        for i in range(len(self.spectrometer)):
            if self.Sconnected[i]:
                self.spectrometer[i].stop_measure()
        
    def isdataready(self):
        ready=True
        for i in range(len(self.spectrometer)):
            if self.Sconnected[i]:
                ready = ready and self.spectrometer[i].isdataready()
        return ready
        
    def read_data(self):
        data=[None for n in range(len(self.spectrometer))]
        for i in range(len(self.spectrometer)):
            if self.Sconnected[i]:
                data[i]=self.spectrometer[i].read_data()
        return data
    
    def disconnect(self):
        """disconnect device"""
        for i in range(len(self.spectrometer)):
            if self.Sconnected[i]:
                self.spectrometer[i].disconnect()
    


#test

# S=Spectrometer()
# S.connect()
# S.config_measure()
# S.measure()
    
# plt.figure(1)
# plt.clf()
# S.show_spec()
    
    
    
# S=MultiSpectrometer()
# T=0.1
# Nav=1
# S.connect([0,1])
# S.config_measure([T,T],[Nav,Nav])
# S.measure()
    
# plt.figure(1)
# plt.clf()
# S.show_spec()
    
    
    
    
    