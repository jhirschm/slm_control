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
        
        self.config_parameters={'vendor' : float('nan'),
                                 'SN' : float('nan'),
                                 'range' : float('nan'),
                                 'NexusID': float('nan'),
                                 'userID': float('nan')}
        
    def find(self,autoconnect=False):
        """find spectometers"""
        #avantes
        self.avantes=AvaSpec()
        self.devices=[]
        self.devices4GUI=[]
        for i in range(self.avantes.Ndev):
            # print(i)
            d=self.avantes.devconfig[i]
            self.devices.append(["Avantes",d[0].decode("utf-8"),d[1]]) #[Spectrometer type, serial number, device ID]
            self.devices4GUI.append(["Avantes",d[0].decode("utf-8"),np.array2string(np.array(d[2]),precision=1)])
        #emulator
        self.devices.append(["Emulator",1,1]) #[Spectrometer type, serial number, device ID]
        self.devices4GUI.append(["Emulator",1])
        if len(self.devices)==1:
            self.devices.append(["Emulator",2,2])
            self.devices4GUI.append(["Emulator",2])
            self.devices.append(["Emulator",3,3])
            self.devices4GUI.append(["Emulator",3])
        print(self.devices)
        if autoconnect:
            self.connect()
        
    def connect(self, DeviceN=0):
        """connects to the device number DeviceN in the list of found devices"""
        if DeviceN < len(self.devices):
            if self.devices[DeviceN][0]=="Avantes":
                self.spectrometer=copy.deepcopy(self.avantes)
                self.spectrometer.connect(deviceID=self.devices[DeviceN][2])
                self.dev_handle=self.spectrometer.dev_handle
                self.lam=self.spectrometer.lam #wavelenghts
                self.connected=True
                self.SN=self.spectrometer.SN
            elif self.devices[DeviceN][0]=="Emulator":
                self.spectrometer=SpecEmulator()
                self.connected=True
                self.lam=self.spectrometer.lam
                self.SN=self.devices[DeviceN][1]
                self.spectrometer.SN=self.SN
                
            self.config_parameters['vendor']=self.spectrometer.Type
            self.config_parameters['SN']=self.SN
            self.config_parameters['range']=[self.lam.min(),self.lam.max()]
        else:
            print("Device number is out of range")
    
    def config_measure(self,Tintegration=0.01,Naverage=1,HighRes=True):
        """configurates spectrometer
        Tintegration is the integration time in ms
        Naverage is the number of spectra to average
        
        Avantes:
            HighRes True enables 16 bit resolution (65535 max value), 
            false uses 14 bit resolution (16383 max value)"""
        if self.connected:
            self.spectrometer.config_measure(Tintegration,Naverage,HighRes) 
            self.measurement_configed=True
        else:
            print("no connection established yet")
        
    def measure(self,Nspec=1):
        """take data
        Nspec is the number of spectra to measure"""
        if self.connected:
            self.spectrometer.measure(Nspec=Nspec)
            self.spectrum=self.spectrometer.spec
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
        self.spectrometer.start_measure(Nspec)
    
    def stop_measure(self):
        """stop measurement"""
        self.spectrometer.stop_measure()
        
    def isdataready(self):
        return self.spectrometer.isdataready()
        
    def read_data(self):
        return self.spectrometer.read_data()
    
    def disconnect(self):
        """disconnect device"""
        self.spectrometer.disconnect()
    

class MultiSpectrometer():
    def __init__(self):
        self.S=Spectrometer()
        self.S.find()
        self.spectrometers=[]
    
    def connect(self,DeviceN):
        """DeviceN is the list of numbers to connect to"""
        for DN in DeviceN:
            S=copy.deepcopy(self.S)
            S.connect(DN)
            self.spectrometers.append(S)
            
    def config_measure(self,Tintegration,Naverage):
        """Naverage is the namber of spectra to average"""
        for i in range(len(self.spectrometers)):
            s=self.spectrometers[i]
            s.config_measure(Tintegration=Tintegration[i],Naverage=Naverage[i])
            
    def measure(self,Nspec=1):
        """Nspec is the number of spectra to take"""
        for s in self.spectrometers:
            s.measure(Nspec)
            
    def stop_measure(self):
        for s in self.spectrometers:
            s.stop_measure()
    
    def disconnect(self):
        """disconnect device"""
        for s in self.spectrometers:
            s.disconnect()

    def show_spec(self,title=None):
        for s in self.spectrometers:
            s.show_spec(title)

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
    
    
    
    
    