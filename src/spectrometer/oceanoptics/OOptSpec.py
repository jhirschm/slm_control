"""
Ocean optics spectrometer class

@author: slava
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

import seabreeze.spectrometers as sb
import numpy as np
import time


class OOptSpec():
    
    def __init__(self):
        self.spec=[]
        self.find()
        self.Type='Oceanoptics'
        self.sleeptime=0.001
        self.running=False
        self.measurement_configed = False

    def find(self, autoconnect=False):
        """finds all compartible devices"""
        devices = sb.list_devices()
        self.Ndev=len(devices)
        print("number of Ocean Optics spectrometers: ", self.Ndev)
        self.devices=devices
        self.devconfig=[]
        for i in range(self.Ndev):
            d=devices[i]
            SN=d.serial_number
            model = d.model
            self.devconfig.append([SN,d,model])
        # print(self.devconfig)
        if autoconnect:
            self.connect()

    def connect(self, DeviceN=0, SN=None):
        """connects to the device number DeviceN in the list of found devices"""
        # mylist = AVS_GetList(1)
        connected = False
        if SN == None and DeviceN < len(self.devices):
            # mylist=self.devices
            # print(mylist[0])
            self.dev_handle = self.devices[DeviceN]
            self.SN = self.devconfig[DeviceN][0]
            self.model = self.devconfig[DeviceN][2]
            connected=True
        elif SN != None:
            SNs=[self.devconfig[i][0] for i in range(len(self.devices))]
            if SN in SNs:
                N=np.argwhere(np.array(SNs) == SN)[0][0]
                self.dev_handle = self.devices[N]
                self.SN = self.devconfig[N][0]
                self.model = self.devconfig[N][2]
                connected=True
            else:
                print('Ocean optics connections failed')
        else:
            print('Ocean optics connections failed')
        # devcon = AVS.AVS.eviceConfigType()
        if connected:
            self.device = sb.Spectrometer(self.dev_handle)
            self.lam= self.device.wavelengths()
            
    def config_measure(self,Tintegration=0.01,Naverage=1,HighRes=False,IntensityCalibration=False,ExtTrig=False):
        """configurates spectrometer
        Tintegration is the integration time in ms 
        Naverage is the number of spectra to average
        HighRes is invalid for Ocean Optics; present for compartability"""
        self.device.integration_time_micros(int(Tintegration*1000)) # the final funciton is in us (must be int)
        self.Naverage=Naverage
        self.averages=np.zeros((Naverage,len(self.lam)))
        
        self.measurement_configed=True
        
    def measure(self,Nspec=1):
        """take data
        Nspec is the number of spectra to measure"""
        if self.measurement_configed:
            self.spec=[]
            nummeas = Nspec
            scans = 0
            stopscanning = False
            while (stopscanning == False):  
                    scans = scans + 1
                    if (scans >= nummeas):
                        stopscanning = True
                    if self.Naverage > 1:
                        for i in range(self.Naverage):
                            self.averages[i]=self.device.intensities()
                        Spec=np.sum(self.averages,axis=0)/self.Naverage
                    else:
                        Spec=self.device.intensities()
                    self.spec.append(np.array(Spec))                    
        
        else:
            print("first call config_measure")
            
    def read_data(self):
        """read data from the spectrometer
        returns (spectrum, timestamp)"""
        timestamp = time.time()
        return self.spec[0], timestamp
            
    def isdataready(self):
        """check if the measured data is ready"""
        return True
            
    def start_measure(self,Nspec=1):
        """start measure but dont wait for ending it"""
        if self.measurement_configed:
            self.running=True
            self.measure()
        else:
            print("first call config_measure")
            
    def stop_measure(self):
        """stop measurement"""
        self.running=False
            
    def disconnect(self):
        """disconnect device"""
        self.device.close()
    










