"""
Qmini class

requires pythonnet
if clr complains : uninstall clr and pythonnet; install pythonnet

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

import numpy as np
import time

import clr #pip install pythonnet
path=os.path.dirname((os.path.abspath(__file__)))

sys.path.append(path)

clr.AddReference(path+"\\RgbDriverKit.dll")

from RgbDriverKit import Qseries
import RgbDriverKit as SDK


class Qmini():
    
    def __init__(self):
        self.spec=[]
        self.find()
        self.Type='Qmini'
        self.sleeptime=0.001
        self.running=False
        self.measurement_configed = False
        
    def find(self, autoconnect=False):
        """finds all compartible devices"""
        devices = Qseries.SearchDevices()
        self.Ndev=devices.Length
        print("number of Qmini spectrometers: ", self.Ndev)
        self.devices=devices
        self.devconfig=[]
        for i in range(self.Ndev):
            d=devices[i]
            q = d
            SN=q.SerialNo
            model = q.ModelName
            self.devconfig.append([SN,d,model])
        if autoconnect:
            self.connect()
    
    def connect(self, DeviceN=0, SN=None):
        """connects to the device number DeviceN in the list of found devices"""
        connected = False
        if SN == None and DeviceN < len(self.devices):
            self.dev_handle = self.devices[DeviceN]
            self.SN = self.devconfig[DeviceN][0]
            self.model = self.devconfig[DeviceN][2]
            connected=True
        elif SN != None:
            SNs=[self.devconfig[i][0] for i in range(len(self.devices))]
            if SN in SNs:
                N=int(np.argwhere(np.array(SNs) == SN)[0][0])
                self.dev_handle = self.devices[N]
                self.SN = self.devconfig[N][0]
                self.model = self.devconfig[N][2]
                connected=True
            else:
                print('Qmini connections failed')
        else:
            print('Qmini connections failed')

        if connected:
            self.device = self.dev_handle
            self.device.Open()
            self.lam= np.array(list(self.device.GetWavelengths()))
            self.minT = self.device.MinExposureTime
            print('min exposure ',self.minT)
            
    def config_measure(self,Tintegration=0.01,Naverage=1,HighRes=False,IntensityCalibration=True,ExtTrig=False):
        """configurates spectrometer
        Tintegration is the integration time in ms 
        Naverage is the number of spectra to average
        HighRes is invalid for Ocean Optics; present for compartability"""
        print('Q calib', IntensityCalibration)
        self.device.UseSensitivityCalibration=IntensityCalibration
        self.device.ExposureTime = Tintegration/1000 #set exposure time in s
        self.Naverage=Naverage
        self.device.Averaging=Naverage
        
        self.measurement_configed=True

    def measure(self,Nspec=1):
        """take data
        Nspec is the number of spectra to measure
        param nummeas: number of measurements to do. -1 is infinite, -2 is used to
        start Dynamic StoreToRam"""
        
        if self.measurement_configed:
            self.spec=[]
            nummeas = Nspec
            scans = 0
            stopscanning = False
            while (stopscanning == False):
                self.device.StartExposure()
                datanotready = True
                while datanotready:
                    datanotready = (self.device.Status == SDK.SpectrometerStatus.TakingSpectrum or 
                          self.device.Status == SDK.SpectrometerStatus.WaitingForTrigger)
                    print('waitng')
                    time.sleep(self.sleeptime)
                scans = scans + 1
                if (scans >= nummeas):
                    stopscanning = True
                self.spec.append(np.array(self.read_data()[0])[:len(self.lam)])                   
           
        
        else:
            print("first call config_measure")

    def read_data(self):
        """read data from the spectrometer
        returns (spectrum, timestamp)"""
        timestamp = self.device.TimeStamp
        spectraldata = np.array(list(self.device.GetSpectrum()))
        return spectraldata, timestamp

    def start_measure(self,Nspec=1):
        """start measure but dont wait for ending it"""
        if self.measurement_configed:
            self.running=True
            self.device.StartExposure()
        else:
            print("first call config_measure")

    def stop_measure(self):
        """stop measurement"""
        self.running=False
        
    def isdataready(self):
        """check if the measured data is ready"""
        return not (self.device.Status == SDK.SpectrometerStatus.TakingSpectrum or 
              self.device.Status == SDK.SpectrometerStatus.WaitingForTrigger)

    def disconnect(self):
        """disconnect device"""
        self.device.Close()


# S=Qmini()


