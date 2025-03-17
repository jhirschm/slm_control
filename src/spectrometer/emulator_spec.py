"""
spectrometer emulator

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

import time
import numpy as np
import time

class SpecEmulator():
    def __init__(self,SN=1,ID=1,lamRange=[600,1100],Nlam=2**12):
        self.spec=[]
        self.SN=SN
        self.ID=ID
        self.connected=False
        self.lam=np.linspace(lamRange[0],lamRange[1],Nlam)
        self.Type='Emulator'
        
    def connect(self):
        self.connected=True
    
    def config_measure(self,Tintegration=0.01,Naverage=1,HighRes=False,IntensityCalibration=False,ExtTrig=False):
        self.Tintegration=Tintegration
        self.Naverage=Naverage
        
    def measure(self,Nspec=1):
        print(self.Tintegration)
        time.sleep(self.Tintegration/1000) #convert sleep to s
        self.spec=[]
        for n in range(Nspec):
            self.spec.append(self.spectrum(self.lam))
            
    def spectrum(self,lam,Lam0=800,Dlam=40,Noise=0.05):
        """generated spectrum"""
        lam0=Lam0*(1+0.05*(np.random.rand()-0.5)*2)
        # print(self.lam.shape)
        noise=np.random.rand(self.lam.shape[0])*Noise
        X=self.lam
        return np.exp(-4*np.log(2)*(X-lam0)**2/Dlam**2)+noise
    
    def read_data(self):
        timestamp=time.time()
        return self.spectrum(self.lam), timestamp
    
    def start_measure(self,Nspec=1):
        self.timer=time.time()
        
    def isdataready(self):
        return time.time()-self.timer > self.Tintegration/1000 #convert ms to s
    
    def stop_measure(self):
        pass
    
    def disconnect(self):
        self.connected=False