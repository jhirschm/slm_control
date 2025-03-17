# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:59:09 2024

@author: slava
"""

import sys
import inspect
import ctypes
import struct
import os
import numpy as np
import time

import sys
import clr
path=os.path.dirname((os.path.abspath(__file__)))

sys.path.append(path)

clr.AddReference(path+"\\RgbDriverKit.dll")

from RgbDriverKit import Qseries
import RgbDriverKit as SDK
devices=Qseries.SearchDevices()
print('number of spectrometers: ', devices.Length)

if devices.Length>0:
    spectrometer=devices[0]
    spectrometer.Open()
    lam=list(spectrometer.GetWavelengths())
    print(lam[:10])
    print("name: ", spectrometer.DetailedDeviceName)
    print("name: ", spectrometer.ModelName)
    print("SN: ", spectrometer.SerialNo)
    print("px: ", spectrometer.PixelCount)
    print('min exp: ', spectrometer.MinExposureTime)
    print('trigger ', spectrometer.CanUseExternalTrigger)
    print(spectrometer.MaxAveraging)
    print('calibrated ', spectrometer is SDK.CalibratedSpectrometer) #doesnt work. probsbly needs ctypes
    print('calibration ', list(spectrometer.SensitivityCalibration)[:10])
    
    spectrometer.UseSensitivityCalibration=True
    
    spectrometer.Averaging=1
    
    spectrometer.ExposureTime = 0.5
    spectrometer.StartExposure()
    print("start exposure")
    
    while (spectrometer.Status == SDK.SpectrometerStatus.TakingSpectrum or 
          spectrometer.Status == SDK.SpectrometerStatus.WaitingForTrigger):
        time.sleep(0.2)
        print("waiting")

    I=list(spectrometer.GetSpectrum())
    print(I[:10])
    
    spectrometer.Close()










