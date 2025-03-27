# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 16:08:40 2025

@author: lasopr
"""

import os
import sys
import sys
import inspect
import ctypes
import struct
import os
import numpy as np
import time

import sys
import clr
from clr_loader import get_coreclr
from pythonnet import load
import matplotlib.pyplot as plt

# # Initialize .NET runtime
# runtime = get_coreclr()
# load(runtime)

# # Now import clr
# import clr

# Add the spectrometer path to sys.path
spectrometer_path = r"C:\my_python\Hardware\spectrometer"
if spectrometer_path not in sys.path:
    sys.path.append(spectrometer_path)

# Verify the path is added
print("Current sys.path:", sys.path)

# Load the .NET DLL
dll_path = os.path.join(spectrometer_path, "broadcom", "RgbDriverKit.dll")
if os.path.exists(dll_path):
    clr.AddReference(dll_path)
    print("DLL loaded successfully.")
else:
    print("DLL path does not exist!")

from RgbDriverKit import Qseries
import RgbDriverKit as SDK
devices=Qseries.SearchDevices()
print('number of spectrometers: ', devices.Length)
print(devices)

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

    spectrometer.UseSensitivityCalibration=False
    
    spectrometer.Averaging=1
    
    spectrometer.ExposureTime = 0.5
    spectrometer.StartExposure()
    print("start exposure")
    
    while (spectrometer.Status == SDK.SpectrometerStatus.TakingSpectrum or 
          spectrometer.Status == SDK.SpectrometerStatus.WaitingForTrigger):
        time.sleep(0.2)
        print("waiting")

    I=list(spectrometer.GetSpectrum())
    spectrometer.Close()
    print(I[:10])
    print(len(I))
    plt.plot(lam,I[0:])
    plt.show()

    
