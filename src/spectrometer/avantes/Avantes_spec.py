"""
Created on Tue Oct 11 20:51:44 2022

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


#from avaspec import *
import avaspec as AVS
import time
import numpy as np


class AvaSpec():
    def __init__(self):
        self.spec=[]
        self.find()
        self.Type='Avantes'
        self.sleeptime=0.001
        self.running=False
        self.measurement_configed=False
        
    def find(self, autoconnect=False):
        """finds all compartible devices"""
        AVS.AVS_Init(0)
        ret = AVS.AVS_GetNrOfDevices()
        self.Ndev=ret
        print("number of Avantes spectrometers: ", self.Ndev)
        # self.devices=AvsIdentityType()
        AVS.AvsIdentityType()
        self.devices=AVS.AVS_GetList(1)
        self.devconfig=[]
        for i in range(self.Ndev):
            d=self.devices[i]
            dev_handle = AVS.AVS_Activate(d)
            devcon = AVS.DeviceConfigType()
            devcon = AVS.AVS_GetParameter(dev_handle, 63484)
            SN=devcon.m_aUserFriendlyId
            lam = np.array(AVS.AVS_GetLambda(dev_handle))
            ind=lam>0.1
            lam=lam[ind]
            self.devconfig.append([SN,d,[lam.min(),lam.max()]])
        # print(self.devconfig)
        if autoconnect:
            self.connect()
        
    def connect(self, DeviceN=0, deviceID=None):
        """connects to the device number DeviceN in the list of found devices"""
        # mylist = AVS_GetList(1)
        if deviceID==None:
            mylist=self.devices
            # print(mylist[0])
            self.dev_handle = AVS.AVS_Activate(mylist[DeviceN])
        else:
            self.dev_handle = AVS.AVS_Activate(deviceID)
        # devcon = AVS.AVS.eviceConfigType()
        devcon = AVS.AVS_GetParameter(self.dev_handle, 63484)
        self.devcon=devcon
        self.SN=devcon.m_aUserFriendlyId.decode("utf-8")
        print(devcon.m_aUserFriendlyId)
        print('trigger mode', devcon.m_StandAlone_m_Meas_m_Trigger_m_Mode)
        self.pixels = devcon.m_Detector_m_NrPixels
        lam = np.array(AVS.AVS_GetLambda(self.dev_handle))
        ind=lam>0.1
        self.lam=lam[ind]
        
        # print('enable TEC ',devcon.m_TecControl_m_Enable)
        # print('set ',devcon.m_TecControl_m_Setpoint)
        # print('tem',devcon.m_Temperature_1_m_aFit)
        # print(devcon.m_TecControl_m_aFit)
        # print(devcon)
        
        # T1=AVS.AVS_GetAnalogIn(self.dev_handle,0,0)
        # print('temperature? ',T1)
        
    def config_measure(self,Tintegration=0.01,Naverage=1,HighRes=True,IntensityCalibration=False,ExtTrig=False):
        """configurates spectrometer
        Tintegration is the integration time in ms
        Naverage is the number of spectra to average
        HighRes True enables 16 bit resolution (65535 max value), 
        false uses 14 bit resolution (16383 max value)"""
        ret = AVS.AVS_UseHighResAdc(self.dev_handle, HighRes)
        measconfig = AVS.MeasConfigType()
        measconfig.m_StartPixel = 0
        measconfig.m_StopPixel = self.pixels - 1
        measconfig.m_IntegrationTime = float(Tintegration)
        measconfig.m_IntegrationDelay = 0
        measconfig.m_NrAverages = int(Naverage)
        measconfig.m_CorDynDark_m_Enable = 0  # nesting of types does NOT work!!
        measconfig.m_CorDynDark_m_ForgetPercentage = 0
        measconfig.m_Smoothing_m_SmoothPix = 0
        measconfig.m_Smoothing_m_SmoothModel = 0
        measconfig.m_SaturationDetection = 0
        if ExtTrig:
            measconfig.m_Trigger_m_Mode = 1
        else:
            measconfig.m_Trigger_m_Mode = 0
        measconfig.m_Trigger_m_Source = 0
        measconfig.m_Trigger_m_SourceType = 0
        measconfig.m_Control_m_StrobeControl = 0
        measconfig.m_Control_m_LaserDelay = 0
        measconfig.m_Control_m_LaserWidth = 0
        measconfig.m_Control_m_LaserWaveLength = 785.0
        measconfig.m_Control_m_StoreToRam = 0
        self.ret = AVS.AVS_PrepareMeasure(self.dev_handle, measconfig)
        self.measurement_configed=True
        
        """Trigger parameters
mode, (0 = Software, 1 = Hardware, 2 = Single Scan)
trigger source, (0 = external trigger, 1 = sync input)
source type, (0 = edge trigger, 1 = level trigger)"""
        
        # self.set_T()
        
    # def set_T(self,EnableCooling=True,T=0):
    #     """set temperature"""
    #     devcon=self.set_standard_devconfig()
    #     if EnableCooling:
    #         devcon.m_TecControl_m_Enable=True
    #     else:
    #         devcon.m_TecControl_m_Enable=False
        
    #     devcon.m_TecControl_m_Setpoint=float(T)
    #     AVS.AVS_SetParameter(self.dev_handle,devcon)
        
    # def set_standard_devconfig(self):
    #     devcon0 = AVS.AVS_GetParameter(self.dev_handle, 63484)
    #     devcon=AVS.DeviceConfigType()
        
    #     devcon.m_Len=devcon0.m_Len
    #     devcon.m_ConfigVersion=devcon0.m_ConfigVersion
    #     devcon.m_aUserFriendlyId=devcon0.m_aUserFriendlyId
    #     devcon.m_Detector_m_SensorType=devcon0.m_Detector_m_SensorType
    #     devcon.m_Detector_m_NrPixels=devcon0.m_Detector_m_NrPixels
    #     devcon.m_Detector_m_aFit=devcon0.m_Detector_m_aFit
    #     devcon.m_Detector_m_NLEnable=devcon0.m_Detector_m_NLEnable
    #     devcon.m_Detector_m_aNLCorrect=devcon0.m_Detector_m_aNLCorrect
    #     devcon.m_Detector_m_aLowNLCounts=devcon0.m_Detector_m_aLowNLCounts
    #     devcon.m_Detector_m_aHighNLCounts=devcon0.m_Detector_m_aHighNLCounts
    #     devcon.m_Detector_m_Gain=devcon0.m_Detector_m_Gain
    #     devcon.m_Detector_m_Reserved=devcon0.m_Detector_m_Reserved
    #     devcon.m_Detector_m_Offset=devcon0.m_Detector_m_Offset
    #     devcon.m_Detector_m_ExtOffset=devcon0.m_Detector_m_ExtOffset
    #     devcon.m_Detector_m_DefectivePixels=devcon0.m_Detector_m_DefectivePixels
    #     devcon.m_Irradiance_m_IntensityCalib_m_Smoothing_m_SmoothPix=devcon0.m_Irradiance_m_IntensityCalib_m_Smoothing_m_SmoothPix
    #     devcon.m_Irradiance_m_IntensityCalib_m_Smoothing_m_SmoothModel=devcon0.m_Irradiance_m_IntensityCalib_m_Smoothing_m_SmoothModel
    #     devcon.m_Irradiance_m_IntensityCalib_m_CalInttime=devcon0.m_Irradiance_m_IntensityCalib_m_CalInttime
    #     devcon.m_Irradiance_m_IntensityCalib_m_aCalibConvers=devcon0.m_Irradiance_m_IntensityCalib_m_aCalibConvers
    #     devcon.m_Irradiance_m_CalibrationType=devcon0.m_Irradiance_m_CalibrationType
    #     devcon.m_Irradiance_m_FiberDiameter=devcon0.m_Irradiance_m_FiberDiameter
    #     devcon.m_Reflectance_m_Smoothing_m_SmoothPix=devcon0.m_Reflectance_m_Smoothing_m_SmoothPix
    #     devcon.m_Reflectance_m_Smoothing_m_SmoothModel=devcon0.m_Reflectance_m_Smoothing_m_SmoothModel
    #     devcon.m_Reflectance_m_CalInttime=devcon0.m_Reflectance_m_CalInttime
    #     devcon.m_Reflectance_m_aCalibConvers=devcon0.m_Reflectance_m_aCalibConvers
    #     devcon.m_SpectrumCorrect=devcon0.m_SpectrumCorrect
    #     devcon.m_StandAlone_m_Enable=devcon0.m_StandAlone_m_Enable
    #     devcon.m_StandAlone_m_Meas_m_StartPixel=devcon0.m_StandAlone_m_Meas_m_StartPixel
    #     devcon.m_StandAlone_m_Meas_m_StopPixel=devcon0.m_StandAlone_m_Meas_m_StopPixel
    #     devcon.m_StandAlone_m_Meas_m_IntegrationTime=devcon0.m_StandAlone_m_Meas_m_IntegrationTime
    #     devcon.m_StandAlone_m_Meas_m_IntegrationDelay=devcon0.m_StandAlone_m_Meas_m_IntegrationDelay
    #     devcon.m_StandAlone_m_Meas_m_NrAverages=devcon0.m_StandAlone_m_Meas_m_NrAverages
    #     devcon.m_StandAlone_m_Meas_m_CorDynDark_m_Enable=devcon0.m_StandAlone_m_Meas_m_CorDynDark_m_Enable 
    #     devcon.m_StandAlone_m_Meas_m_CorDynDark_m_ForgetPercentage=devcon0.m_StandAlone_m_Meas_m_CorDynDark_m_ForgetPercentage
    #     devcon.m_StandAlone_m_Meas_m_Smoothing_m_SmoothPix=devcon0.m_StandAlone_m_Meas_m_Smoothing_m_SmoothPix
    #     devcon.m_StandAlone_m_Meas_m_Smoothing_m_SmoothModel=devcon0.m_StandAlone_m_Meas_m_Smoothing_m_SmoothModel
    #     devcon.m_StandAlone_m_Meas_m_SaturationDetection=devcon0.m_StandAlone_m_Meas_m_SaturationDetection
    #     devcon.m_StandAlone_m_Meas_m_Trigger_m_Mode=devcon0.m_StandAlone_m_Meas_m_Trigger_m_Mode
    #     devcon.m_StandAlone_m_Meas_m_Trigger_m_Source=devcon0.m_StandAlone_m_Meas_m_Trigger_m_Source
    #     devcon.m_StandAlone_m_Meas_m_Trigger_m_SourceType=devcon0.m_StandAlone_m_Meas_m_Trigger_m_SourceType
    #     devcon.m_StandAlone_m_Meas_m_Control_m_StrobeControl=devcon0.m_StandAlone_m_Meas_m_Control_m_StrobeControl
    #     devcon.m_StandAlone_m_Meas_m_Control_m_LaserDelay=devcon0.m_StandAlone_m_Meas_m_Control_m_LaserDelay
    #     devcon.m_StandAlone_m_Meas_m_Control_m_LaserWidth=devcon0.m_StandAlone_m_Meas_m_Control_m_LaserWidth
    #     devcon.m_StandAlone_m_Meas_m_Control_m_LaserWaveLength=devcon0.m_StandAlone_m_Meas_m_Control_m_LaserWaveLength
    #     devcon.m_StandAlone_m_Meas_m_Control_m_StoreToRam=devcon0.m_StandAlone_m_Meas_m_Control_m_StoreToRam
    #     devcon.m_StandAlone_m_Nmsr=devcon0.m_StandAlone_m_Nmsr
    #     devcon.m_StandAlone_m_Reserved=devcon0.m_StandAlone_m_Reserved
    #     devcon.m_Temperature_1_m_aFit =devcon0.m_Temperature_1_m_aFit
    #     devcon.m_Temperature_2_m_aFit =devcon0.m_Temperature_2_m_aFit
    #     devcon.m_Temperature_3_m_aFit =devcon0.m_Temperature_3_m_aFit
    #     devcon.m_TecControl_m_Enable = devcon0.m_TecControl_m_Enable
    #     devcon.m_TecControl_m_Setpoint =devcon0.m_TecControl_m_Setpoint
    #     devcon.m_TecControl_m_aFit =devcon0.m_TecControl_m_aFit
    #     devcon.m_ProcessControl_m_AnalogLow =devcon0.m_ProcessControl_m_AnalogLow
    #     devcon.m_ProcessControl_m_AnalogHigh =devcon0.m_ProcessControl_m_AnalogHigh
    #     devcon.m_ProcessControl_m_DigitalLow =devcon0.m_ProcessControl_m_DigitalLow
    #     devcon.m_ProcessControl_m_DigitalHigh =devcon0.m_ProcessControl_m_DigitalHigh
    #     devcon.m_EthernetSettings_m_IpAddr = devcon0.m_EthernetSettings_m_IpAddr
    #     devcon.m_EthernetSettings_m_NetMask =devcon0.m_EthernetSettings_m_NetMask
    #     devcon.m_EthernetSettings_m_Gateway =devcon0.m_EthernetSettings_m_Gateway
    #     devcon.m_EthernetSettings_m_DhcpEnabled =devcon0.m_EthernetSettings_m_DhcpEnabled
    #     devcon.m_EthernetSettings_m_TcpPort =devcon0.m_EthernetSettings_m_TcpPort
    #     devcon.m_EthernetSettings_m_LinkStatus =devcon0.m_EthernetSettings_m_LinkStatus
    #     devcon.m_Reserved =devcon0.m_Reserved
    #     devcon.m_OemData = devcon0.m_OemData
        
    #     return devcon
        
    def measure(self,Nspec=1):
        """take data
        Nspec is the number of spectra to measure
        param nummeas: number of measurements to do. -1 is infinite, -2 is used to
        start Dynamic StoreToRam"""
        
        if self.measurement_configed:
            if (self.ret ==  0):
                self.spec=[]
                nummeas = Nspec
                scans = 0
                stopscanning = False
                while (stopscanning == False):
                    ret = AVS.AVS_Measure(self.dev_handle, 0, nummeas)
                    dataready = False
                    while (dataready == False):
                        dataready = (AVS.AVS_PollScan(self.dev_handle) == True)
                        time.sleep(self.sleeptime)
                    if dataready == True:
                        scans = scans + 1
                        if (scans >= nummeas):
                            stopscanning = True
                        self.spec.append(np.array(self.read_data()[0])[:len(self.lam)])             
                    # time.sleep(0.001)        
            else:
                print("Error in the measurement ",self.ret)
                #add raise
        
        else:
            print("first call config_measure")
            
    def read_data(self):
        """read data from the spectrometer
        returns (spectrum, timestamp)"""
        ret = AVS.AVS_GetScopeData(self.dev_handle)
        timestamp = ret[0]
        spectraldata = ret[1]
        return spectraldata, timestamp
    
    def start_measure(self,Nspec=1):
        """start measure but dont wait for ending it"""
        if self.measurement_configed:
            # if not self.running:
            ret = AVS.AVS_Measure(self.dev_handle, 0, Nspec)
            self.running=True
        else:
            print("first call config_measure")
            
    def isdataready(self):
        """check if the measured data is ready"""
        return (AVS.AVS_PollScan(self.dev_handle) == True)
    
    def stop_measure(self):
        """stop measurement"""
        ret = AVS.AVS_StopMeasure(self.dev_handle)
        self.running=False
    
    def disconnect(self):
        """disconnect device
        (actually, probably, disconnects all avantes spectrometers)"""
        AVS.AVS_Done()
    
    
    
    
    
    
    
# #test
# S=AvaSpec()
# S.connect()
# # #deviceID=S.devices[1]
# S.config_measure(Tintegration=1)
# S.measure()
# S.disconnect()
    
# S2=AvaSpec()
# S2.connect(1)
# #deviceID=S.devices[1]
# S2.config_measure()
# S2.measure()
    
    
    
    
    
    
    
    
        