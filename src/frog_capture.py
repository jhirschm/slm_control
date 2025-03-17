import ctypes
import thorlabs_apt as apt
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import clr
from clr_loader import get_coreclr
from pythonnet import load
import matplotlib.pyplot as plt

spectrometer_path = r"C:\slm_control\src\spectrometer"
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


def main():
    # Set up spectrometer
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
        
        spectrometer.ExposureTime = 0.5 #seconds
        
        
    # Set up motor    

    print(apt.list_available_devices())

    motor1_ID = apt.list_available_devices()[0][1]
    print("Using motor id: ")
    print(motor1_ID)

    # Initialize motor
    motor1 = apt.Motor(motor1_ID)

    # Adjust motor parameters if needed
    motor1.set_stage_axis_info(-4, 4, 1, 1.0)

    # Print current motor positions
    pos_initial = motor1.position
    print('Motor 1 position: ' + str(pos_initial) + ' mm')
    home_position = 0.165
    motor1.move_to(home_position, True)

    ########################
    ### PARAMETERS START ###
    ########################

    # Data folder name
    folder_name = 'Data_14'

    # Spectrum integration time (s)
    spectrum_integration_time = 0.5

    # Number of spectra to average
    spectrum_averages = 1

    # Number of scans to average
    scan_averages = 1

    # Center position of scan
    center = 0

    # Range of scan
    scan_start = -0.05
    scan_end = 0.05
    step_size = 0.001
    
    # Number of steps in scan
    scan_steps = int((scan_end - scan_start) / step_size) + 1

    # Generate scan positions including both start and end, ensuring symmetry
    scan_positions = np.linspace(scan_start, scan_end, scan_steps) + home_position

    # # Motor homing
    # home = False

    # Spectrum region for raw trace (nm)
    spectrum_wavelength_minimum = 490
    spectrum_wavelength_maximum = 560

    

    # Data directory
    data_directory = os.path.join(os.getcwd(), 'Data/' + folder_name + '/')

    ######################
    ### PARAMETERS END ###
    ######################

    #######################
    ### DATA COLLECTION ###
    #######################

    # Check if data directory exists
    if os.path.isdir(data_directory):
        input('Warning: Data directory already exists. Press enter on the command line to continue. Press Ctrl + C repeatedly to quit.')
    else:
        os.makedirs(data_directory)

    lam=list(spectrometer.GetWavelengths())

    #Scan
    trace = np.zeros((len(lam), len(scan_positions)))
    motor1_actual_positions = np.zeros((len(scan_positions),1))

    for k1, motor_position in enumerate(scan_positions):
        motor1.move_to(motor_position)
        position = motor1.position
        print('Step: ' + str(k1+1) + '/' + str(scan_steps)  + ', Position: ' + str(np.round(position,4)) + ' mm')

        motor1_actual_positions[k1] = position

        spectrometer.StartExposure()
        print("start exposure")
    
        while (spectrometer.Status == SDK.SpectrometerStatus.TakingSpectrum or 
            spectrometer.Status == SDK.SpectrometerStatus.WaitingForTrigger):
            time.sleep(0.2)
            print("waiting")

        I=np.asarray(list(spectrometer.GetSpectrum()))

        trace[:,k1]=I
    spectrometer.Close()
    
    ## PLOT DATA
    # Trace
    fig_result1, ((ax_trace)) = plt.subplots(1, 1, num=201, figsize=(6, 5))
    fig_result1.tight_layout(pad=5)
    ax_trace.pcolor(scan_positions,lam,trace,cmap='jet')
    ax_trace.set_xlabel('Position (mm)')
    ax_trace.set_ylabel('Wavelength (nm)')
    ax_trace.set_title('Trace')
    # fig_result1.savefig(os.path.join(data_directory, 'trace_raw.png'))

    print('Close figure to continue with basic analysis.')
    plt.show()

if __name__ == '__main__':
    print('here')
    main()