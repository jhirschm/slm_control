import ctypes
import thorlabs_apt as apt
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import clr
from clr_loader import get_coreclr
from pythonnet import load
from scipy.constants import c  # Speed of light in m/s
print("here")
# Spectrometer setup
spectrometer_path = r"C:\slm_control\src\spectrometer"
if spectrometer_path not in sys.path:
    sys.path.append(spectrometer_path)

# Load the .NET DLL
dll_path = os.path.join(spectrometer_path, "broadcom", "RgbDriverKit.dll")
if os.path.exists(dll_path):
    clr.AddReference(dll_path)
else:
    raise FileNotFoundError("DLL path does not exist!")

from RgbDriverKit import Qseries
import RgbDriverKit as SDK

class FROG:
    def __init__(self, integration_time=0.5, averaging=1, central_motor_position=0.0,
                 scan_range=(-0.05, 0.05), step_size=0.001):
        """
        Initialize the FROG system with motor and spectrometer.

        Parameters:
        - integration_time: float, exposure time for the spectrometer (seconds)
        - averaging: int, number of spectra to average per position
        - central_motor_position: float, center position of motor scan (mm)
        - scan_range: tuple, (start, end) range of motor scan relative to center (mm)
        - step_size: float, step size for motor (mm)
        """

        # Motor Setup
        motor_devices = apt.list_available_devices()
        if not motor_devices:
            raise RuntimeError("No Thorlabs motor detected.")
        self.motor_id = motor_devices[0][1]
        self.motor = apt.Motor(self.motor_id)
        self.motor.set_stage_axis_info(-4, 4, 1, 1.0)  # Define motor limits

        # Spectrometer Setup
        devices = Qseries.SearchDevices()
        if devices.Length == 0:
            raise RuntimeError("No spectrometer detected.")
        self.spectrometer = devices[0]
        self.spectrometer.Open()
        self.wavelengths = np.array(list(self.spectrometer.GetWavelengths()))

        # Spectrometer parameters
        self.integration_time = integration_time
        self.averaging = averaging
        self.spectrometer.ExposureTime = integration_time
        self.spectrometer.Averaging = averaging

        # Motor scan parameters
        self.central_motor_position = central_motor_position
        self.scan_start, self.scan_end = scan_range
        self.step_size = step_size

        # Generate scan positions
        self.scan_positions = np.linspace(self.scan_start, self.scan_end,
                                          int((self.scan_end - self.scan_start) / step_size) + 1) + central_motor_position

        self.delay_vector = None
        self.wavelength_vector = None
    def close_frog(self):
        self.spectrometer.Close()
    def run(self, close=True):
        """
        Perform the FROG scan and return collected spectra.
        
        Returns:
        - trace: numpy array of shape (wavelengths, motor positions, averaging)
        - real_positions: numpy array of actual motor positions (motor_steps,)
        """

        num_steps = len(self.scan_positions)
        num_wavelengths = len(self.wavelengths)
        trace = np.zeros((num_wavelengths, num_steps, self.averaging))
        real_positions = np.zeros(num_steps)

        for k, pos in enumerate(self.scan_positions):
            self.motor.move_to(pos, True)
            real_positions[k] = self.motor.position
            print(f"Step {k+1}/{num_steps}, Position: {real_positions[k]:.4f} mm")

            for avg in range(self.averaging):
                self.spectrometer.StartExposure()
                while self.spectrometer.Status in [SDK.SpectrometerStatus.TakingSpectrum, 
                                                   SDK.SpectrometerStatus.WaitingForTrigger]:
                    time.sleep(0.2)

                trace[:, k, avg] = np.array(list(self.spectrometer.GetSpectrum()))

        if close:
           self.close_frog()
        
        return trace, real_positions

    def get_info(self):
        """
        Retrieve useful system parameters.
        """
        info = {
            "Integration Time (s)": self.integration_time,
            "Averaging": self.averaging,
            "Motor Center (mm)": self.central_motor_position,
            "Scan Range (mm)": (self.scan_start, self.scan_end),
            "Step Size (mm)": self.step_size,
            "Number of Steps": len(self.scan_positions),
            "Number of Wavelengths": len(self.wavelengths)
        }
        return info
    
    

    def plot(self, trace, real_positions, wavelength_range=(490, 560), time_axis=False):
        """
        Plot the collected FROG trace.

        Parameters:
        - trace: The 3D array (wavelengths x motor steps x averaging)
        - real_positions: The actual scanned positions
        - wavelength_range: Tuple (min, max) to limit displayed wavelengths
        - time_axis: Boolean, if True, converts spatial steps to time using the speed of light
        """

        # Compute mean over averages
        trace_mean = np.mean(trace, axis=2)

        # Limit wavelength range
        mask = (self.wavelengths >= wavelength_range[0]) & (self.wavelengths <= wavelength_range[1])
        wavelengths = self.wavelengths[mask]
        trace_mean = trace_mean[mask, :]

        # Convert motor steps to time if required
        if time_axis:
            time_steps = (real_positions * 1e-3) / c * 1e15  # Convert mm to fs
            time_steps -= np.mean(time_steps)  # Center around zero
            x_label = "Time Delay (fs)"
            x_values = time_steps
        else:
            x_label = "Position (mm)"
            x_values = real_positions

        # Plot the trace
        plt.figure(figsize=(8, 6))
        plt.pcolor(x_values, wavelengths, trace_mean, cmap="jet", shading='auto')
        plt.colorbar(label="Intensity")
        plt.xlabel(x_label)
        plt.ylabel("Wavelength (nm)")
        plt.title("FROG Trace (Temporal)" if time_axis else "FROG Trace (Spatial)")
        plt.show()

    def mask_trace(self, trace, wavelength_range=(490, 560)):
        mask = (self.wavelengths >= wavelength_range[0]) & (self.wavelengths <= wavelength_range[1])
        wavelengths = self.wavelengths[mask]
        trace = trace[mask,:]

        return wavelengths, trace
    
    def set_wavelength_vector(self, wavelength_vector):
        self.wavelength_vector = wavelength_vector
    
    def set_delay_vector(self, delay_vector):
        self.delay_vector =  delay_vector


# frog = FROG(integration_time=0.1, averaging=1, central_motor_position=0.165, scan_range=(-0.05, 0.05), step_size=0.001)
# frog = FROG(integration_time=0.1, averaging=1, central_motor_position=-.27, scan_range=(-0.05, 0.05), step_size=0.001)


# # # Collect Data
# trace, real_positions = frog.run()

# # Get System Info
# info = frog.get_info()
# print(info)

# # Plot in Spatial Domain
# frog.plot(trace, real_positions, wavelength_range=(490, 560), time_axis=False)

# Plot in Temporal Domain
# frog.plot(trace, real_positions, wavelength_range=(490, 560), time_axis=True)