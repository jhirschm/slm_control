"""A class representing an Ocean spectrometer"""

import time
import numpy as np

from rgbdriverkit.qseriesdriver import Qseries

from .base import Spectrometer


class BroadcomSpectrometer(Spectrometer):
    """A class representing a Broadcom spectrometer"""

    def id(self):
        """Return manufacturer"""

        return "Broadcom"

    def initialize(self):
        """Initialize all devices"""
        device_address_list = Qseries.search_devices()
        if device_address_list is not None:
            for device_address in device_address_list:
                q = Qseries(device_address)
                self.device_id_list.append(
                    q.model_name + "-" + q.serial_number
                )
                self.device_list.append(q)
                self.device_list[-1].open()
            del q, device_address_list

    def close(self):
        """Close the devices"""
        for device in self.device_list:
            device.parameter_reset()
            device.device_reset()

    def collect(self):
        """Collect an intensity spectrum"""
        self.device_list[self.device_index].start_exposure(1)
        time.sleep(self.device_list[self.device_index].exposure_time + 0.001)
        while not self.device_list[self.device_index].available_spectra:
            time.sleep(0.001)
        spec = self.device_list[self.device_index].get_spectrum_data()
        return spec.Spectrum

    def wavelengths(self):
        """Collect the wavelength axis"""
        wavelengths = self.device_list[self.device_index].get_wavelengths()
        return wavelengths

    def sensitivity(self, calibrated_sensitivity: np.ndarray = None):
        """Collect the wavelength axis"""

        if calibrated_sensitivity is None:
            print("Ignoring sensitivity: calibrated curve must be provided.")

        raise NotImplementedError("Spectrometer must be defined")

    def exposure(self, exposure_time: float):
        """Set the exposure time for a single spectrum"""
        self.device_list[self.device_index].exposure_time = exposure_time
