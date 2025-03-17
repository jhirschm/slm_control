"""A class representing an Ocean spectrometer"""

import numpy as np
import seabreeze.spectrometers as sb

from .base import Spectrometer


class OceanSpectrometer(Spectrometer):
    """A class representing an Ocean spectrometer"""

    def id(self):
        """Return manufacturer"""

        return "Ocean"

    def check_id(self):
        """Check if manufacturer matches this class"""

        return self.manufacturer == self.id()

    def initialize(self):
        """Initialize all devices"""
        for device in sb.list_devices():
            self.device_id_list.append(
                device.model + "-" + device.serial_number
            )
            self.device_list.append(sb.Spectrometer(device))

    def close(self):
        """Close the device"""
        for device in self.device_list:
            device.close()

    def collect(self):
        """Collect an intensity spectrum"""
        return self.device_list[self.device_index].intensities()

    def wavelengths(self):
        """Collect the wavelength axis"""
        return self.device_list[self.device_index].wavelengths()

    def sensitivity(self, calibrated_sensitivity: np.ndarray = None):
        """Collect the wavelength axis"""

        if calibrated_sensitivity is None:
            print("Ignoring sensitivity: calibrated curve must be provided.")

        raise NotImplementedError("Spectrometer must be defined")

    def exposure(self, exposure_time: float):
        """Set the exposure time for a single spectrum"""
        self.device_list[self.device_index].integration_time_micros(
            exposure_time * 1e6
        )
