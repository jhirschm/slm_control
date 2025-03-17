"""Base class and settings for spectrometer"""

import numpy as np


class Spectrometer:
    """The base class for a spectrometer device

    Keyword arguments:
    - device_id -- Device number if there are multiple
    - manufacturer -- Broadcom or Ocean"""

    def __init__(self, manufacturer: str = None):
        self.manufacturer = manufacturer
        self.device_id_list = []
        self.device_list = []
        self.device_index = 0

    def id(self):
        """Return manufacturer"""
        return ""

    def check_id(self):
        """Check if manufacturer matches this class"""
        return self.manufacturer == self.id()

    def device_id(self):
        """Return id for specified device"""
        return self.device_id_list[self.device_index]

    def initialize(self):
        """Initialize all devices"""
        raise NotImplementedError("Device must be defined")

    def close(self):
        """Close the devices"""
        raise NotImplementedError("Device must be defined")

    def collect(self):
        """Collect an intensity spectrum"""
        raise NotImplementedError("Device must be defined")

    def wavelengths(self):
        """Collect the wavelength axis"""
        raise NotImplementedError("Device must be defined")

    def sensitivity(self, calibrated_sensitivity: np.ndarray = None):
        """Collect the wavelength axis"""
        raise NotImplementedError("Device must be defined")

    def exposure(self, exposure_time: float):
        """Set the exposure time for a single spectrum"""
        raise NotImplementedError("Device must be defined")
