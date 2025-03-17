"""A class representing a spectrum"""

import os
import dataclasses
import numpy as np
from time import localtime, strftime
from matplotlib import pyplot as plt

from spectrometers.devices.base import Spectrometer


@dataclasses.dataclass
class SpectrumSettings:
    """Acquisition settings for spectrum

    Keyword arguments:
    - exposure -- time to integrate (seconds)
    - averages -- number of times to average spectrum
    - wavelengths -- wavelength axis for spectrum
    - background -- background for spectrum
    - calibration -- calibration for spectrum"""

    exposure: float = 0.1
    averages: int = 1
    wavelengths = None
    background = None
    sensitivity = None


class Spectrum:
    """A class representing a spectrum

    Keyword arguments:
    - device -- Spectrometer device object
    - settings -- Spectrum settings object"""

    def __init__(
        self,
        device: Spectrometer,
        settings: SpectrumSettings = SpectrumSettings(),
    ):
        self.device = device
        self.settings = settings
        self.intensities = None
        self.device.exposure(self.settings.exposure)
        self.settings.wavelengths = self.device.wavelengths()
        self.settings.background = np.zeros_like(self.settings.wavelengths)
        self.settings.sensitivity = np.ones_like(self.settings.wavelengths)

    def exposure(self, exposure: float = 0.1):
        """Change the exposure time

        Keyword arguments:
        - exposure -- Exposure time in seconds"""

        self.settings.exposure = exposure
        self.device.exposure(self.settings.exposure)

    def collect(self, sensitivity: bool = False):
        """Return a spectrum

        Keyword arguments:
        - sensitivity -- Whether to apply sensitivity"""

        self.intensities = np.zeros_like(self.settings.wavelengths)
        for _ in range(self.settings.averages):
            self.intensities += (
                self.device.collect() - self.settings.background
            ) / self.settings.averages

        if sensitivity:
            self.intensities /= self.settings.sensitivity
        return self.intensities

    def background(self):
        """Collect a background"""

        self.settings.background = np.zeros_like(self.settings.wavelengths)
        for _ in range(self.settings.averages):
            self.settings.background += (
                self.device.collect() / self.settings.averages
            )

    def sensitivity(self, reference=None):
        """Store sensitivity calibration"""

        self.settings.sensitivity = self.device.sensitivity(reference)

    def plot(
        self,
        wavelength_limits: list[float] = None,
        intensity_limits: list[float] = None,
    ):
        """Plot the spectrum"""

        if self.intensities is None:
            self.collect()

        plt.plot(self.settings.wavelengths, self.intensities)
        if wavelength_limits is not None:
            plt.xlim(wavelength_limits)
        if intensity_limits is not None:
            plt.ylim(intensity_limits)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity (arb. u.)")
        plt.show()

    def save(self, save_directory: str = "", save_filename: str = None):
        """Save the spectrum"""

        if save_filename is None:
            save_filename = strftime("%Y%m%d%H%M%S", localtime()) + "_spectrum"
        save_path = os.path.join(save_directory, save_filename)

        if self.intensities is None:
            self.collect()

        np.savetxt(
            os.path.join(save_path + ".csv"),
            np.transpose(
                np.vstack(
                    (
                        self.settings.wavelengths,
                        self.intensities,
                    )
                )
            ),
            delimiter=",",
        )
