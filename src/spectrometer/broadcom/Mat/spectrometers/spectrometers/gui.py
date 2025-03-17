"""Spectrometer GUI functions"""

import time
import threading
import copy
import dataclasses
import numpy as np

from multiprocessing import Process, Queue

from PyQt5 import QtWidgets, QtGui

from spectrometers.ui.mainwindow import Ui_MainWindow
from spectrometers.device import find_devices, choose_device
from spectrometers.spectrum import Spectrum, SpectrumSettings


@dataclasses.dataclass
class PlotSpectrumSettings:
    """Object containing spectrum plot settings"""

    xlims: list[float] = None
    ylims: list[float] = None
    ylimtype: str = "Free"
    xdata = None
    line = None
    exposure = None


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """Main window for spectrometer GUI"""

    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.device = None
        self.device_info = None
        self.find_devices()
        self.choose_device(self.comboBoxSpectrometers.currentText())
        self.spectrum = Spectrum(self.device)
        self.lineEditExposure.setText(str(self.spectrum.settings.exposure))
        self.comboBoxSpectrometers.textActivated.connect(self.choose_device)
        self.lineEditExposure.editingFinished.connect(self.update_exposure)
        self.lineEditWavelengthMin.editingFinished.connect(
            self.plot_spectrum_xlim_min
        )
        self.lineEditWavelengthMax.editingFinished.connect(
            self.plot_spectrum_xlim_max
        )
        self.plot_spectrum_settings = PlotSpectrumSettings(
            xlims=[
                self.spectrum.settings.wavelengths[0],
                self.spectrum.settings.wavelengths[-1],
            ]
        )
        self.lineEditWavelengthMin.setText(
            str(np.round(self.plot_spectrum_settings.xlims[0], 1))
        )
        self.lineEditWavelengthMax.setText(
            str(np.round(self.plot_spectrum_settings.xlims[1], 1))
        )
        self.plot_spectrum_settings_stored = PlotSpectrumSettings()
        self.plot_spectrum()
        self.thread_connection = Queue()
        self.thread_activity = PlotSpectrum(self, self.thread_connection)
        self.thread_activity.start()

    def find_devices(self):
        """Find all devices"""
        device_coordinate_list, device_id_list, device_object_list = (
            find_devices()
        )
        if len(device_id_list) == 0:
            device_id_list = ["No devices found."]
        self.device_info = [
            device_coordinate_list,
            device_id_list,
            device_object_list,
        ]
        self.comboBoxSpectrometers.clear()
        self.comboBoxSpectrometers.addItems(self.device_info[1])

    def choose_device(self, new_device_list_label):
        """Choose a device from the drop down list"""
        try:
            new_device_index = self.device_info[1].index(new_device_list_label)
            self.device = choose_device(
                self.device_info[2], self.device_info[0][new_device_index]
            )
            self.spectrum = Spectrum(self.device)
            self.plot_spectrum_settings = PlotSpectrumSettings(
                xlims=[
                    self.spectrum.settings.wavelengths[0],
                    self.spectrum.settings.wavelengths[-1],
                ]
            )
            self.lineEditWavelengthMin.setText(
                str(np.round(self.plot_spectrum_settings.xlims[0], 1))
            )
            self.lineEditWavelengthMax.setText(
                str(np.round(self.plot_spectrum_settings.xlims[1], 1))
            )
        except ValueError as exc:
            raise ValueError("Device not found.") from exc

    def update_exposure(self):
        """Update exposure time"""

        new_exposure_str = self.lineEditExposure.text()
        current_exposure = self.spectrum.settings.exposure
        if new_exposure_str.replace(".", "", 1).isdecimal():
            new_exposure = float(new_exposure_str)
            if (new_exposure > 1e-6) and (new_exposure < 100):
                self.plot_spectrum_settings.exposure = new_exposure
            else:
                self.lineEditExposure.setText(str(current_exposure))
        else:
            self.lineEditExposure.setText(str(current_exposure))

    def plot_spectrum(self):
        """Plot the spectrum"""

        spectrum_current = copy.copy(self.spectrum)
        device_update = False
        if (
            self.plot_spectrum_settings_stored.exposure
            != self.plot_spectrum_settings.exposure
        ):
            self.spectrum.exposure(self.plot_spectrum_settings.exposure)
            self.plot_spectrum_settings_stored.exposure = (
                self.plot_spectrum_settings.exposure
            )
            device_update = True

        self.spectrum.collect()
        if np.shape(self.plot_spectrum_settings_stored.xdata) != np.shape(
            self.spectrum.settings.wavelengths
        ):
            self.MplWidget.canvas.axes.clear()
            (self.plot_spectrum_settings_stored.line,) = (
                self.MplWidget.canvas.axes.plot(
                    self.spectrum.settings.wavelengths,
                    self.spectrum.intensities,
                )
            )
            self.plot_spectrum_settings_stored.xdata = (
                self.spectrum.settings.wavelengths
            )
            self.MplWidget.canvas.axes.set_title("Spectrum")
            self.MplWidget.canvas.axes.set_xlabel("Wavelength (nm)")
            self.MplWidget.canvas.axes.set_ylabel("Intensity (arb. u.)")
        else:
            self.plot_spectrum_settings_stored.line.set_ydata(
                self.spectrum.intensities
            )
            self.MplWidget.canvas.axes.set_ylim(
                [
                    np.min(self.spectrum.intensities),
                    np.max(self.spectrum.intensities),
                ]
            )

        if (
            np.any(
                self.plot_spectrum_settings_stored.xlims
                != self.plot_spectrum_settings.xlims
            )
            or device_update
        ):
            self.MplWidget.canvas.axes.set_xlim(
                self.plot_spectrum_settings.xlims
            )
            self.plot_spectrum_settings_stored.xlims = (
                self.plot_spectrum_settings.xlims
            )

        self.MplWidget.canvas.draw()

    def plot_spectrum_xlim_min(self):
        """Update minimum wavelength limit"""

        new_xlim_str = self.lineEditWavelengthMin.text()
        current_xlim = self.plot_spectrum_settings.xlims[0]
        if new_xlim_str.replace(".", "", 1).isdecimal():
            self.plot_spectrum_settings.xlims = np.sort(
                [float(new_xlim_str), self.plot_spectrum_settings.xlims[1]]
            )
        else:
            self.lineEditWavelengthMin.setText(str(current_xlim))

    def plot_spectrum_xlim_max(self):
        """Update maximum wavelength limit"""

        new_xlim_str = self.lineEditWavelengthMax.text()
        current_xlim = self.plot_spectrum_settings.xlims[1]
        if new_xlim_str.replace(".", "", 1).isdecimal():
            self.plot_spectrum_settings.xlims = np.sort(
                [self.plot_spectrum_settings.xlims[0], float(new_xlim_str)]
            )
        else:
            self.lineEditWavelengthMax.setText(str(current_xlim))

    def closeEvent(self, *args, **kwargs):
        """Run when the window is closed."""
        super(QtWidgets.QMainWindow, self).closeEvent(*args, **kwargs)
        self.thread_activity.stop()


class PlotSpectrum(threading.Thread):
    """The live spectrum plot"""

    def __init__(self, main_window, connection):
        threading.Thread.__init__(self)
        self.main_window = main_window
        self.connection = connection
        self.maintain = True

    def run(self):
        """Start the thread"""
        while self.maintain:
            if not self.connection.empty():
                self.main_window = self.connection.get(timeout=0.3)
            self.main_window.plot_spectrum()
            sleep_time = np.max(
                [0.05, self.main_window.spectrum.settings.exposure]
            )
            time.sleep(sleep_time)

    def stop(self):
        """Stop the thread"""
        self.maintain = False


def run():
    """Run the spectrometer GUI"""

    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
