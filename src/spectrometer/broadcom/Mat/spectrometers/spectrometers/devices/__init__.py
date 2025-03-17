"""Initialization"""

import importlib
import pkgutil
import os

from .base import Spectrometer

devices_directory = os.path.dirname(__file__)
for _, name, _ in pkgutil.iter_modules([devices_directory]):
    importlib.import_module("." + name, __package__)

device_classes = {
    DeviceClass.__name__: DeviceClass
    for DeviceClass in Spectrometer.__subclasses__()
}
