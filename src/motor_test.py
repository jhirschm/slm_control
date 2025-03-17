import ctypes
# apt_dll_path = "C:\\slm_control\\src\\APT.dll"
# apt_dll = ctypes.CDLL(apt_dll_path)
import thorlabs_apt as apt
import os, sys
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Find all motors
# print(apt.list_available_devices())

motor1_ID = apt.list_available_devices()[0][1]
print(motor1_ID)
# Initialize motor
motor1 = apt.Motor(motor1_ID)

# Adjust motor parameters if needed
motor1.set_stage_axis_info(-4, 4, 1, 1.0)

# Print current motor positions
pos1 = motor1.position
print('Motor 1 position: ' + str(pos1) + ' mm')
motor1.move_to(.165, True)