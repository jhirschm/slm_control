print("here")
# Imports
import _slm_win as slm
import ctypes
import time
import signal
import pickle
import os
from pathlib import Path
from datetime import datetime
import argparse
import inspect
import pandas as pd
import matplotlib.pyplot as plt
# Set PDF-friendly font rendering
plt.rcParams.update({
    "font.family": "serif",
    "pdf.fonttype": 42,  # Ensures fonts are stored as actual text in the PDF
    "ps.fonttype": 42
})
print("here")
# from frog_class import FROG
from santec_slm import SantecSLM
print("here")
import yaml
import h5py
import platform
import pkg_resources
import numpy as np
import sys



# === Parameters ===
initial_coefficients = [0, 3.25305999722143e-17, -2.999964343100393e-8, 2.4090017016278843e-4, 0, 0]  # a5 → a0
additional_coefficients = [0, 0, .5e-8, 0, 0, 0]  # modify freely
additional_coefficients = [0, 0, -.5e-8, 0, 0, 0]  # modify freely
additional_coefficients = [0, 0, 1e-8, 0, 0, 0]  # modify freely
additional_coefficients = [0, 0, 1e-8, .01e-4, 0, 0]  # modify freely
additional_coefficients = [0, 0, 2e-8, .0, 0, 0]  # modify freely
additional_coefficients = [0, 0, 3e-8, .0, 0, 0]  # modify freely
additional_coefficients = [0, 0, -3e-8, .0, 0, 0]  # modify freely
additional_coefficients = [0, 1e-11, 0, .0, 0, 0]  # modify freely
additional_coefficients = [0, 0, 1e-7, -5e-6, 0, 0]  # modify freely






coefficients = [a + b for a, b in zip(initial_coefficients, additional_coefficients)]

slm_parameters = {
    "slm_number": 1,
    "bitdepth": 10,
    "wave_um": 1.035,
    "rate": 120,
    "phase_range": 218,
    "scale": 1023,
    "effective_scale": 939,
    "width": 1920,
    "height": 1080
}

working_dir = Path.cwd()
csv_path = working_dir / "load.csv"

# === Phase mask generation ===
def phase_mask(polynomial_coef, scaler, num_pixels):
    x_center = num_pixels // 2
    x = np.arange(num_pixels)
    x_normalized = x - x_center
    phi = np.polyval(polynomial_coef, x_normalized)
    phi_scaled = phi / (2 * np.pi) * scaler
    return np.mod(phi_scaled, scaler + 1)

def generate_phase_pattern(width, height, phase_values, output_csv):
    pattern = np.tile(phase_values, (height, 1)).astype(int)
    df = pd.DataFrame(pattern)
    df.insert(0, "Y/X", range(height))
    df.columns = ["Y/X"] + list(range(width))
    df = df.astype(int)
    df.to_csv(output_csv, index=False)
    return pattern

# === Build and upload pattern ===
unwrapped_phase = phase_mask(coefficients, slm_parameters["effective_scale"], slm_parameters["width"])
pattern = generate_phase_pattern(slm_parameters["width"], slm_parameters["height"], unwrapped_phase, csv_path)

slm = SantecSLM(
    slm_number=slm_parameters["slm_number"],
    bitdepth=slm_parameters["bitdepth"],
    wave_um=slm_parameters["wave_um"],
    rate=slm_parameters["rate"],
    phase_range=slm_parameters["phase_range"]
)
slm.load_csv(str(csv_path))

print(f"[INFO] Phase mask loaded to SLM from {csv_path}")
