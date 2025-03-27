# ================================
# Main script to control two devices:
# 1. FROG
# 2. SLM
# Steps
# 1) Initiate FROG v
# 2) Initiate SLM v 
# 3) Setup working directory
#   3.1) Save file of parameters for diagnostics, python environment and versions, number of steps and error minimum
#   3.2) Create H5 file for run time
# 4) Load target FROG spectrogram, must match to FROG settings
#   4.5) Save target spectrogram to H5 file
# 5) Create initial profile for SLM, save to temp CSV, load CSV
# 6) Take FROG measurement
# 7) Calculate error between new FROG measurement and FROG Target
# 8) Save new Frog measurement, 1D SLM phase mask, polynomial coefficients, error to H5
# 9) Update polynomial coefficients, generate new phase mask, overwrite temp csv
# 10) Upload new phase mask to SLM
# 11) Repeat 6)--10) until error minimum reached or number of steps times out
# 12) P

#Structuring H5 Like
# runtime_data.h5
# ├── target_spectrogram
# ├── settings/
# │   ├── frog
# │   └── slm
# ├── iterations/
#     ├── 0/
#     │   ├── phase_mask
#     │   ├── frog_spectrum
#     │   ├── polynomial_coeffs
#     │   └── error
#     ├── 1/
#     ...
# ================================

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

import pandas as pd

from frog_class import FROG
from santec_slm import SantecSLM

import yaml
import h5py
import platform
import pkg_resources
import numpy as np
import sys

# Global Functions
# Global flag for user interrupt
interrupted = False
PARAMS_FILE = "params.yaml"
H5_FILE = "runtime_data.h5"


def signal_handler(sig, frame):
    global interrupted
    print("\n[INFO] Interrupt received. Cleaning up...")
    interrupted = True
# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# ------------------------
# Load state if available
# ------------------------
STATE_FILE = "optimizer_state.pkl"

def save_state(step, best_params, optimizer_state):
    with open(STATE_FILE, "wb") as f:
        pickle.dump({
            "step": step,
            "best_params": best_params,
            "optimizer_state": optimizer_state
        }, f)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "rb") as f:
            return pickle.load(f)
    return None


def setup_working_directory(args):
    dir_path = Path(args.dir).resolve()

    if args.resume:
        if not dir_path.exists():
            print(f"Error: Resume selected but directory {dir_path} does not exist.", file=sys.stderr)
            sys.exit(1)
        print(f"[Resume Mode] Resuming from existing directory: {dir_path}")
    else:
        if dir_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_path = dir_path.parent / f"{dir_path.name}_{timestamp}"
            print(f"[New Run] Directory already exists. Creating new directory with timestamp: {dir_path}")
        dir_path.mkdir(parents=True, exist_ok=False)
        print(f"[New Run] Created new working directory: {dir_path}")

    print(f"[INFO] Max Steps: {args.max_steps}")
    print(f"[INFO] Error Minimum: {args.error_min:.2e}")

    return dir_path

def parse_args():
    parser = argparse.ArgumentParser(description="FROG + SLM Optimization Run")
    parser.add_argument('--dir', required=True, help='Path to working directory')
    parser.add_argument('--resume', action='store_true', help='Resume from previous run')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum number of optimization steps')
    parser.add_argument('--error_min', type=float, default=1e-4, help='Error threshold for convergence')
    parser.add_argument('--target_spectrogram', type=str, help='Path to target FROG spectrogram (.npy file)')

    return parser.parse_args()
    
def save_params_yaml(path, frog_params, slm_params, max_steps, error_min):
    env_info = {
        "python_version": platform.python_version(),
        "packages": {
            dist.key: dist.version for dist in pkg_resources.working_set
        }
    }

    config = {
        "frog_parameters": frog_params,
        "slm_parameters": slm_params,
        "max_steps": max_steps,
        "error_min": error_min,
        "environment": env_info
    }

    with open(path / PARAMS_FILE, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def create_h5_file(path, frog_params, slm_params):
    h5_path = path / H5_FILE
    with h5py.File(h5_path, 'w') as f:
        # Save parameters
        frog_group = f.create_group("settings/frog")
        for k, v in frog_params.items():
            frog_group.attrs[k] = v

        slm_group = f.create_group("settings/slm")
        for k, v in slm_params.items():
            slm_group.attrs[k] = v

    return h5_path

def load_or_save_target_spectrogram(path, resume, target_spectrogram):
    h5_path = path / H5_FILE

    if resume:
        print("Loading target spectrogram from H5 file...")
        with h5py.File(h5_path, 'r') as f:
            if 'target_spectrogram/spectral' not in f:
                raise ValueError("H5 file missing target spectrogram datasets.")
            
            # Load stored datasets
            wavelength_data = f['target_spectrogram/wavelength'][:]
            delay_data = f['target_spectrogram/delay'][:]
            spectral_array = f['target_spectrogram/spectral'][:]

        return wavelength_data, delay_data, spectral_array

    else:
        print(f"Loading target spectrogram from file: {target_spectrogram}")
        target_wavelength, target_delay, target_spectral = load_pyfrogNative_target_spectrogram(target_spectrogram)

        print(f"Target spectrogram shape: {target_spectral.shape}")

        with h5py.File(h5_path, 'a') as f:
            grp = f.create_group("target_spectrogram")
            grp.create_dataset("wavelength", data=target_wavelength)
            grp.create_dataset("delay", data=target_delay)
            grp.create_dataset("spectral", data=target_spectral)

        return target_wavelength, target_delay, target_spectral
    
def load_pyfrogNative_target_spectrogram(file_path):
    # Initialize lists to hold data components
    header = ""
    delay_data = []
    wavelength_data = []
    spectral_data = []

    # Read the file and parse header, delay, wavelength, and spectral data
    with open(file_path, 'r') as file:
        lines = file.readlines()
        header = lines[0].strip()  # First line is the header
        
        # Parse delay data (first line after header)
        delay_data = np.array([float(value) for value in lines[1].strip().split('\t')])
        
        # Parse wavelength calibration data (second line after header)
        wavelength_data = np.array([float(value) for value in lines[2].strip().split('\t')])
        
        # Parse the spectral data (remaining lines)
        for line in lines[3:]:
            spectral_row = [float(value) for value in line.strip().split('\t')]
            spectral_data.append(spectral_row)

    # Convert spectral data to a 2D numpy array
    spectral_array = np.array(spectral_data)

    # Transpose the array if necessary to align with axes
    if spectral_array.shape[0] != len(wavelength_data):
        spectral_array = spectral_array.T

    return wavelength_data, delay_data, spectral_array

def generate_phase_pattern(width, height, phase_values, output_csv):
    """
    Generate a 2D phase pattern based on column-wise phase values.

    Args:
        width (int): Width of the array (number of columns).
        height (int): Height of the array (number of rows).
        phase_values (list or array): Array specifying phase values (0-1023) for each column.
        output_csv (str): File path to save the CSV.

    Returns:
        None: Saves the CSV and displays the image.
    """
    if len(phase_values) != width:
        raise ValueError(f"Length of phase_values ({len(phase_values)}) must match the width ({width}).")

    # Ensure output path is valid
    output_csv = Path(output_csv).resolve()

    # Create a 2D array by repeating the phase values for each row
    pattern = np.tile(phase_values, (height, 1)).astype(int)  # Ensure integer values

    # Save the pattern as a CSV file
    df = pd.DataFrame(pattern)
    df.insert(0, "Y/X", range(height))  # Add row labels as the first column
    df.columns = ["Y/X"] + list(range(width))  # Add column labels as integers

    # Explicitly ensure all values are integers
    df = df.astype(int)

    # Save to CSV in the required format
    df.to_csv(output_csv, index=False)
    return pattern

def phase_mask(polynomial_coef, scaler, num_pixels):
    x_center = num_pixels//2
    x = np.arange(num_pixels)
    x_normalized = x-x_center

    phi = np.polyval(polynomial_coef, x_normalized)
    phi_scaled = phi/(2 * np.pi) *scaler
    phi_wrapped = np.mod(phi_scaled, scaler+1)

    return phi_wrapped
# --------------------------------
# Main Logic
# --------------------------------
def main():
    args = parse_args()

    print("-------------------")
    print("Initiating main logic")
    print("-------------------")


    print("-------------------")
    print("Setting up FROG")
    print("-------------------")
    # FROG Parameters
    frog_parameters = {
        "integration_time": 0.5,
        "averaging": 1,
        "central_motor_position": 0.165,
        "scan_range": (-0.05, 0.05),
        "step_size": 0.005
    }

    # Print nicely
    for key, value in frog_parameters.items():
        print(f"{key}: {value}")

    frog = FROG(**frog_parameters) #FROG Checked and runs properly
    
    print("-------------------")
    print("FROG Initiated")
    print("-------------------")

    print("-------------------")
    print("Setting up SLM")
    print("-------------------")

    slm_parameters = {
        "slm_number": 1,
        "bitdepth": 10,
        "wave_um":1.035,
        "rate":120,
        "phase_range":218, #hard to set 200
        "scale" : 1023,
        "effective_scale":939, #1023/2.18*2
        "width": 1920,
        "height": 1080
    }
    slm_init_params = {key: slm_parameters[key] for key in ["slm_number", "bitdepth", "wave_um", "rate", "phase_range"]}

    slm = SantecSLM(**slm_init_params)

    print("-------------------")
    print("SLM Initiated")
    print("-------------------")

    print("-------------------")
    print("Load target FROG Spectrogram")
    print("-------------------")

    print("-------------------")
    print("Setup working directory")
    print("-------------------")

    # working_dir, resumed_flag, max_steps, error_min = parse_args_and_setup_directory()
    working_dir = setup_working_directory(args)

    # Save parameters and environment info (Step 3.1)
    
    params_path = working_dir / PARAMS_FILE
    if not args.resume:
        if params_path.exists():
            raise FileExistsError(f"{PARAMS_FILE} already exists in {working_dir}")
        save_params_yaml(working_dir, frog_parameters, slm_parameters, args.max_steps, args.error_min)
        print(f"Saved parameters to {params_path}")
    else:
        print(f"[Resume Mode] Parameters will be loaded from existing {params_path}")


    create_h5_file(working_dir, frog_parameters, slm_parameters)
    print("-------------------")
    print("Working directory ready")
    print("-------------------")

    print("-------------------")
    print("Loading Target Frog Spectrogram")
    print("-------------------")

    target_wavelength_data, target_delay_data, target_spectral_array = load_or_save_target_spectrogram(working_dir, args.resume, args.target_spectrogram)

    print("-------------------")
    print("Completed Loading Target Frog Spectrogram")
    print("-------------------")

    print("-------------------")
    print("Generating initial phase mask")
    print("-------------------")

    # Define polynomial coefficients [a5, a4, a3, a2, a1, a0]
    coefficients = [0.0, -0.000000000005, -0.000000031, 0.00024, 0.00, 0]  # Example polynomial
    unwrapped_phase = phase_mask(coefficients, slm_parameters["effective_scale"], slm_parameters["width"])
    pattern = generate_phase_pattern(slm_parameters["width"], slm_parameters["height"], unwrapped_phase, working_dir / "temp.csv")
    # target_path = 
    csv_path = working_dir / "temp.csv"
    print(csv_path)
    slm.load_csv(str(csv_path))
    # slm.load_csv(r"C:/Users/lasopr/Downloads/phase_13.csv")
    slm.close()

    






if __name__ == "__main__":
    main()