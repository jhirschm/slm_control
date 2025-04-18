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
import inspect
import pandas as pd
import matplotlib.pyplot as plt
# Set PDF-friendly font rendering
plt.rcParams.update({
    "font.family": "serif",
    "pdf.fonttype": 42,  # Ensures fonts are stored as actual text in the PDF
    "ps.fonttype": 42
})

from frog_class import FROG
from santec_slm import SantecSLM

import yaml
import h5py
import platform
import pkg_resources
import numpy as np
import sys

from scipy.optimize import minimize
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ssim

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
    
def save_params_yaml(path, frog_params, slm_params, error_params, max_steps, error_min):
    """
    Saves the experimental parameters to a YAML file.

    Args:
        path (Path): Path to working directory.
        frog_params (dict): Dictionary of FROG parameters.
        slm_params (dict): Dictionary of SLM parameters.
        error_params (dict): Dictionary of error weighting parameters.
        max_steps (int): Maximum number of iterations.
        error_min (float): Error threshold for convergence.
    """
    env_info = {
        "python_version": platform.python_version(),
        "packages": {
            dist.key: dist.version for dist in pkg_resources.working_set
        }
    }

    config = {
        "frog_parameters": frog_params,
        "slm_parameters": slm_params,
        "error_parameters": error_params,
        "max_steps": max_steps,
        "error_min": error_min,
        "environment": env_info
    }

    yaml_path = path / PARAMS_FILE
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"[YAML] Saved parameters to {yaml_path}")

def create_h5_file(path, frog_params, slm_params, error_params):
    """
    Creates an HDF5 file and saves settings including FROG, SLM, and error parameters.

    Args:
        path (Path): Path to working directory.
        frog_params (dict): Dictionary of FROG parameters.
        slm_params (dict): Dictionary of SLM parameters.
        error_params (dict): Dictionary of error weighting parameters.
    """
    h5_path = path / H5_FILE
    with h5py.File(h5_path, 'w') as f:
        # Save FROG parameters
        frog_group = f.create_group("settings/frog")
        for k, v in frog_params.items():
            frog_group.attrs[k] = v

        # Save SLM parameters
        slm_group = f.create_group("settings/slm")
        for k, v in slm_params.items():
            slm_group.attrs[k] = v

        # Save Error parameters
        error_group = f.create_group("settings/error")
        for k, v in error_params.items():
            error_group.attrs[k] = v

    print("[H5] Settings saved (FROG, SLM, Error parameters).")
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

# Loss function components
def wasserstein_loss(target, current):
    """Compute Wasserstein (Earth Mover's Distance)"""
    return wasserstein_distance(target.flatten(), current.flatten())

def hybrid_loss(target, current, alpha=1.0, beta=1.0):
    """
    Compute a weighted hybrid loss function:
    L = alpha * Wasserstein Distance + beta * (1 - SSIM)
    """
    w_dist = wasserstein_loss(target, current)
    ssim_val = ssim(target, current, data_range=target.max() - target.min())

    # Compute the final loss
    loss = alpha * w_dist + beta * (1 - ssim_val)
    return loss, w_dist, ssim_val

def plot(delay, wavelengths, masked_target, masked_trace, save_name, working_dir):
    """
    Plots the target spectrogram, current spectrogram, and pixel-wise difference,
    then saves the figure as a high-quality PDF.

    Args:
        delay (array): Delay axis values.
        wavelengths (array): Wavelength axis values.
        masked_target (array): Target spectrogram.
        masked_trace (array): Current spectrogram.
        save_name (str): Name of the saved PDF file.
        working_dir (str): Directory to save the PDF.
    """

    # Ensure the save directory exists
    # Ensure the provided working_dir exists (but do NOT create it)
    if not os.path.exists(working_dir):
        raise FileNotFoundError(f"Error: The directory {working_dir} does not exist.")

    # Define the full save path
    save_path = os.path.join(working_dir, f"{save_name}.pdf")

    # Create figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Plot the target spectrogram
    im1 = axes[0].pcolor(delay, wavelengths, masked_target, cmap="jet", shading='auto')
    axes[0].set_title("Target Spectrogram")
    axes[0].set_xlabel("Delay (ps)")
    axes[0].set_ylabel("Wavelength (nm)")
    fig.colorbar(im1, ax=axes[0], label="Intensity")

    # Plot the current spectrogram
    im2 = axes[1].pcolor(delay, wavelengths, masked_trace, cmap="jet", shading='auto')
    axes[1].set_title("Current Spectrogram")
    axes[1].set_xlabel("Delay (ps)")
    axes[1].set_ylabel("Wavelength (nm)")
    fig.colorbar(im2, ax=axes[1], label="Intensity")

    # Compute and plot the pixel-by-pixel difference
    difference = masked_trace - masked_target  # Ensure both have the same shape before subtracting
    im3 = axes[2].pcolor(delay, wavelengths, difference, cmap="bwr", shading='auto',
                         vmin=-np.max(abs(difference)), vmax=np.max(abs(difference)))
    axes[2].set_title("Pixel-by-Pixel Difference")
    axes[2].set_xlabel("Delay (ps)")
    axes[2].set_ylabel("Wavelength (nm)")
    fig.colorbar(im3, ax=axes[2], label="Difference Intensity")

    # Adjust layout
    plt.tight_layout()

    # Save the figure as a high-quality PDF
    fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")

    # Close figure to free memory
    plt.close(fig)

    print(f"Figure saved at: {save_path}")

def create_runtime_plots_directory(working_dir):
    """
    Creates a 'runtimePlots' subdirectory inside the given working directory.
    If 'runtimePlots' already exists, appends a timestamp to create a unique directory.

    Args:
        working_dir (str): The parent directory where the subdirectory will be created.

    Returns:
        str: The path of the created directory.
    """
    base_dir = os.path.join(working_dir, "runtimePlots")

    # If the directory exists, append timestamp
    if os.path.exists(base_dir):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_dir = f"{base_dir}_{timestamp}"
    else:
        new_dir = base_dir

    # Create the directory
    os.makedirs(new_dir, exist_ok=True)

    print(f"Directory created: {new_dir}")
    return new_dir

def save_iteration_data(h5_path, iteration, phase_mask, trace, masked_trace, polynomial_coeffs, wasserstein_loss, ssim, total_loss):
    """
    Saves per-iteration data into the HDF5 file.

    Args:
        h5_path (Path): Path to the HDF5 file.
        iteration (int): Current iteration number.
        phase_mask (2D array): The phase mask applied to the SLM.
        trace (2D array): The measured FROG trace.
        masked_trace (2D array): The masked FROG trace.
        polynomial_coeffs (1D array): The polynomial coefficients used for the phase mask.
        wasserstein_loss (float): Wasserstein distance loss.
        ssim (float): Structural similarity index.
        total_loss (float): Weighted loss function combining Wasserstein and SSIM.
    """
    h5_path = Path(h5_path)  # Ensure it's a Path object
    with h5py.File(h5_path, 'a') as f:  # Open in append mode

        # Check if the iteration already exists
        if f"iterations/{iteration}" in f:
            raise ValueError(f"Error: Iteration {iteration} already exists in HDF5 file. Data will not be overwritten.")

        # Create the iteration group
        iter_group = f.create_group(f"iterations/{iteration}")

        # Save phase mask
        iter_group.create_dataset("phase_mask", data=phase_mask)

        # Save FROG traces
        iter_group.create_dataset("trace", data=trace)
        iter_group.create_dataset("masked_trace", data=masked_trace)

        # Save polynomial coefficients
        iter_group.create_dataset("polynomial_coeffs", data=polynomial_coeffs)

        # Save loss metrics as attributes
        iter_group.attrs["wasserstein_loss"] = wasserstein_loss
        iter_group.attrs["ssim"] = ssim
        iter_group.attrs["total_loss"] = total_loss

    print(f"[H5] Iteration {iteration} data saved successfully.")

def save_static_data(h5_path, real_positions, delay_data):
    """
    Saves real_positions and delay_data one time in HDF5.

    Args:
        h5_path (Path): Path to the HDF5 file.
        real_positions (1D array): The real positions from the FROG measurement.
        delay_data (1D array): The delay axis data.
    """
    h5_path = Path(h5_path)  # Ensure it's a Path object
    with h5py.File(h5_path, 'a') as f:  # 'a' mode to append

        # Ensure static data is stored only once
        if "static_data" not in f:
            static_grp = f.create_group("static_data")
            static_grp.create_dataset("real_positions", data=real_positions, compression="gzip")
            static_grp.create_dataset("delay_data", data=delay_data, compression="gzip")
            print("[H5] Static data saved (real_positions & delay_data).")
        else:
            print("[H5] Static data already exists. Skipping save.")

# --------------------------------
# Main Logic
# --------------------------------
def main():
    args = parse_args()

    print("-------------------")
    print("Initiating main logic")
    print("-------------------")

    #Run Parameters

    #Error
    error_parameters = {
        "wasserstein_weight": 0.002,
        "ssim_weight": 1
    }

    # FROG Parameters
    frog_parameters = {
        "integration_time": 0.5,
        "averaging": 1,
        "central_motor_position": 0.165,
        "scan_range": (-0.05, 0.05),
        "step_size": 0.001,
        "wavelength_range":(480,560)
    }

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

    print("-------------------")
    print("Setting up FROG")
    print("-------------------")
    

    # Print nicely
    for key, value in frog_parameters.items():
        print(f"{key}: {value}")

    # Get valid parameters for FROG's __init__ method
        valid_frog_params = inspect.signature(FROG.__init__).parameters

    # Filter out invalid parameters
    filtered_frog_params = {k: v for k, v in frog_parameters.items() if k in valid_frog_params}

    frog = FROG(**filtered_frog_params) #FROG Checked and runs properly
    
    print("-------------------")
    print("FROG Initiated")
    print("-------------------")

    print("-------------------")
    print("Setting up SLM")
    print("-------------------")

    
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
    plot_dir = create_runtime_plots_directory(working_dir)

    # Save parameters and environment info (Step 3.1)
    
    params_path = working_dir / PARAMS_FILE
    if not args.resume:
        if params_path.exists():
            raise FileExistsError(f"{PARAMS_FILE} already exists in {working_dir}")
        save_params_yaml(working_dir, frog_parameters, slm_parameters, error_parameters, args.max_steps, args.error_min)
        print(f"Saved parameters to {params_path}")
    else:
        print(f"[Resume Mode] Parameters will be loaded from existing {params_path}")


    create_h5_file(working_dir, frog_parameters, slm_parameters, error_parameters)
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

    #combine these steps
    unwrapped_phase = phase_mask(coefficients, slm_parameters["effective_scale"], slm_parameters["width"])
    pattern = generate_phase_pattern(slm_parameters["width"], slm_parameters["height"], unwrapped_phase, working_dir / "temp.csv")

    csv_path = working_dir / "temp.csv"
    slm.load_csv(str(csv_path))

    print("-------------------")
    print("Initial phase mask loaded")
    print("-------------------")

    print("-------------------")
    print("Take Initial FROG Measurment")
    print("-------------------")
    trace, real_positions = frog.run()
    trace = trace.squeeze()
    wavelengths, masked_trace = frog.mask_trace(trace, frog_parameters["wavelength_range"])
    wavelegnths_target, masked_target = frog.mask_trace(target_spectral_array, frog_parameters["wavelength_range"])



    # frog.plot(trace, real_positions, wavelength_range=(490, 560), time_axis=True) 
    print("-------------------")
    print("FROG Measurment Complete")
    print("-------------------")

    print("-------------------")
    print("Calculate Initial Error")
    print("-------------------")

    
    plot(target_delay_data, wavelengths, masked_target, masked_trace, "plot0.pdf", plot_dir)
    loss, wasserstein_distance, ssim = hybrid_loss(masked_target, masked_trace, error_parameters["wasserstein_weight"], error_parameters["ssim_weight"])
    print(wasserstein_distance)
    print(ssim)
    print(loss)

    # Save static data (only once)
    save_static_data(working_dir / H5_FILE, real_positions, target_delay_data)

    # Inside iteration loop (or initial step)
    iteration = 0  # Update accordingly in loop
    save_iteration_data(
        working_dir / H5_FILE,
        iteration,
        phase_mask=pattern,
        trace=trace,
        masked_trace=masked_trace,
        polynomial_coeffs=coefficients,
        wasserstein_loss=wasserstein_distance,
        ssim=ssim,
        total_loss=loss
    )



    

    
    #save frog and phase to h5

    #start error scan

    #make folder in directory that has some pdfs. the pdf will have image of current frog, target frog, error and phase mask with coefficients labeled. generate every 20 iterations and save

    # slm.load_csv(r"C:/Users/lasopr/Downloads/phase_13.csv")


    slm.close()

    






if __name__ == "__main__":
    main()