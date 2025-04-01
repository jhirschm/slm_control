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


    if dir_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_path = dir_path.parent / f"{dir_path.name}_{timestamp}"
        print(f"[New Run] Directory already exists. Creating new directory with timestamp: {dir_path}")
    dir_path.mkdir(parents=True, exist_ok=False)
    print(f"[New Run] Created new working directory: {dir_path}")



    return dir_path

def parse_args():
    parser = argparse.ArgumentParser(description="FROG + SLM Optimization Run")
    parser.add_argument('--dir', required=True, help='Path to working directory')
    parser.add_argument('--target_spectrogram', type=str, help='Path to target FROG spectrogram (.npy file)')


    return parser.parse_args()
    
def save_params_yaml(path, frog_params, slm_params, sampling_params):
    env_info = {
        "python_version": platform.python_version(),
        "packages": {
            dist.key: dist.version for dist in pkg_resources.working_set
        }
    }

    config = {
        "frog_parameters": frog_params,
        "slm_parameters": slm_params,
        "sampling_parameters": sampling_params,
        "environment": env_info
    }

    yaml_path = path / PARAMS_FILE
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"[YAML] Saved parameters to {yaml_path}")


def create_h5_file(path, frog_params, slm_params):
    """
    Creates an HDF5 file and saves settings including FROG, SLM, and error parameters.

    Args:
        path (Path): Path to working directory.
        frog_params (dict): Dictionary of FROG parameters.
        slm_params (dict): Dictionary of SLM parameters.
        error_params (dict): Dictionary of error weighting parameters.
    """
    def flatten_dict(d, parent_key='', sep='/'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple, np.ndarray)):
                items.append((new_key, str(v)))  # Save as string for HDF5 compatibility
            else:
                items.append((new_key, v))
        return dict(items)
    
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


    print("[H5] Settings saved (FROG, SLM, Sampling parameters).")
    return h5_path


def load_or_save_target_spectrogram(path, target_spectrogram):
    h5_path = path / H5_FILE

    
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

    # If the file exists, remove it before saving new content
    if output_csv.exists():
        output_csv.unlink()
        print(f"[INFO] Deleted existing file: {output_csv}")

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




def plot(delay, wavelengths, masked_target, masked_trace, save_name, working_dir):
    """
    Plots the target spectrogram, current spectrogram, and pixel-wise difference,
    then saves the figure as a high-quality PDF.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    if not os.path.exists(working_dir):
        raise FileNotFoundError(f"Error: The directory {working_dir} does not exist.")

    # Ensure masked_target and masked_trace are 2D arrays
    masked_target = np.array(masked_target)
    masked_trace = np.array(masked_trace)

    # Validate delay and wavelength sizes
    ny, nx = masked_trace.shape

    if len(wavelengths) != ny:
        raise ValueError(f"Wavelengths length ({len(wavelengths)}) does not match trace height ({ny})")
    
    if delay is None or len(delay) != nx:
        print(f"[WARNING] Delay vector is invalid or does not match width ({nx}). Reconstructing delay axis...")
        delay = np.linspace(-0.05, 0.05, nx)

    save_path = os.path.join(working_dir, f"{save_name}.pdf")

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    im1 = axes[0].pcolor(delay, wavelengths, masked_target, cmap="jet", shading='auto')
    axes[0].set_title("Target Spectrogram")
    axes[0].set_xlabel("Delay (ps)")
    axes[0].set_ylabel("Wavelength (nm)")
    fig.colorbar(im1, ax=axes[0], label="Intensity")

    im2 = axes[1].pcolor(delay, wavelengths, masked_trace, cmap="jet", shading='auto')
    axes[1].set_title("Current Spectrogram")
    axes[1].set_xlabel("Delay (ps)")
    axes[1].set_ylabel("Wavelength (nm)")
    fig.colorbar(im2, ax=axes[1], label="Intensity")

    difference = masked_trace - masked_target
    im3 = axes[2].pcolor(delay, wavelengths, difference, cmap="bwr", shading='auto',
                         vmin=-np.max(abs(difference)), vmax=np.max(abs(difference)))
    axes[2].set_title("Pixel-by-Pixel Difference")
    axes[2].set_xlabel("Delay (ps)")
    axes[2].set_ylabel("Wavelength (nm)")
    fig.colorbar(im3, ax=axes[2], label="Difference Intensity")

    plt.tight_layout()
    fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
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


def save_iteration_data(h5_path, iteration, phase_mask, trace, masked_trace,
                        polynomial_coeffs):
    h5_path = Path(h5_path)
    with h5py.File(h5_path, 'a') as f:
        if f"iterations/{iteration}" in f:
            raise ValueError(f"Iteration {iteration} already exists.")

        grp = f.create_group(f"iterations/{iteration}")
        grp.create_dataset("phase_mask", data=phase_mask)
        grp.create_dataset("trace", data=trace)
        grp.create_dataset("masked_trace", data=masked_trace)
        grp.create_dataset("polynomial_coeffs", data=polynomial_coeffs)

        
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

def generate_random_data_samples(num_samples, bounds, slm, slm_parameters, frog, frog_parameters, h5_path, plot_dir, target , seed=42):
    np.random.seed(seed)
    print(f"[INIT] Generating {num_samples} samples with seed {seed}")

    for step in range(num_samples):
        # Sample random coefficients from bounds
        coeffs = np.array([
            np.random.uniform(low, high) for (low, high) in bounds
        ])

        # Generate and load phase pattern
        unwrapped_phase = phase_mask(coeffs, slm_parameters["effective_scale"], slm_parameters["width"])
        pattern = generate_phase_pattern(
            slm_parameters["width"],
            slm_parameters["height"],
            unwrapped_phase,
            Path(h5_path).parent / "temp.csv"
        )
        slm.load_csv(str(Path(h5_path).parent / "temp.csv"))
        time.sleep(1) 
        # Capture trace
        trace, real_positions = frog.run(close=False)
        trace = trace.squeeze()

        # Optionally normalize or crop trace here if needed
        wavelengths, masked_trace = frog.mask_trace(trace, frog_parameters["wavelength_range"])

        # Save sample
        save_iteration_data(
            h5_path=h5_path,
            iteration=step,
            phase_mask=pattern,
            trace=trace,
            masked_trace=masked_trace,
            polynomial_coeffs=coeffs,
        )

        if step % 10 == 0:
            plot_name = f"plot_sample{step}"
            plot(frog.delay_vector, wavelengths, target, masked_trace, plot_name, plot_dir)

        print(f"Coefficients:")
        print(coeffs)
        print(f"[SAVED] Sample {step} complete")

    print("[DONE] Random data generation complete.")




# --------------------------------
# Main Logic
# --------------------------------
def main():
    args = parse_args()

    print("-------------------")
    print("Initiating main logic")
    print("-------------------")

    
    sampling_parameters = {

        "random_seed": 100,
        "num_samples":100,
        # Coefficient bounds: required by both optimizers
        "bounds": [
            (0.0, 0.0),             # a5 fixed
            (0, 0),        # a4 free
            (-1e-5, 1e-5),          # a3 disjoint constraint applied
            (0, 1e-1),           # a2 free
            (0.0, 0.0),             # a1 fixed
            (0.0, 0.0)              # a0 fixed
        ]

    }
        



    # FROG Parameters
    frog_parameters = {
        "integration_time": 0.1,
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

    working_dir = setup_working_directory(args)
    plot_dir = create_runtime_plots_directory(working_dir)

    # Save parameters and environment info (Step 3.1)
    
    params_path = working_dir / PARAMS_FILE
    

    create_h5_file(working_dir, frog_parameters, slm_parameters)
    print("-------------------")
    print("Working directory ready")
    print("-------------------")

    print("-------------------")
    print("Loading Target Frog Spectrogram")
    print("-------------------")

    target_wavelength_data, target_delay_data, target_spectral_array = load_or_save_target_spectrogram(working_dir, args.target_spectrogram)
    # Compute number of steps in current FROG scan
    num_current_delay_points = int(
        (frog_parameters["scan_range"][1] - frog_parameters["scan_range"][0]) / frog_parameters["step_size"]
    ) + 1

    # Crop the target spectrogram delay axis to match current scan (centered)
    if len(target_delay_data) > num_current_delay_points:
        extra = len(target_delay_data) - num_current_delay_points
        start = extra // 2
        end = start + num_current_delay_points

        print(f"[INFO] Cropping target delay axis: {len(target_delay_data)} → {num_current_delay_points}")
        target_delay_data = target_delay_data[start:end]
        target_spectral_array = target_spectral_array[:, start:end]
    elif len(target_delay_data) < num_current_delay_points:
        raise ValueError("Target spectrogram has fewer delay points than current scan range. Cannot align.")

    print("-------------------")
    print("Completed Loading Target Frog Spectrogram")
    print("-------------------")

    print("-------------------")
    print("Generating initial phase mask")
    print("-------------------")

    # Define polynomial coefficients [a5, a4, a3, a2, a1, a0]
    # initial_coefficients =  [0.0, 0, 0, 0, 0.00, 0] 
    initial_coefficients = [0, 0, 1.051e-6, 1.611e-2, 0, 0]

    # initial_coefficients = [0, 0, 1.051e-6, 1.611e-2, 0, 0]
    coefficients = initial_coefficients

    #combine these steps
    unwrapped_phase = phase_mask(coefficients, slm_parameters["effective_scale"], slm_parameters["width"])
    pattern = generate_phase_pattern(slm_parameters["width"], slm_parameters["height"], unwrapped_phase, working_dir / "temp.csv")

    csv_path = working_dir / "temp.csv"
    slm.load_csv(str(csv_path))
    time.sleep(1)

    print("-------------------")
    print("Initial phase mask loaded")
    print("-------------------")

    print("-------------------")
    print("Take Initial FROG Measurment")
    print("-------------------")
    trace, real_positions = frog.run(close=False)
    original_trace = trace
    trace = trace.squeeze()
    wavelengths, masked_trace = frog.mask_trace(trace, frog_parameters["wavelength_range"])
    wavelegnths_target, masked_target = frog.mask_trace(target_spectral_array, frog_parameters["wavelength_range"])

    print(masked_trace.shape)
    print(real_positions.shape)
    frog.plot(original_trace, real_positions, wavelength_range=frog_parameters["wavelength_range"], time_axis=False)

    # frog.plot(trace, real_positions, wavelength_range=(490, 560), time_axis=True) 
    print("-------------------")
    print("FROG Measurment Complete")
    print("-------------------")

    initial_coefficients = [0, 0, 0, 0, 0, 0]
    coefficients = initial_coefficients

    #combine these steps
    unwrapped_phase = phase_mask(coefficients, slm_parameters["effective_scale"], slm_parameters["width"])
    pattern = generate_phase_pattern(slm_parameters["width"], slm_parameters["height"], unwrapped_phase, working_dir / "temp.csv")

    csv_path = working_dir / "temp.csv"
    slm.close()
    slm = SantecSLM(**slm_init_params)
    slm.load_csv(str(csv_path))
    time.sleep(1)

    print("-------------------")
    print("Initial phase mask loaded")
    print("-------------------")

    print("-------------------")
    print("Take Initial FROG Measurment")
    print("-------------------")
    trace, real_positions = frog.run(close=False)
    original_trace = trace
    trace = trace.squeeze()
    wavelengths, masked_trace = frog.mask_trace(trace, frog_parameters["wavelength_range"])
    wavelegnths_target, masked_target = frog.mask_trace(target_spectral_array, frog_parameters["wavelength_range"])

    print(masked_trace.shape)
    print(real_positions.shape)
    frog.plot(original_trace, real_positions, wavelength_range=frog_parameters["wavelength_range"], time_axis=False)


    
    # # Save static data (only once)
    # save_static_data(working_dir / H5_FILE, real_positions, target_delay_data)



    # generate_random_data_samples(
    #     num_samples=sampling_parameters["num_samples"],
    #     bounds=sampling_parameters["bounds"],
    #     slm=slm,
    #     slm_parameters=slm_parameters,
    #     frog=frog,
    #     frog_parameters=frog_parameters,
    #     h5_path=working_dir / H5_FILE,
    #     plot_dir=plot_dir,
    #     seed=sampling_parameters["random_seed"],
    #     target = masked_target
    # )






    
    

    
    

   

    frog.close_frog()
    slm.close()

    






if __name__ == "__main__":
    main()