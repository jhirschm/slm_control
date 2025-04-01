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
import ot

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
    parser.add_argument('--patience', type=int, default=5, help='Number of steps without improvement before early stopping')
    parser.add_argument('--min_delta', type=float, default=1e-5, help='Minimum change in loss to count as improvement')


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

        # Flatten and save error parameters
        error_group = f.create_group("settings/error")
        flat_error = flatten_dict(error_params)
        for k, v in flat_error.items():
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

def hybrid_loss(target, current, alpha=1.0, beta=1.0, normalize=False, use_emd2=False):
    """
    Compute a weighted hybrid loss function:
    L = alpha * Wasserstein Distance + beta * (1 - SSIM)
    
    Params:
        target, current: 2D arrays
        alpha: weight on Wasserstein
        beta: weight on SSIM
        normalize: if True, normalize both inputs before computing loss
        use_emd2: if True, use 2D EMD from POT instead of 1D Wasserstein
    """
    if normalize:
        target = target / np.max(target) if np.max(target) > 0 else target
        current = current / np.max(current) if np.max(current) > 0 else current

    if use_emd2:
        w_dist = emd2_pot(target, current)
    else:
        w_dist = wasserstein_loss(target, current)

    ssim_val = ssim(target, current, data_range=target.max() - target.min())
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


def save_iteration_data(h5_path, iteration, phase_mask, trace, masked_trace,
                        polynomial_coeffs, metrics: dict):
    h5_path = Path(h5_path)
    with h5py.File(h5_path, 'a') as f:
        if f"iterations/{iteration}" in f:
            raise ValueError(f"Iteration {iteration} already exists.")

        grp = f.create_group(f"iterations/{iteration}")
        grp.create_dataset("phase_mask", data=phase_mask)
        grp.create_dataset("trace", data=trace)
        grp.create_dataset("masked_trace", data=masked_trace)
        grp.create_dataset("polynomial_coeffs", data=polynomial_coeffs)

        for k, v in metrics.items():
            try:
                grp.attrs[k] = float(v)
            except (TypeError, ValueError):
                grp.attrs[k] = str(v)  # fallback: store as string

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

def optimize_coefficients(initial_coeffs, target_image, slm_parameters, slm_init_params, frog, frog_parameters,
                          error_parameters, h5_path, plot_dir, working_dir):
    from scipy.optimize import minimize, differential_evolution
    global interrupted
    state = load_state(working_dir)
    start_time = datetime.now()

    if state:
        print(f"[RESUME] Resuming from iteration {state['step']}")
        initial_coeffs = state["best_params"]
        step = state["step"]
    else:
        step = 1

    best_loss = float("inf")
    best_coeffs = None
    best_step = -1

    def objective_function(scaled_coeffs, disjoint_indices=None, disjoint_region=None, penalty_factor=1e6):
        nonlocal step, best_loss, best_coeffs, best_step

        # === Unscale coefficients ===
        scalers = error_parameters.get("coefficient_scalers", [1.0] * len(scaled_coeffs))
        coeffs = np.array(scaled_coeffs) * np.array(scalers)

        # === Penalize disallowed disjoint region ===
        penalty = 0.0
        if disjoint_indices and disjoint_region:
            for i in disjoint_indices:
                if disjoint_region[0] < coeffs[i] < disjoint_region[1]:
                    penalty += penalty_factor

        # === Main optimization steps ===
        unwrapped_phase = phase_mask(coeffs, slm_parameters["effective_scale"], slm_parameters["width"])
        pattern = generate_phase_pattern(
            slm_parameters["width"],
            slm_parameters["height"],
            unwrapped_phase,
            Path(working_dir) / "temp.csv"
        )
        
        slm = SantecSLM(**slm_init_params)
        slm.load_csv(str(Path(working_dir) / "temp.csv"))
        time.sleep(1)
    

        trace, real_positions = frog.run(close=False)
        trace = trace.squeeze()
        _, masked_trace = frog.mask_trace(trace, frog_parameters["wavelength_range"])
        _, masked_target = frog.mask_trace(target_image, frog_parameters["wavelength_range"])

        loss, w_dist, ssim_val, grad_loss, ncc = hybrid_loss_v2(
            masked_target,
            masked_trace,
            error_parameters
        )

        total_loss = loss + penalty

        metrics = {
            "wasserstein_loss": w_dist,
            "ssim": ssim_val,
            "total_loss": total_loss,
            "gradient_loss": grad_loss,
            "ncc": ncc
        }
        save_iteration_data(h5_path, step, pattern, trace, masked_trace, coeffs, metrics)

        if step % 10 == 0:
            plot(frog.delay_vector, frog.wavelength_vector, masked_target, masked_trace, f"plot{step}.pdf", plot_dir)

        save_state(step, coeffs, working_dir)

        print(f"[Step {step}] Loss: {loss:.6f} + Penalty: {penalty:.4f} → Total: {total_loss:.6f} | "
              f"Wasserstein: {w_dist:.8f} | SSIM: {ssim_val:.4f} | NCC: {ncc:.4f} | Gradient: {grad_loss:.8f}")

        if total_loss < best_loss:
            best_loss = total_loss
            best_coeffs = coeffs.copy()
            best_step = step

        step += 1
        slm.close()
        if interrupted:
            print(f"[INTERRUPT] Optimization halted at step {step-1}")
            sys.exit(0)

        

        return total_loss

    print(f"[OPTIMIZER INIT] Starting from coefficients: {initial_coeffs}")
    optimizer = error_parameters.get("optimizer", "L-BFGS-B")
    disjoint_indices = error_parameters.get("disjoint_indices")
    disjoint_region = error_parameters.get("disjoint_region")
    penalty_factor = error_parameters.get("penalty_factor", 1e6)

    if optimizer == "L-BFGS-B":
        result = minimize(
            objective_function,
            initial_coeffs,
            args=(disjoint_indices, disjoint_region, penalty_factor),
            method="L-BFGS-B",
            bounds=error_parameters["bounds"],
            options={"maxiter": error_parameters["max_steps"], "disp": True}
        )
    elif optimizer == "DE":
        result = differential_evolution(
            lambda x: objective_function(x, disjoint_indices, disjoint_region, penalty_factor),
            bounds=error_parameters["bounds"],
            strategy="best1bin",
            maxiter=error_parameters["max_steps"],
            disp=True
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    end_time = datetime.now()
    print(f"[COMPLETE] Optimization finished in {end_time - start_time}. Final loss: {result.fun:.6f}")

    if best_coeffs is not None:
        print(f"[BEST] Lowest loss: {best_loss:.6f} at iteration {best_step}")
        best_phase = phase_mask(best_coeffs, slm_parameters["effective_scale"], slm_parameters["width"])
        best_pattern = generate_phase_pattern(
            slm_parameters["width"],
            slm_parameters["height"],
            best_phase,
            Path(working_dir) / "best_temp.csv"
        )
        slm = SantecSLM(**slm_init_params)
        slm.load_csv(str(Path(working_dir) / "best_temp.csv"))

        trace, _ = frog.run(close=False)
        trace = trace.squeeze()
        _, masked_trace = frog.mask_trace(trace, frog_parameters["wavelength_range"])
        _, masked_target = frog.mask_trace(target_image, frog_parameters["wavelength_range"])

        best_plot_name = f"plot_best_iter{best_step}_min_error"
        plot(frog.delay_vector, frog.wavelength_vector, masked_target, masked_trace, best_plot_name, plot_dir)
        print(f"[PLOT] Saved minimal error result: {best_plot_name}.pdf")
        slm.close()

    return best_coeffs

def save_state(step, best_params, working_dir):
    state_path = Path(working_dir) / "optimizer_state.pkl"
    with open(state_path, "wb") as f:
        pickle.dump({
            "step": step,
            "best_params": best_params
        }, f)
    print(f"[STATE] Saved optimizer state at iteration {step} to {state_path}")


def load_state(working_dir):
    state_path = Path(working_dir) / "optimizer_state.pkl"
    if state_path.exists():
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None
def generate_bounds_from_initial(initial_coeffs, percent=30, adjustable_indices=None):
    """
    Generate bounds for each coefficient.

    Args:
        initial_coeffs (list or array): Initial polynomial coefficients.
        percent (float): Percentage defining bound size for adjustable coefficients.
        adjustable_indices (list of ints): Indices to allow adjustment. All others fixed.

    Returns:
        list of (min, max) tuples for each coefficient.
    """
    bounds = []
    scale = (1 + percent)

    for i, coeff in enumerate(initial_coeffs):
        if adjustable_indices and i in adjustable_indices:
            base = abs(coeff)
            if base == 0:
                base = 1e-12  # Avoid zero-bound if coeff is exactly zero
            b = scale * base
            bounds.append((-b, b))
        else:
            bounds.append((coeff, coeff))  # Fixed

    return bounds
def generate_magnitude_bounds(initial_coeffs, exp_range=1, adjustable_indices=None):
    """
    Generate bounds that preserve a limited exponent range but allow sign flips.

    Args:
        initial_coeffs (list or array): Initial values of coefficients.
        exp_range (float): Allowed range in log10 scale (e.g., ±1 → one order of magnitude up/down).
        adjustable_indices (list): Indices to allow adjustment.

    Returns:
        bounds: list of (min, max) tuples.
    """
    bounds = []

    for i, coeff in enumerate(initial_coeffs):
        if adjustable_indices and i not in adjustable_indices:
            bounds.append((coeff, coeff))
            continue

        abs_val = abs(coeff)
        if abs_val == 0:
            abs_val = 10**-12  # fallback

        log10_val = np.log10(abs_val)
        min_exp = log10_val - exp_range
        max_exp = log10_val + exp_range

        lower = -10 ** max_exp
        upper = 10 ** max_exp
        bounds.append((lower, upper))

    return bounds
def emd2_pot(target, current, epsilon=1e-12):
    """
    Computes the 2D Earth Mover's Distance (Wasserstein-2) using POT.
    Ensures both inputs are normalized and valid.
    """
    if target.shape != current.shape:
        raise ValueError(f"Target and current must have the same shape. Got {target.shape} vs {current.shape}")

    # Avoid divide-by-zero or invalid mass vectors
    target = np.clip(target, 0, None) + epsilon
    current = np.clip(current, 0, None) + epsilon

    a = target / np.sum(target)
    b = current / np.sum(current)

    print(a.shape)
    print(b.shape)
    print(a)
    print(b)
    print(np.sum(a))
    print(np.sum(b))
    print("Are they equal?", np.allclose(a,b))

    if not np.isclose(np.sum(a), 1.0) or not np.isclose(np.sum(b), 1.0):
        raise ValueError("Normalized distributions a and b do not sum to 1.")

    n, m = target.shape
    x, y = np.meshgrid(np.arange(m), np.arange(n))
    coords = np.stack([x.ravel(), y.ravel()], axis=1)
    print(coords)
    M = ot.dist(coords, coords, metric='euclidean') ** 2
    print(M)
    # emd2 = ot.emd2(a.ravel(), b.ravel(), M)
    emd2 =  ot.sinkhorn2(a.ravel(), b.ravel(), M, reg=1e-2)

    print(emd2)
    return emd2

def normalized_cross_correlation(target, current):
    target = (target - np.mean(target)) / (np.std(target) + 1e-8)
    current = (current - np.mean(current)) / (np.std(current) + 1e-8)
    return np.mean(target * current)

def gradient_loss(target, current):
    grad_target_x = np.gradient(target, axis=1)
    grad_target_y = np.gradient(target, axis=0)
    grad_current_x = np.gradient(current, axis=1)
    grad_current_y = np.gradient(current, axis=0)

    loss_x = np.mean((grad_target_x - grad_current_x) ** 2)
    loss_y = np.mean((grad_target_y - grad_current_y) ** 2)

    return (loss_x + loss_y) / 2

def hybrid_loss_v2(target, current, weights):
    from skimage.metrics import structural_similarity as ssim

    if weights.get("normalize_data", False):
        target = target / np.max(target) if np.max(target) > 0 else target
        current = current / np.max(current) if np.max(current) > 0 else current

    w_dist = wasserstein_loss(target,current)
    ssim_val = ssim(target, current, data_range=target.max() - target.min())
    grad = gradient_loss(target, current)
    ncc = normalized_cross_correlation(target, current)

    loss = (
        weights.get("wasserstein_weight", 0) * w_dist +
        weights.get("ssim_weight", 0) * (1 - ssim_val) +
        weights.get("gradient_weight", 0) * grad +
        weights.get("ncc_weight", 0) * (1 - ncc)
    )

    return loss, w_dist, ssim_val, grad, ncc





# --------------------------------
# Main Logic
# --------------------------------
def main():
    args = parse_args()

    print("-------------------")
    print("Initiating main logic")
    print("-------------------")

    #Run Parameters
      # Placing here because used to determine bounds for LBFGS
    # adjustable_indices = [2, 3]  # a4, a3, a2
    # bounds = generate_bounds_from_initial(initial_coefficients, percent = 10, adjustable_indices=adjustable_indices)
    # print("Using bounds for LBFGS:")
    # print(bounds)
    #Error
    # error_parameters = {
    #     "optimizer": "L-BFGS-B",
    #     "max_steps": args.max_steps,
    #     "error_min": args.error_min,
    #     "wasserstein_weight": 1E2,
    #     "ssim_weight": 1,
    #     "early_stopping": {
    #         "patience": args.patience,
    #         "min_delta": args.min_delta
    #     },
    #     "bounds": bounds,
    #     "normalize_data":True,
    #     "use_emd2": False, #uses 2D Wasserstein style,
    #     "gradient_weight": 1E4,
    #     "ncc_weight": 0.5

    # }
    error_parameters = {
        "optimizer": "L-BFGS-B",#"DE",
        "max_steps": args.max_steps,
        "error_min": args.error_min,

        "wasserstein_weight": 100,
        "ssim_weight": 1,
        "gradient_weight": 1E4,
        "ncc_weight": 1,

        "normalize_data":True,
        "use_emd2": False, #uses 2D Wasserstein style,

        "early_stopping": {
            "patience": args.patience,
            "min_delta": args.min_delta
        },
        
        # Coefficient bounds: required by both optimizers
        "bounds": [
            (0.0, 0.0),             # a5 fixed
            (0, 2),        # a4 free
            (-4, -2),          # a3 disjoint constraint applied
            (2, 3),           # a2 free
            (0.0, 0.0),             # a1 fixed
            (0.0, 0.0)              # a0 fixed
        ],

        "coefficient_scalers": [
            1.0,          # a5 (fixed)
            1e-12,          # a4 (fixed)
            1e-8,         # a3 → scaled value ~ [-3.5, -2.5]
            1e-4,         # a2 → scaled value ~ [2, 3]
            1.0,          # a1 (fixed)
            1.0           # a0 (fixed)
        ],

        "initial_scaled_coefficients": [
            0.0,      # a5 (fixed)
            0.0,      # a4 (fixed)
        -2.999980608084964,      # a3 → corresponds to -3e-8 unscaled -2.999980608084964e-8
            2.40899207870675,    # a2 → corresponds to 2.409e-4 unscaled 0.000240899207870675
            0.0,      # a1 (fixed)
            0.0       # a0 (fixed)
        ]
        
        # # Disjoint penalty settings (only used if disjoint_indices is not empty)
        # "disjoint_indices": [2],             # apply penalty to a3
        # "disjoint_region": (-1e-4, 1e-4),     # values between -1e-4 and 1e-4 will be penalized
        # "penalty_factor": 1e6                # severity of penalty
    }
        
    # Scale initial coefficients
    scaled_initial_coefficients = error_parameters["initial_scaled_coefficients"]
    coefficient_scalers = error_parameters.get("coefficient_scalers", [1.0] * len(scaled_initial_coefficients))
    initial_coefficients = [c * s for c, s in zip(scaled_initial_coefficients, coefficient_scalers)]
    

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
    coefficients = initial_coefficients

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
    trace, real_positions = frog.run(close=False)
    trace = trace.squeeze()
    wavelengths, masked_trace = frog.mask_trace(trace, frog_parameters["wavelength_range"])
    wavelegnths_target, masked_target = frog.mask_trace(target_spectral_array, frog_parameters["wavelength_range"])

    slm.close()

    # frog.plot(trace, real_positions, wavelength_range=(490, 560), time_axis=True) 
    print("-------------------")
    print("FROG Measurment Complete")
    print("-------------------")

    print("-------------------")
    print("Calculate Initial Error")
    print("-------------------")

    
    plot(target_delay_data, wavelengths, masked_target, masked_trace, "plot0.pdf", plot_dir)
    frog.set_delay_vector(target_delay_data)
    frog.set_wavelength_vector(wavelengths)
    # loss, wasserstein_distance, ssim = hybrid_loss(masked_target, masked_trace, error_parameters["wasserstein_weight"], error_parameters["ssim_weight"], normalize=error_parameters["normalize_data"], use_emd2=error_parameters["use_emd2"])
    loss, wasserstein_distance, ssim, grad_val, ncc_val = hybrid_loss_v2(masked_target, masked_trace, error_parameters)

    # Save static data (only once)
    save_static_data(working_dir / H5_FILE, real_positions, target_delay_data)

    # Inside iteration loop (or initial step)
    iteration = 0  # Update accordingly in loop
    metrics = {
        "wasserstein_loss": wasserstein_distance,
        "ssim": ssim,
        "total_loss": loss,
        "gradient_loss": grad_val,
        "ncc": ncc_val
    }
    save_iteration_data(working_dir / H5_FILE, iteration, pattern, trace, masked_trace, coefficients, metrics)
    # save_iteration_data(
    #     working_dir / H5_FILE,
    #     iteration,
    #     phase_mask=pattern,
    #     trace=trace,
    #     masked_trace=masked_trace,
    #     polynomial_coeffs=coefficients,
    #     wasserstein_loss=wasserstein_distance,
    #     ssim=ssim,
    #     total_loss=loss
    # )



    
    optimized_coeffs = optimize_coefficients(
        initial_coeffs=error_parameters["initial_scaled_coefficients"],
        target_image=target_spectral_array,
        slm_parameters=slm_parameters,
        slm_init_params = slm_init_params,
        frog=frog,
        frog_parameters=frog_parameters,
        error_parameters=error_parameters,
        h5_path=working_dir / H5_FILE,
        plot_dir=plot_dir,
        working_dir=working_dir
    )

    print("-------------------")
    print("L-BFGS Optimization Complete")
    print("-------------------")
    print("Optimized Coefficients:", optimized_coeffs)
    

    #start error scan

    #make folder in directory that has some pdfs. the pdf will have image of current frog, target frog, error and phase mask with coefficients labeled. generate every 20 iterations and save

    # slm.load_csv(r"C:/Users/lasopr/Downloads/phase_13.csv")

    frog.close_frog()

    






if __name__ == "__main__":
    main()