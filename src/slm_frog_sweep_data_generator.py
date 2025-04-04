import os
import time
import h5py
import yaml
import numpy as np
from pathlib import Path
import itertools
from santec_slm import SantecSLM
from frog_class import FROG
import pandas as pd
import inspect

PARAMS_FILE = "params.yaml"
H5_FILE = "runtime_data.h5"

# Define structured dtype for metadata
sweep_dtype = np.dtype([
    ('sweep_number', 'i4'),
    ('added_coeffs', 'f8', (6,)),
    ('final_coeffs', 'f8', (6,))
])

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

def setup_directory(base_dir):
    dir_path = Path(base_dir).resolve()
    if dir_path.exists():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dir_path = dir_path.parent / f"{dir_path.name}_{timestamp}"
    dir_path.mkdir(parents=True)
    return dir_path

def save_params_yaml(path, frog_params, slm_params, shaping_params):
    config = {
        "frog_parameters": frog_params,
        "slm_parameters": slm_params,
        "shaping_parameters": shaping_params
    }
    with open(path / PARAMS_FILE, 'w') as f:
        yaml.dump(config, f)

def create_h5_file(path, frog_params, slm_params, shaping_params):
    h5_path = path / H5_FILE
    with h5py.File(h5_path, 'w') as f:
        f.attrs['created'] = time.ctime()

        frog_group = f.create_group("frog")
        for k, v in frog_params.items():
            frog_group.attrs[k] = v

        slm_group = f.create_group("slm")
        for k, v in slm_params.items():
            slm_group.attrs[k] = v

        shaping_group = f.create_group("shaping")
        for k, v in shaping_params.items():
            shaping_group.attrs[k] = str(v)  # stringify for consistency

    return h5_path

def generate_values(start, stop, steps):
    if steps < 2:
        print(f"[WARNING] Only one point ({start}) will be used for range ({start}, {stop}) with steps={steps}.")
        return [start]
    return np.linspace(start, stop, steps).tolist()

def expand_coefficients(bounds_dict, total_coeffs=6):
    coeff_options = []
    for i in range(total_coeffs):
        if i in bounds_dict:
            ranges = bounds_dict[i]
            values = []
            for start, stop, steps in ranges:
                values.extend(generate_values(start, stop, steps))
        else:
            values = [0.0]
        coeff_options.append(values)
    return list(itertools.product(*coeff_options))


def run_shaping(base_dir, shaping_params, frog_params, slm_params):
    workdir = setup_directory(base_dir)
    save_params_yaml(workdir, frog_params, slm_params, shaping_params)
    h5_file = create_h5_file(workdir, frog_params, slm_params, shaping_params)

    valid_frog_params = inspect.signature(FROG.__init__).parameters
    filtered_frog_params = {k: v for k, v in frog_params.items() if k in valid_frog_params}
    frog = FROG(**filtered_frog_params)

    baseline_coeffs = shaping_params["baseline"]
    phase = phase_mask(baseline_coeffs, slm_params["effective_scale"], slm_params["width"])
    pattern = generate_phase_pattern(slm_params["width"], slm_params["height"], phase, workdir / "baseline_temp.csv")

    slm = SantecSLM(**{k: slm_params[k] for k in ["slm_number", "bitdepth", "wave_um", "rate", "phase_range"]})
    slm.load_csv(str(workdir / "baseline_temp.csv"))
    time.sleep(1)

    trace, real_positions = frog.run(close=False)
    trace = trace.squeeze()
    _, masked_trace = frog.mask_trace(trace, frog_params["wavelength_range"])

    with h5py.File(h5_file, 'a') as f:
        g = f.create_group("baseline")
        g.create_dataset("coefficients", data=baseline_coeffs)
        g.create_dataset("phase_mask", data=pattern)
        g.create_dataset("trace", data=trace)
        g.create_dataset("masked_trace", data=masked_trace)

    print("[Baseline] Saved baseline FROG and SLM data.")
    slm.close()

    sweep_coeffs = expand_coefficients(shaping_params["bounds"])
    total_sweeps = len(sweep_coeffs)
    sweep_metadata = np.zeros(total_sweeps, dtype=sweep_dtype)
    print(sweep_coeffs)
    print(sweep_metadata)
    for idx, added in enumerate(sweep_coeffs):
        print(f"[{idx + 1}/{total_sweeps}] Generating phase mask and taking measurement...")

        final_coeffs = [a + b for a, b in zip(shaping_params["baseline"], added)]
        phase = phase_mask(final_coeffs, slm_params["effective_scale"], slm_params["width"])
        pattern = generate_phase_pattern(slm_params["width"], slm_params["height"], phase, workdir / "temp.csv")

        slm = SantecSLM(**{k: slm_params[k] for k in ["slm_number", "bitdepth", "wave_um", "rate", "phase_range"]})
        slm.load_csv(str(workdir / "temp.csv"))
        time.sleep(1)

        trace, real_positions = frog.run(close=False)
        trace = trace.squeeze()
        _, masked_trace = frog.mask_trace(trace, frog_params["wavelength_range"])

        with h5py.File(h5_file, 'a') as f:
            g = f.create_group(f"sweep/{idx}")
            g.create_dataset("added_coefficients", data=added)
            g.create_dataset("total_coefficients", data=final_coeffs)
            g.create_dataset("phase_mask", data=pattern)
            g.create_dataset("trace", data=trace)
            g.create_dataset("masked_trace", data=masked_trace)

        sweep_metadata[idx] = (idx, np.array(added), np.array(final_coeffs))

        print(f"[{idx + 1}/{total_sweeps}] Sweep Step Complete.\n")
        slm.close()

    frog.close_frog()

    # Save sweep metadata
    with h5py.File(h5_file, 'a') as f:
        meta_group = f.require_group("metadata")
        meta_group.create_dataset("sweep_info", data=sweep_metadata)

# Example usage:
if __name__ == "__main__":
    shaping_params = {
        "baseline": [0, 3.25305999722143e-17, -2.999964343100393e-8, 2.4090017016278843e-4, 0, 0],
        "bounds": {
            0: [(0, 0, 1)],
            1: [(0, 0, 1)],
            2: [(0,0,1), (1e-8, 10e-8, 10), (-10e-8, -1e-8, 10)],
            3: [(0,0,1), (1e-6, 10e-6, 10), (-10e-6, -1e-6, 10)],
            4: [(0, 0, 1)],
            5: [(0, 0, 1)]
        }
    }

    frog_params = {
        "integration_time": 0.1,
        "averaging": 1,
        "central_motor_position": -.27,
        "scan_range": (-0.08, 0.08),
        "step_size": 0.001,
        "wavelength_range": (480, 560)
    }
    slm_params = {
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

    run_shaping("C:\\FROG_SLM_DataGen\\slm_sweep_output_first", shaping_params, frog_params, slm_params)
