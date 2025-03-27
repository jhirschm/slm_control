import os
import h5py
import json
import yaml
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import minimize
from frog_class import FROG
from santec import SantecSLM
from SLMPhasePattern import SLMPhasePattern

class OptimizationRun:
    def __init__(self, base_dir="runs", label="test_run", target_file=None, init_poly=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, f"{label}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.h5_file = os.path.join(self.run_dir, "data.h5")
        self.config_file = os.path.join(self.run_dir, "config.json")
        self.target_file = target_file
        self.init_poly = init_poly
        
        # Save config
        config = {
            "label": label,
            "timestamp": timestamp,
            "initial_polynomial": init_poly,
            "target_file": target_file
        }
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=4)
        
        # Load target spectrogram
        if target_file:
            self.target_spectrogram = np.load(target_file)
            target_save_path = os.path.join(self.run_dir, "target_spectrogram.npy")
            np.save(target_save_path, self.target_spectrogram)
        else:
            raise ValueError("Target spectrogram file must be provided.")
    
    def save_iteration(self, iteration, poly_coeffs, slm_pattern, frog_trace, error):
        with h5py.File(self.h5_file, "a") as f:
            grp = f.create_group(f"iteration_{iteration}")
            grp.create_dataset("polynomial", data=poly_coeffs)
            grp.create_dataset("slm_pattern", data=slm_pattern)
            grp.create_dataset("frog_trace", data=frog_trace)
            grp.create_dataset("error", data=error)
    
    def save_description(self, description):
        with open(os.path.join(self.run_dir, "description.txt"), "w") as f:
            f.write(description)

class ErrorMinimization:
    def __init__(self, slm, frog, target_file, max_iters=10, init_poly=None, variation_scale=0.001):
        self.slm = slm
        self.frog = frog
        self.max_iters = max_iters
        self.poly_coeffs = np.array(init_poly if init_poly else [0, 0, 0, 0, 0, 0])
        self.variation_scale = variation_scale
        self.run = OptimizationRun(init_poly=self.poly_coeffs.tolist(), target_file=target_file)
        self.iteration = 0  # Track optimization iterations
    
    def compute_error(self, frog_trace):
        return np.linalg.norm(frog_trace - self.run.target_spectrogram)
    
    def objective_function(self, poly_coeffs):
        print(f"Iteration {self.iteration + 1} - Testing Polynomial Coefficients: {poly_coeffs}")
        slm_pattern = SLMPhasePattern(poly_coeffs=poly_coeffs).generate()
        csv_file = SLMPhasePattern(poly_coeffs=poly_coeffs).save_csv()
        self.slm.load_csv(csv_file)
        time.sleep(5)  # Wait for upload
        frog_trace, _ = self.frog.run()
        error = self.compute_error(np.mean(frog_trace, axis=2))
        self.run.save_iteration(self.iteration, poly_coeffs, slm_pattern, frog_trace, error)
        print(f"Iteration {self.iteration + 1} - Error: {error}")
        self.iteration += 1  # Increment iteration counter
        return error
    
    def optimize(self):
        print("Starting optimization...")
        result = minimize(self.objective_function, self.poly_coeffs, method='L-BFGS-B', options={'maxiter': self.max_iters})
        self.poly_coeffs = result.x
        print("Optimization Complete")
        print("Final Polynomial Coefficients:", self.poly_coeffs)