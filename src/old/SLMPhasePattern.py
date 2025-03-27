import os
import h5py
import json
import yaml
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

class SLMPhasePattern:
    def __init__(self, width=1920, height=1200, poly_coeffs=None, variation_scale=0.001):
        self.width = width
        self.height = height
        self.poly_coeffs = np.array(poly_coeffs if poly_coeffs else [0, 0, 0, 0, 0, 0])
        self.variation_scale = variation_scale
    
    def generate(self):
        x = np.arange(self.width)
        x_norm = x - self.width // 2
        phase = np.polyval(self.poly_coeffs, x_norm)
        phase_scaled = (phase / (2 * np.pi)) * 1023
        phase_wrapped = np.mod(phase_scaled, 1024)
        return phase_wrapped.astype(int)
    
    def save_csv(self, filename="slm_phase.csv"):
        phase_values = self.generate()
        df = pd.DataFrame(np.tile(phase_values, (self.height, 1)), dtype=int)
        df.to_csv(filename, index=False)
        return filename