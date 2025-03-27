import numpy as np
import matplotlib.pyplot as plt
from frog_class import FROG  # Import the FROG class
from slm_controller import SLMController  # Import the SLM controller class
import time

def main():
    """
    Main experiment function:
    - Initialize SLM and load a phase profile
    - Run a FROG scan
    - Close SLM and plot results
    """

    #  **Step 1: Initialize SLM**
    print("Initializing SLM...")
    slm = SLMController()

    # slm_info = slm.get_parameters()
    # print(slm_info)
    # Set uniform grayscale
    slm.set_uniform_grayscale(1023)

    # **Step 2: Load a phase profile**
    phase_profile_path = "C:\\Users\\lasopr\\Downloads\\phase_13.csv"
    success = slm.upload_grayscale_csv(phase_profile_path)
    if not success:
        raise RuntimeError("Failed to upload phase profile to SLM!")

    # **Step 3: Initialize FROG**
    print("Initializing FROG system...")
    frog = FROG(integration_time=0.5, averaging=1, central_motor_position=0.165,
                scan_range=(-0.05, 0.05), step_size=0.001)

    # **Step 4: Run the FROG scan while keeping SLM active**
    print("Running FROG scan...")
    trace, real_positions = frog.run()

    # **Step 5: Close SLM after FROG scan is complete**
    print("Closing SLM...")
    slm.close_slm()

    # **Step 6: Plot the FROG results**
    print("Plotting results...")
    frog.plot(trace, real_positions, wavelength_range=(490, 560), time_axis=False)
    frog.plot(trace, real_positions, wavelength_range=(490, 560), time_axis=True)

if __name__ == "__main__":
    main()
