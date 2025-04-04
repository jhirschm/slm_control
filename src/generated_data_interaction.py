import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from santec_slm import SantecSLM
from frog_class import FROG
from scipy.constants import c



def load_metadata(h5_file):
    with h5py.File(h5_file, 'r') as f:
        return f['metadata/sweep_info'][:]

# def plot_masked_trace(h5_file, sweep_name):
#     with h5py.File(h5_file, 'r') as f:
#         trace = f[f"{sweep_name}/masked_trace"][:]

#     plt.imshow(trace, aspect='auto', origin='lower', cmap='jet')
#     plt.colorbar(label="Intensity")
#     plt.title(f"Masked Trace - {sweep_name}")
#     plt.xlabel("Delay Index")
#     plt.ylabel("Wavelength Index")
#     plt.show()
def plot_masked_trace(h5_file, sweep_name):
    with h5py.File(h5_file, 'r') as f:
        trace = f[f"{sweep_name}/masked_trace"][:]
        frog_attrs = f["frog"].attrs

        scan_range = frog_attrs["scan_range"]
        step_size = frog_attrs["step_size"]

        # Reconstruct real motor positions
        num_steps = trace.shape[1]
        real_positions = np.linspace(scan_range[0], scan_range[1], num_steps)

        # Convert motor positions in mm to time delays in fs
        delay_fs = (real_positions * 1e-3) / c * 1e15  # mm → m → fs
        delay_fs -= np.mean(delay_fs)  # center around 0 fs

        # Wavelength axis (index only)
        wavelengths = np.arange(trace.shape[0])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(delay_fs, wavelengths, trace, shading='auto', cmap='jet')
    plt.xlabel("Time Delay (fs)")
    plt.ylabel("Wavelength Index")
    plt.title(f"Masked Trace - {sweep_name}")
    plt.colorbar(label="Intensity")
    plt.tight_layout()
    plt.show()

def load_phase_mask_to_slm(h5_file, sweep_name, slm_params):
    import pandas as pd
    from pathlib import Path
    import tempfile

    with h5py.File(h5_file, 'r') as f:
        pattern = f[f"{sweep_name}/phase_mask"][:]

    # Save to temporary CSV
    temp_csv = Path(tempfile.gettempdir()) / "temp_slm_pattern.csv"
    df = pd.DataFrame(pattern)
    df.insert(0, "Y/X", range(pattern.shape[0]))
    df.columns = ["Y/X"] + list(range(pattern.shape[1]))
    df.to_csv(temp_csv, index=False)

    slm = SantecSLM(**{k: slm_params[k] for k in ["slm_number", "bitdepth", "wave_um", "rate", "phase_range"]})
    slm.load_csv(str(temp_csv))
    print(f"[SLM] Loaded phase mask from {sweep_name}")
    slm.close()
    return 1

def run_frog_check(frog_params):
    valid_frog_params = FROG.__init__.__code__.co_varnames
    filtered_frog_params = {k: v for k, v in frog_params.items() if k in valid_frog_params}
    filtered_frog_params["averaging"] = int(filtered_frog_params["averaging"])  # ensure correct type
    print(frog_params)
    frog = FROG(**filtered_frog_params)
    trace, real_positions = frog.run(close=False)
    frog.plot(trace, real_positions, wavelength_range=frog_params["wavelength_range"], time_axis=True)
    # trace = trace.squeeze()
    # _, masked_trace = frog.mask_trace(trace, frog_params["wavelength_range"])

    # delay_fs = (real_positions * 1e-3) / c * 1e15
    # delay_fs -= np.mean(delay_fs)  # center around zero

    # fig, ax = plt.subplots(figsize=(10, 6))
    # cax = ax.pcolormesh(delay_fs, frog_params["wavelength_range"], trace, shading='auto', cmap='jet')

    # ax.set_xlabel("Time Delay (fs)")
    # ax.set_ylabel("Wavelength (nm)")
    # ax.set_title("Live Frog Trace")

    # # Top axis showing index
    # ax_top = ax.twiny()
    # ax_top.set_xlim(ax.get_xlim())
    # step_indices = np.arange(len(real_positions))
    # tick_spacing = max(1, len(step_indices) // 10)
    # ax_top.set_xticks(delay_fs[::tick_spacing])
    # ax_top.set_xticklabels(step_indices[::tick_spacing])
    # ax_top.set_xlabel("Sweep Step Index")

    # fig.colorbar(cax, ax=ax, label="Intensity")
    # plt.tight_layout()
    # plt.show()
    frog.close_frog()

def find_closest_sweep(meta, user_coeffs):
    added = np.stack(meta['added_coeffs'])
    dists = np.linalg.norm(added - np.array(user_coeffs), axis=1)
    idx = np.argmin(dists)
    print("Closest match:")
    print(f"  Sweep Number: {meta['sweep_number'][idx]}")
    print(f"  Added Coefficients: {meta['added_coeffs'][idx]}")
    print(f"  Final Coefficients: {meta['final_coeffs'][idx]}")
    return meta['sweep_number'][idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help="Path to H5 file")
    args = parser.parse_args()

    h5_file = args.file
    meta = load_metadata(h5_file)

    with h5py.File(h5_file, 'r') as f:
        slm_params = dict(f['slm'].attrs)
        frog_params = dict(f['frog'].attrs)

    total_sweeps = len(meta)

    while True:
        cmd = input("Enter command (view, load, frog confirm, load and close, search, close): ").strip().lower()

        if cmd == "close":
            print("Exiting...")
            break

        elif cmd in ["view", "load", "load and close"]:
            choice = input(f"Enter 'baseline' or sweep number (0--{total_sweeps-1}): ").strip()
            if choice == "baseline":
                group_name = "baseline"
            else:
                if not choice.isdigit() or not (0 <= int(choice) < total_sweeps):
                    print("Invalid sweep number.")
                    continue
                group_name = f"sweep/{choice}"

            with h5py.File(h5_file, 'r') as f:
                if group_name not in f:
                    print(f"Sweep {group_name} not found.")
                    continue
                added = f[f"{group_name}/added_coefficients"][:] if 'added_coefficients' in f[group_name] else np.zeros(6)
                total = f[f"{group_name}/total_coefficients"][:] if 'total_coefficients' in f[group_name] else added

            print(f"Sweep: {choice if choice != 'baseline' else 'baseline'}")
            print(f"  Added Coefficients: {added}")
            print(f"  Total Coefficients: {total}")

            if cmd == "view":
                plot_masked_trace(h5_file, group_name)

            elif cmd == "load":
                cmd = input("Is RA OFF? (enter yes/no) ")
                if cmd == "yes" or "Yes":
                    load_phase_mask_to_slm(h5_file, group_name, slm_params)
                else:
                    print("Ensure carbide RA is off and try loading again.")


            elif cmd == "load and close":
                load_phase_mask_to_slm(h5_file, group_name, slm_params)
                print("Phase mask loaded. Exiting...")
                break

        elif cmd == "frog confirm":
            run_frog_check(frog_params)

        elif cmd == "search":
            print("Please enter 6 added coefficients (a5 ... a0) separated by space:")
            try:
                vals = input().strip().split()
                if len(vals) != 6:
                    raise ValueError
                user_coeffs = [float(v) for v in vals]
                closest_idx = find_closest_sweep(meta, user_coeffs)
            except ValueError:
                print("Invalid input. Please enter exactly 6 numerical values.")

        else:
            print("Invalid command.")

if __name__ == "__main__":
    main()
