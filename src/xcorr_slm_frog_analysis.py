import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import math
import h5py
from scipy.constants import c


# Step 1: Define the structured dtype
crosscorr_dtype = np.dtype([
    ("x", object),
    ("y", object),
    ("iteration", int),
    ("averaging", "U8"),
    ("filter_flag", int)
])
def plot_frog_traces_by_data_array(h5_path, data_array, vmax=None):
    from scipy.constants import c
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    import math

    # Filter and sort by iteration number (exclude baseline)
    valid_entries = data_array[data_array["iteration"] != -1]
    sorted_indices = np.argsort(valid_entries["iteration"])
    sorted_entries = valid_entries[sorted_indices]
    sweep_nums = np.unique(sorted_entries["iteration"])

    with h5py.File(h5_path, 'r') as f:
        frog_attrs = f["frog"].attrs
        scan_range = frog_attrs["scan_range"]
        step_size = frog_attrs["step_size"]

        num_plots = len(sweep_nums)
        grid_size = math.ceil(math.sqrt(num_plots))
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3.5, grid_size * 2.8))
        axs = axs.flatten()

        for i, sweep_num in enumerate(sweep_nums):
            sweep_path = f"sweep/{sweep_num}"
            if sweep_path not in f:
                print(f"[Warning] {sweep_path} not found in HDF5 file.")
                continue

            trace = f[sweep_path + "/masked_trace"][:]
            num_steps = trace.shape[1]
            real_positions = np.linspace(scan_range[0], scan_range[1], num_steps)
            delay_fs = (real_positions * 1e-3) / c * 1e15
            delay_fs -= np.mean(delay_fs)

            ax = axs[i]
            im = ax.pcolormesh(delay_fs, np.arange(trace.shape[0]), trace,
                               shading='auto', cmap='jet', vmax=vmax)
            ax.set_title(f"Sweep {sweep_num}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused subplots
        for j in range(num_plots, len(axs)):
            axs[j].axis("off")

        fig.suptitle("Masked FROG Traces (Ordered by Iteration)", fontsize=14)
        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])
        plt.show()

def load_sweep_metadata(h5_path):
    with h5py.File(h5_path, 'r') as f:
        sweep_info = f["metadata/sweep_info"][:]
        # Convert to dict: {sweep_number: added_coeffs}
        return {int(entry["sweep_number"]): entry["added_coeffs"] for entry in sweep_info}
    

# Step 2: Function to parse and extract from one file
def parse_cross_correlation_file(file_path: str, file_name: str):
    data = pd.read_csv(file_path, skiprows=22, sep="\t", names=["Delay [ps]", "Intensity [arb.u.]"])

    # Regex that matches either 'baseline' or 'iter###' at the beginning
    match = re.match(r"(baseline|iter(\d+))_avg-(\w+)_res-\w+_filter-(\w+)", file_name)
    if not match:
        return None  # Skip files with non-matching names

    # Determine iteration value
    if match.group(1) == "baseline":
        iteration = -1
    else:
        iteration = int(match.group(2))

    averaging = f"avg-{match.group(3)}"
    filter_str = match.group(4).lower()
    filter_flag = 1 if filter_str == "yes" else 0

    x = data["Delay [ps]"].values
    y = data["Intensity [arb.u.]"].values

    return (x, y, iteration, averaging, filter_flag)

# Step 3: Load all matching files into structured array
def load_all_cross_correlation_files(directory):
    entries = []

    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if (
            os.path.isfile(file_path)
            and ("iter" in file_name or "baseline" in file_name)
            and "avg-" in file_name
            and "filter-" in file_name
        ):
            parsed = parse_cross_correlation_file(file_path, file_name)
            if parsed:
                entries.append(parsed)

    result_array = np.array(entries, dtype=crosscorr_dtype)
    return result_array

# def plot_cross_correlation_grid(data_array, averaging="avg-low", filter_flag=1, xlim_range=(0, 40)):
#     # Filter for desired averaging and filter_flag
#     filtered_data = data_array[
#         (data_array["averaging"] == averaging) &
#         (data_array["filter_flag"] == filter_flag)
#     ]

#     # Get baseline
#     baseline_entry = filtered_data[filtered_data["iteration"] == -1]
#     if len(baseline_entry) == 0:
#         print(f"No baseline found for {averaging}, filter={filter_flag}")
#         return
#     baseline = baseline_entry[0]

#     # Get iteration entries (excluding baseline)
#     iteration_entries = filtered_data[filtered_data["iteration"] != -1]
#     num_plots = len(iteration_entries)

#     if num_plots == 0:
#         print(f"No iteration entries found for {averaging}, filter={filter_flag}")
#         return

#     # Determine square grid size
#     grid_size = math.ceil(math.sqrt(num_plots))
#     fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3, grid_size * 2.5), sharex=True, sharey=True)
#     axs = axs.flatten()

#     for i, entry in enumerate(iteration_entries):
#         ax = axs[i]
#         ax.plot(baseline["x"], baseline["y"], label="Baseline", linestyle="--", alpha=0.6)
#         ax.plot(entry["x"], entry["y"], label=f"Iter {entry['iteration']}")
#         ax.set_title(f"Iter {entry['iteration']}")
#         ax.set_xlim(*xlim_range)
#         ax.grid(True)

#     # Hide unused subplots
#     for j in range(num_plots, len(axs)):
#         axs[j].axis("off")

#     # Global labels and formatting
#     axs[0].legend()
#     fig.text(0.5, 0.04, 'Delay [ps]', ha='center', fontsize=12)
#     fig.text(0.04, 0.5, 'Intensity [arb.u.]', va='center', rotation='vertical', fontsize=12)
#     fig.suptitle(f"Cross-Correlation ({averaging}, filter={filter_flag}) vs Baseline", fontsize=14)
#     plt.tight_layout(rect=[0.12, 0.12, 1, 0.95])
#     plt.show()
def plot_cross_correlation_grid(data_array, averaging="avg-low", filter_flag=1, xlim_range=(0, 40), coeff_label_dict=None):
    

    filtered_data = data_array[
        (data_array["averaging"] == averaging) &
        (data_array["filter_flag"] == filter_flag)
    ]

    baseline_entry = filtered_data[filtered_data["iteration"] == -1]
    if len(baseline_entry) == 0:
        print(f"No baseline found for {averaging}, filter={filter_flag}")
        return
    baseline = baseline_entry[0]

    iteration_entries = filtered_data[filtered_data["iteration"] != -1]
    num_plots = len(iteration_entries)

    if num_plots == 0:
        print(f"No iteration entries found for {averaging}, filter={filter_flag}")
        return

    grid_size = math.ceil(math.sqrt(num_plots))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3, grid_size * 2.5), sharex=True, sharey=True)
    axs = axs.flatten()
    iteration_entries = np.sort(iteration_entries, order="iteration")

    for i, entry in enumerate(iteration_entries):
        ax = axs[i]
        ax.plot(baseline["x"], baseline["y"], label="Baseline", linestyle="--", alpha=0.6)
        ax.plot(entry["x"], entry["y"], label=f"Iter {entry['iteration']}")
        ax.set_xlim(*xlim_range)
        ax.set_title(f"Iter {entry['iteration']}")
        ax.grid(True)

        if coeff_label_dict and entry["iteration"] in coeff_label_dict:
            coeffs = coeff_label_dict[entry["iteration"]]
            # Format 3 per line, split into two lines
            line1 = ", ".join(f"{c:.1e}" for c in coeffs[:3])
            line2 = ", ".join(f"{c:.1e}" for c in coeffs[3:])
            subtitle = f"({line1},\n {line2})"
            ax.text(0.5, -0.3, subtitle, ha='center', va='top', fontsize=8, transform=ax.transAxes)

    for j in range(num_plots, len(axs)):
        axs[j].axis("off")

    axs[0].legend()
    fig.text(0.5, 0.04, 'Delay [ps]', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Intensity [arb.u.]', va='center', rotation='vertical', fontsize=12)
    fig.suptitle(f"Cross-Correlation ({averaging}, filter={filter_flag}) vs Baseline", fontsize=14)
    plt.tight_layout(rect=[0.12, 0.12, 1, 0.95])
    plt.show()

directory_path = "/Users/jhirschm/Downloads/SLM_Tests_04-04-2025"
data_array = load_all_cross_correlation_files(directory_path)
h5_file_path = "/Users/jhirschm/Downloads/SLM_Tests_04-04-2025/runtime_data.h5"
coeff_labels = load_sweep_metadata(h5_file_path)
plot_cross_correlation_grid(data_array, averaging="avg-med", filter_flag=0, coeff_label_dict=coeff_labels)
plot_cross_correlation_grid(data_array, averaging="avg-low", filter_flag=1, coeff_label_dict=coeff_labels)
plot_frog_traces_by_data_array(h5_file_path, data_array)



# Low averaging, filtered (your original case)
# plot_cross_correlation_grid(data_array, averaging="avg-low", filter_flag=1)

# Medium averaging, unfiltered
# plot_cross_correlation_grid(data_array, averaging="avg-med", filter_flag=0)

# Accessing individual entries
# print(data_array[0]["iteration"])
# print(data_array[0]["averaging"])
# print(data_array[0]["x"][:5])

# all_iterations = np.unique(data_array["iteration"])
# print("Available iteration numbers:", all_iterations)

# # 2. Choose one to plot (replace this with any number from the list)
# target_iter = 227  # <--- you can change this!

# # 3. Search and plot
# match = data_array[
#     (data_array["iteration"] == target_iter) &
#     (data_array["filter_flag"] == 1) &
#     (data_array["averaging"] == "avg-low")
# ]
# if len(match) > 0:
#     entry = match[0]
#     plt.figure(figsize=(10, 5))
#     plt.plot(entry["x"], entry["y"])
#     plt.title(f"Iteration {entry['iteration']} | {entry['averaging']} | Filtered: {bool(entry['filter_flag'])}")
#     plt.xlabel("Delay [ps]")
#     plt.ylabel("Intensity [arb.u.]")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
# else:
#     print(f"No entry found for iteration {target_iter}")

# # Define the list of iteration numbers you want to plot
# target_iters = [-1, 178, 210, 220, 375]  # Replace with your actual values

# # Create a figure
# plt.figure(figsize=(10, 6))

# # Loop through the targets and plot if matching entry exists
# for target in target_iters:
#     match = data_array[
#         (data_array["iteration"] == target) &
#         (data_array["filter_flag"] == 1) &
#         (data_array["averaging"] == "avg-low")
#     ]
    
#     if len(match) > 0:
#         entry = match[0]
#         label = "Baseline" if target == -1 else f"Iter {target}"
#         plt.plot(entry["x"], entry["y"], label=f"Iter {target}")
#     else:
#         print(f"No matching entry for iteration {target}")

# # Plot styling
# plt.title("Cross-Correlation Traces for Selected Iterations")
# plt.xlabel("Delay [ps]")
# plt.xlim(0,60)
# plt.ylabel("Intensity [arb.u.]")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Filter data for low averaging and filtering
# filtered_data = data_array[
#     (data_array["averaging"] == "avg-low") &
#     (data_array["filter_flag"] == 1)
# ]

# # Get baseline entry
# baseline_entry = filtered_data[filtered_data["iteration"] == -1]
# if len(baseline_entry) == 0:
#     raise ValueError("Baseline entry not found (iteration == -1)")
# baseline = baseline_entry[0]

# # Get all valid iterations (excluding baseline)
# iteration_entries = filtered_data[filtered_data["iteration"] != -1]

# # Total number of iterations to plot
# num_plots = len(iteration_entries)
# print(f"Found {num_plots} valid iterations (excluding baseline)")

# # Calculate grid size (e.g., 5x5 for 25, 6x6 for up to 36, etc.)
# grid_size = math.ceil(math.sqrt(num_plots))
# fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3, grid_size * 2.5), sharex=True, sharey=True)
# axs = axs.flatten()

# # Plot each iteration with baseline overlay
# for i, entry in enumerate(iteration_entries):
#     ax = axs[i]
#     ax.plot(baseline["x"], baseline["y"], label="Baseline", linestyle="--", alpha=0.6)
#     ax.plot(entry["x"], entry["y"], label=f"Iter {entry['iteration']}")
#     ax.set_title(f"Iter {entry['iteration']}")
#     ax.grid(True)
#     ax.set_xlim(0, 40)


# # Hide any unused subplots
# for j in range(num_plots, len(axs)):
#     axs[j].axis("off")

# # Add legend to the first subplot only
# axs[0].legend()

# # Global axis labels and title
# fig.text(0.5, 0.04, 'Delay [ps]', ha='center', fontsize=12)
# fig.text(0.04, 0.5, 'Intensity [arb.u.]', va='center', rotation='vertical', fontsize=12)
# fig.suptitle("Cross-Correlation (avg-low, filter-yes) vs Baseline", fontsize=14)

# # Adjust layout spacing
# plt.tight_layout(rect=[0.12, 0.12, 1, 0.95])
# plt.show()