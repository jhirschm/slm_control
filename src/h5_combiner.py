import h5py

def merge_h5_sweeps(old_h5_path, new_h5_path):
    with h5py.File(old_h5_path, 'r') as old_f, h5py.File(new_h5_path, 'a') as new_f:
        if "sweep" not in old_f:
            raise ValueError("Old file does not contain 'sweep' group.")
        
        old_sweeps = old_f["sweep"]
        new_sweeps = new_f.require_group("sweep")

        # Get list of existing sweep indices in new file
        existing_indices = set(int(k) for k in new_sweeps.keys() if k.isdigit())

        merged_count = 0
        for sweep_name in old_sweeps:
            if not sweep_name.isdigit():
                continue
            idx = int(sweep_name)
            if idx in existing_indices:
                print(f"[SKIP] Sweep {idx} already exists.")
                continue

            # Create the group in new file
            old_group = old_sweeps[sweep_name]
            new_group = new_sweeps.create_group(sweep_name)

            for key in old_group:
                new_group.create_dataset(key, data=old_group[key][:])
            
            print(f"[MERGED] Sweep {idx} copied.")
            merged_count += 1

    print(f"\n✅ Merge complete. {merged_count} sweep(s) copied from old to new.")

# Example usage
merge_h5_sweeps(
    old_h5_path="C:/FROG_SLM_DataGen/slm_sweep_output_third/runtime_data.h5",
    new_h5_path="C:/FROG_SLM_DataGen/slm_sweep_output_third_20250407_125152/runtime_data.h5"
)
