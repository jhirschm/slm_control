import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
import datetime
from santec_slm import SantecSLM
from frog_class import FROG

# === Helper Functions ===
def generate_polynomial(num_pixels, coeffs, shift=0):
    x = np.arange(num_pixels) - num_pixels // 2 + shift
    return np.polyval(coeffs, x)

def generate_sinc_1d(size, kind='sinc', extent=None, a=np.pi, b=50):
    x = np.arange(size) - size // 2
    y = np.sinc(x / b)
    if kind == 'sinc2':
        y = y ** 2
    y *= a
    if extent:
        y[np.abs(x) > extent // 2] = 0
    return y

def generate_sinc_2d(width, height, kind='sinc', extent_x=None, extent_y=None, a=np.pi, b=(50, 50)):
    x = np.arange(width) - width // 2
    y = np.arange(height) - height // 2
    xx, yy = np.meshgrid(x, y)
    bx, by = b if isinstance(b, tuple) else (b, b)
    zx = np.sinc(xx / bx)
    zy = np.sinc(yy / by)
    if kind == 'sinc2':
        zx = zx ** 2
        zy = zy ** 2
    z = a * zx * zy
    if extent_x:
        z[:, np.abs(x) > extent_x // 2] = 0
    if extent_y:
        z[np.abs(y) > extent_y // 2, :] = 0
    return z

def save_mask_and_config(mask, directory, identifier, config_dict):
    csv_path = Path(directory) / f"{identifier}.csv"
    txt_path = Path(directory) / f"{identifier}_desc.txt"
    np.savetxt(csv_path, mask.astype(int), delimiter=",", fmt="%d")
    with open(txt_path, 'w') as f:
        for key, value in config_dict.items():
            f.write(f"{key}: {value}\n")
    print(f"[Saved] Mask: {csv_path}, Config: {txt_path}")

def apply_blank_mask(mask, blank_mask):
    return mask * blank_mask

# === Blanking Window ===
class BlankingWindow:
    def __init__(self, parent, mask_shape, blank_mask):
        self.top = tk.Toplevel(parent)
        self.top.title("Column Blanking")
        self.blank_mask = blank_mask

        label = tk.Label(self.top, text="Enter column range(s) to blank (e.g. 100-120,300-305):")
        label.pack()
        self.entry = tk.Entry(self.top, width=50)
        self.entry.pack()

        btn_apply = tk.Button(self.top, text="Apply", command=self.apply_blanking)
        btn_apply.pack(pady=2)

        reset_btn = tk.Button(self.top, text="Reset All", command=self.reset_all)
        reset_btn.pack(pady=5)

    def apply_blanking(self):
        ranges = self.entry.get().split(',')
        for r in ranges:
            if '-' in r:
                try:
                    start, end = map(int, r.strip().split('-'))
                    self.blank_mask[:, start:end+1] = 0
                except ValueError:
                    continue
            else:
                try:
                    idx = int(r.strip())
                    self.blank_mask[:, idx] = 0
                except ValueError:
                    continue

    def reset_all(self):
        self.blank_mask[:, :] = 1

# === Main GUI ===
class SLMGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SLM Mask Generator - Experiment Mode")
        self.working_dir = None
        self.upload_counter = 1

        self.slm_params = {
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
        self.blank_mask = np.ones((self.slm_params["height"], self.slm_params["width"]))
        self.baseline_coeffs = [0, 3.25305999722143e-17, -2.999964343100393e-8, 2.4090017016278843e-4, 0, 0]

        # State
        self.use_poly = tk.BooleanVar()
        self.use_sinc2d = tk.BooleanVar()
        self.use_sinc1d_col = tk.BooleanVar()
        self.use_sinc1d_row = tk.BooleanVar()
        self.baseline_on = tk.BooleanVar(value=True)
        self.view_baseline = tk.BooleanVar(value=True)

        self.poly_coeffs = tk.StringVar(value="0 0 0 0 0 0")
        self.poly_shift = tk.IntVar(value=0)

        self.extent_x = tk.IntVar(value=1000)
        self.extent_y = tk.IntVar(value=1000)
        self.sinc_mode = tk.StringVar(value='sinc2')

        self.sinc2d_a = tk.DoubleVar(value=2.0)
        self.sinc2d_bx = tk.DoubleVar(value=50)
        self.sinc2d_by = tk.DoubleVar(value=50)
        self.sinc1d_col_a = tk.DoubleVar(value=2.0)
        self.sinc1d_col_b = tk.DoubleVar(value=50)
        self.sinc1d_row_a = tk.DoubleVar(value=2.0)
        self.sinc1d_row_b = tk.DoubleVar(value=50)

        self.setup_ui()
        self.mask = np.zeros((self.slm_params["height"], self.slm_params["width"]))

    def setup_ui(self):
        self.frog_settings = {
            "averaging": tk.IntVar(value=1),
            "central_motor_position": tk.DoubleVar(value=-0.27),
            "integration_time": tk.DoubleVar(value=0.1),
            "scan_range_start": tk.DoubleVar(value=-0.06),
            "scan_range_end": tk.DoubleVar(value=0.06),
            "step_size": tk.DoubleVar(value=0.001),
            "wavelength_start": tk.IntVar(value=480),
            "wavelength_end": tk.IntVar(value=560)
        }

        ttk.Button(self.root, text="Load Saved Mask", command=self.load_saved_mask).pack(pady=4)
        frm = ttk.Frame(self.root)
        frm.pack(padx=10, pady=5)

        ttk.Checkbutton(frm, text="Polynomial", variable=self.use_poly).grid(row=0, column=0)
        ttk.Entry(frm, textvariable=self.poly_coeffs, width=30).grid(row=0, column=1)
        ttk.Label(frm, text="Shift").grid(row=0, column=2)
        ttk.Entry(frm, textvariable=self.poly_shift, width=6).grid(row=0, column=3)

        ttk.Checkbutton(frm, text="2D Sinc", variable=self.use_sinc2d).grid(row=1, column=0)
        ttk.Checkbutton(frm, text="1D Sinc Columns", variable=self.use_sinc1d_col).grid(row=1, column=1)
        ttk.Checkbutton(frm, text="1D Sinc Rows", variable=self.use_sinc1d_row).grid(row=1, column=2)

        ttk.Label(frm, text="Extent X").grid(row=2, column=0)
        ttk.Entry(frm, textvariable=self.extent_x, width=6).grid(row=2, column=1)
        ttk.Label(frm, text="Extent Y").grid(row=2, column=2)
        ttk.Entry(frm, textvariable=self.extent_y, width=6).grid(row=2, column=3)

        sinc_frame = ttk.LabelFrame(frm, text="Sinc Parameters")
        sinc_frame.grid(row=3, column=0, columnspan=6, pady=5, sticky="ew")
        for col in range(6):
            sinc_frame.columnconfigure(col, weight=1)

        ttk.Label(sinc_frame, text="2D Sinc Amp").grid(row=0, column=0)
        ttk.Entry(sinc_frame, textvariable=self.sinc2d_a, width=6).grid(row=1, column=0)
        ttk.Label(sinc_frame, text="2D Sinc X Scale").grid(row=0, column=1)
        ttk.Entry(sinc_frame, textvariable=self.sinc2d_bx, width=6).grid(row=1, column=1)
        ttk.Label(sinc_frame, text="2D Sinc Y Scale").grid(row=0, column=2)
        ttk.Entry(sinc_frame, textvariable=self.sinc2d_by, width=6).grid(row=1, column=2)

        ttk.Label(sinc_frame, text="1D Sinc Col Amp").grid(row=0, column=3)
        ttk.Entry(sinc_frame, textvariable=self.sinc1d_col_a, width=6).grid(row=1, column=3)
        ttk.Label(sinc_frame, text="1D Sinc Col Scale").grid(row=0, column=4)
        ttk.Entry(sinc_frame, textvariable=self.sinc1d_col_b, width=6).grid(row=1, column=4)

        ttk.Label(sinc_frame, text="1D Sinc Row Amp").grid(row=0, column=5)
        ttk.Entry(sinc_frame, textvariable=self.sinc1d_row_a, width=6).grid(row=1, column=5)
        ttk.Label(sinc_frame, text="1D Sinc Row Scale").grid(row=0, column=6)
        ttk.Entry(sinc_frame, textvariable=self.sinc1d_row_b, width=6).grid(row=1, column=6)

        ttk.Label(frm, text="Sinc Type").grid(row=4, column=0)
        ttk.Combobox(frm, textvariable=self.sinc_mode, values=['sinc', 'sinc2'], width=6).grid(row=4, column=1)

        ttk.Checkbutton(frm, text="Add Baseline", variable=self.baseline_on).grid(row=5, column=0)
        ttk.Checkbutton(frm, text="View Baseline", variable=self.view_baseline).grid(row=5, column=1)

        ttk.Button(frm, text="Show Wrapped Phase", command=self.display_wrapped_mask).grid(row=5, column=2)

        ttk.Button(frm, text="Update Mask", command=self.update_mask).grid(row=6, column=0)
        ttk.Button(frm, text="Load to SLM", command=self.load_to_slm).grid(row=6, column=1)
        ttk.Button(frm, text="Run FROG", command=self.run_frog).grid(row=6, column=2)
        ttk.Button(frm, text="Edit FROG Settings", command=self.edit_frog_settings).grid(row=6, column=3)
        ttk.Button(frm, text="Blanking", command=self.open_blanking).grid(row=6, column=4)

        self.upload_label = ttk.Label(frm, text="Upload: None")
        self.upload_label.grid(row=7, column=0, columnspan=3, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        self.cbar = None
    def load_saved_mask(self):
        file_path = filedialog.askopenfilename(title="Load saved mask CSV", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        self.mask = np.loadtxt(file_path, delimiter=",")

        config_path = Path(file_path).with_name(Path(file_path).stem + "_desc.txt")
        if config_path.exists():
            with open(config_path, "r") as f:
                lines = f.readlines()
            config = {}
            for line in lines:
                if ":" in line:
                    k, v = line.strip().split(":", 1)
                    config[k.strip()] = v.strip()

            self.poly_coeffs.set(config.get("poly", "0 0 0 0 0 0"))
            self.use_poly.set(config.get("use_poly", "True") == "True")
            self.baseline_on.set(config.get("baseline", "True") == "True")
            self.use_sinc2d.set(config.get("use_sinc2d", "False") == "True")
            self.use_sinc1d_col.set(config.get("use_sinc1d_col", "False") == "True")
            self.use_sinc1d_row.set(config.get("use_sinc1d_row", "False") == "True")
            self.sinc_mode.set(config.get("sinc_mode", "sinc2"))
            self.extent_x.set(int(config.get("extent_x", 1000)))
            self.extent_y.set(int(config.get("extent_y", 1000)))

        self.update_mask()

    def update_mask(self):
        height, width = self.slm_params["height"], self.slm_params["width"]
        total = np.zeros((height, width))

        base_coeffs = np.array(self.baseline_coeffs)
        poly_coeffs = list(map(float, self.poly_coeffs.get().split())) if self.use_poly.get() else []

        if self.baseline_on.get() or self.use_poly.get():
            padded_poly = np.pad(poly_coeffs, (len(base_coeffs) - len(poly_coeffs), 0)) if poly_coeffs else np.zeros_like(base_coeffs)
            full_coeffs = base_coeffs + padded_poly if self.baseline_on.get() else padded_poly
        else:
            full_coeffs = np.zeros_like(base_coeffs)

        poly = generate_polynomial(width, full_coeffs, self.poly_shift.get())
        total += np.tile(poly, (height, 1))

        if self.use_sinc2d.get():
            total += generate_sinc_2d(width, height, kind=self.sinc_mode.get(), extent_x=self.extent_x.get(), extent_y=self.extent_y.get(), a=self.sinc2d_a.get()*np.pi, b=(self.sinc2d_bx.get(), self.sinc2d_by.get()))

        if self.use_sinc1d_col.get():
            cwave = generate_sinc_1d(width, kind=self.sinc_mode.get(), extent=self.extent_x.get(), a=self.sinc1d_col_a.get()*np.pi, b=self.sinc1d_col_b.get())
            total += np.tile(cwave, (height, 1))

        if self.use_sinc1d_row.get():
            rwave = generate_sinc_1d(height, kind=self.sinc_mode.get(), extent=self.extent_y.get(), a=self.sinc1d_row_a.get()*np.pi, b=self.sinc1d_row_b.get())
            total += np.tile(rwave[:, None], (1, width))

        view_mask = total.copy()
        if self.baseline_on.get() and not self.view_baseline.get():
            base_only = np.polyval(base_coeffs, np.arange(width) - width // 2)
            view_mask -= np.tile(base_only, (height, 1))

        masked = apply_blank_mask(view_mask, self.blank_mask)
        self.mask = masked

        if self.cbar:
            self.cbar.remove()
            self.cbar = None

        self.ax.clear()
        im = self.ax.imshow(self.mask, cmap='jet', aspect='auto')
        self.cbar = self.fig.colorbar(im, ax=self.ax, orientation='vertical')

        self.ax.set_title("Phase Mask")
        self.canvas.draw()

    def display_wrapped_mask(self):
        wrapped = (self.mask / (2 * np.pi)) * self.slm_params["effective_scale"]
        wrapped = np.mod(wrapped, self.slm_params["effective_scale"] + 1)

        plt.figure("Wrapped Phase Mask")
        plt.imshow(wrapped, cmap='jet', aspect='auto')
        plt.colorbar(label="SLM Value (0 to 939)")
        plt.title("SLM-Wrapped Phase Mask")
        plt.xlabel("Pixel Column")
        plt.ylabel("Pixel Row")
        plt.tight_layout()
        plt.show()

    def open_blanking(self):
        BlankingWindow(self.root, self.blank_mask.shape, self.blank_mask)

    def select_working_directory(self):
        directory = filedialog.askdirectory(title="Select Working Directory")
        if directory:
            self.working_dir = Path(directory)
            messagebox.showinfo("Directory Selected", f"Working directory set to:\n{self.working_dir}")

    def load_to_slm(self):
        if not messagebox.askyesno("Confirm", "Is RA off and SLM ready?"):
            return

        if not self.working_dir:
            self.select_working_directory()
            if not self.working_dir:
                return

        # === IMPORTANT: Freshly rebuild the mask based on current settings ===
        height, width = self.slm_params["height"], self.slm_params["width"]
        total = np.zeros((height, width))

        base_coeffs = np.array(self.baseline_coeffs)
        poly_coeffs = list(map(float, self.poly_coeffs.get().split())) if self.use_poly.get() else []

        if self.baseline_on.get() or self.use_poly.get():
            padded_poly = np.pad(poly_coeffs, (len(base_coeffs) - len(poly_coeffs), 0)) if poly_coeffs else np.zeros_like(base_coeffs)
            full_coeffs = base_coeffs + padded_poly if self.baseline_on.get() else padded_poly
        else:
            full_coeffs = np.zeros_like(base_coeffs)

        poly = generate_polynomial(width, full_coeffs, self.poly_shift.get())
        total += np.tile(poly, (height, 1))

        if self.use_sinc2d.get():
            total += generate_sinc_2d(width, height, kind=self.sinc_mode.get(), extent_x=self.extent_x.get(), extent_y=self.extent_y.get(), a=self.sinc2d_a.get()*np.pi, b=(self.sinc2d_bx.get(), self.sinc2d_by.get()))
        if self.use_sinc1d_col.get():
            cwave = generate_sinc_1d(width, kind=self.sinc_mode.get(), extent=self.extent_x.get(), a=self.sinc1d_col_a.get()*np.pi, b=self.sinc1d_col_b.get())
            total += np.tile(cwave, (height, 1))
        if self.use_sinc1d_row.get():
            rwave = generate_sinc_1d(height, kind=self.sinc_mode.get(), extent=self.extent_y.get(), a=self.sinc1d_row_a.get()*np.pi, b=self.sinc1d_row_b.get())
            total += np.tile(rwave[:, None], (1, width))

        # Apply blank mask
        view_mask = total.copy()
        masked = apply_blank_mask(view_mask, self.blank_mask)
        self.mask = masked  # Update self.mask freshly

        # === Save and Upload ===
        identifier = f"upload_{self.upload_counter:03d}"
        mask_csv_path = self.working_dir / f"{identifier}.csv"
        wrapped = (self.mask / (2 * np.pi)) * self.slm_params["effective_scale"]
        wrapped = np.mod(wrapped, self.slm_params["effective_scale"] + 1)
        np.savetxt(mask_csv_path, wrapped.astype(int), delimiter=",", fmt="%d")

        # Show preview
        plt.figure("Phase Profile Sent to SLM")
        plt.plot(np.mean(wrapped, axis=0))
        plt.title(f"1D Phase Profile Before Upload (Upload {self.upload_counter})")
        plt.xlabel("Pixel Index")
        plt.ylabel("SLM Value (0 to 939)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Upload to SLM
        slm = SantecSLM(
            slm_number=self.slm_params["slm_number"],
            bitdepth=self.slm_params["bitdepth"],
            wave_um=self.slm_params["wave_um"],
            rate=self.slm_params["rate"],
            phase_range=self.slm_params["phase_range"]
        )
        slm.load_csv(str(mask_csv_path))
        slm.close()

        # Save config and log
        config = self.generate_current_config()
        config_path = self.working_dir / f"{identifier}_config.txt"
        with open(config_path, 'w') as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")

        log_path = self.working_dir / "upload_log.txt"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a") as log_file:
            log_file.write(f"Upload {self.upload_counter:03d} | {now} | {mask_csv_path.name}\n")

        self.upload_label.config(text=f"Saved Upload: {self.upload_counter}")
        self.upload_counter += 1



    def run_frog(self):
        p = self.frog_settings
        frog_params = {
            "averaging": p["averaging"].get(),
            "central_motor_position": p["central_motor_position"].get(),
            "integration_time": p["integration_time"].get(),
            "scan_range": (p["scan_range_start"].get(), p["scan_range_end"].get()),
            "step_size": p["step_size"].get()
        }

        frog = FROG(**frog_params)
        trace, real_positions = frog.run(close=False)
        frog.plot(trace, real_positions)

        default_name = f"frog_upload_{self.upload_counter-1:03d}.npz"
        if messagebox.askyesno("Save FROG?", f"Save FROG trace as {default_name}?"):
            if not self.working_dir:
                self.select_working_directory()
            save_path = self.working_dir / default_name
            np.savez(save_path, trace=trace, real_positions=real_positions)

            frog_log_path = self.working_dir / "frog_log.txt"
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(frog_log_path, "a") as flog:
                flog.write(f"{default_name} | {now}\n")

            messagebox.showinfo("Saved", f"FROG saved to {save_path}")

        frog.close_frog()

    def edit_frog_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Edit FROG Settings")
        params = self.frog_settings

        labels = [
            ("Averaging", params["averaging"]),
            ("Central Motor Position", params["central_motor_position"]),
            ("Integration Time (s)", params["integration_time"]),
            ("Scan Range Start (mm)", params["scan_range_start"]),
            ("Scan Range End (mm)", params["scan_range_end"]),
            ("Step Size (mm)", params["step_size"]),
            ("Wavelength Start (nm)", params["wavelength_start"]),
            ("Wavelength End (nm)", params["wavelength_end"]),
        ]

        for i, (label, var) in enumerate(labels):
            ttk.Label(win, text=label).grid(row=i, column=0)
            ttk.Entry(win, textvariable=var).grid(row=i, column=1)

        ttk.Button(win, text="Save Settings", command=lambda: self.save_frog_settings(win)).grid(row=len(labels), column=0, columnspan=2, pady=10)

    def save_frog_settings(self, window):
        if not self.working_dir:
            self.select_working_directory()
            if not self.working_dir:
                return

        settings_dict = {k: v.get() for k, v in self.frog_settings.items()}
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        settings_path = self.working_dir / f"frog_settings_{timestamp}.txt"

        with open(settings_path, "w") as f:
            for key, value in settings_dict.items():
                f.write(f"{key}: {value}\n")

        window.destroy()
        messagebox.showinfo("Saved", f"FROG settings saved to {settings_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = SLMGUI(root)
    root.mainloop()
