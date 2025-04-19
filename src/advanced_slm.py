import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
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

def generate_sinc_2d(width, height, kind='sinc', extent_x=None, extent_y=None, a=np.pi, b=50):
    x = np.arange(width) - width // 2
    y = np.arange(height) - height // 2
    xx, yy = np.meshgrid(x, y)
    zx = np.sinc(xx / b)
    zy = np.sinc(yy / b)
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

class BlankingWindow:
    def __init__(self, parent, mask_shape, blank_mask):
        self.top = tk.Toplevel(parent)
        self.top.title("Interactive Blanking")
        self.blank_mask = blank_mask
        self.canvas = tk.Canvas(self.top, width=800, height=450, bg='white')
        self.canvas.pack()
        self.rects = []
        self.rows, self.cols = mask_shape
        self.pixel_w = 800 // self.cols
        self.pixel_h = 450 // self.rows
        self.draw_grid()

        btn_frame = tk.Frame(self.top)
        btn_frame.pack()
        tk.Button(btn_frame, text="Reset All", command=self.reset_mask).pack(side=tk.LEFT)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<Button-1>", self.on_click)

    def draw_grid(self):
        self.rects.clear()
        self.canvas.delete("all")
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                color = 'black' if self.blank_mask[i, j] == 0 else 'white'
                r = self.canvas.create_rectangle(j*self.pixel_w, i*self.pixel_h,
                                                 (j+1)*self.pixel_w, (i+1)*self.pixel_h,
                                                 fill=color, outline='gray')
                row.append(r)
            self.rects.append(row)

    def toggle_pixel(self, i, j):
        self.blank_mask[i, j] = 1 - self.blank_mask[i, j]
        color = 'black' if self.blank_mask[i, j] == 0 else 'white'
        self.canvas.itemconfig(self.rects[i][j], fill=color)

    def on_click(self, event):
        i = event.y // self.pixel_h
        j = event.x // self.pixel_w
        if 0 <= i < self.rows and 0 <= j < self.cols:
            self.toggle_pixel(i, j)

    def on_drag(self, event):
        self.on_click(event)

    def reset_mask(self):
        self.blank_mask[:, :] = 1
        self.draw_grid()

# === GUI ===
class SLMGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SLM Mask Generator")

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
        self.wrap_output = tk.BooleanVar()

        self.poly_coeffs = tk.StringVar(value="0 0 0 0 0 0")
        self.poly_shift = tk.IntVar(value=0)
        self.extent_x = tk.IntVar(value=1000)
        self.extent_y = tk.IntVar(value=1000)
        self.sinc_a = tk.DoubleVar(value=2.0)
        self.sinc_b = tk.DoubleVar(value=50)
        self.sinc_mode = tk.StringVar(value='sinc2')

        self.setup_ui()
        self.mask = np.zeros((self.slm_params["height"], self.slm_params["width"]))

    def setup_ui(self):
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

        ttk.Label(frm, text="Sinc Amp (0–2π)").grid(row=3, column=0)
        ttk.Scale(frm, variable=self.sinc_a, from_=0.0, to=2.0, orient=tk.HORIZONTAL, length=150).grid(row=3, column=1)
        ttk.Label(frm, text="Sinc Scale (b)").grid(row=3, column=2)
        ttk.Entry(frm, textvariable=self.sinc_b, width=6).grid(row=3, column=3)

        ttk.Label(frm, text="Sinc Type").grid(row=4, column=0)
        ttk.Combobox(frm, textvariable=self.sinc_mode, values=['sinc', 'sinc2'], width=6).grid(row=4, column=1)

        ttk.Checkbutton(frm, text="Add Baseline", variable=self.baseline_on).grid(row=5, column=0)
        ttk.Checkbutton(frm, text="Wrap for SLM", variable=self.wrap_output).grid(row=5, column=1)

        ttk.Button(frm, text="Update Mask", command=self.update_mask).grid(row=6, column=0)
        ttk.Button(frm, text="Load to SLM", command=self.load_to_slm).grid(row=6, column=1)
        ttk.Button(frm, text="Save Mask", command=self.save_mask).grid(row=6, column=2)
        ttk.Button(frm, text="Run FROG", command=self.run_frog).grid(row=6, column=3)
        ttk.Button(frm, text="Blanking", command=self.open_blanking).grid(row=6, column=4)

        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.cbar = None

    def update_mask(self):
        height, width = self.slm_params["height"], self.slm_params["width"]
        total = np.zeros((height, width))

        if self.use_poly.get():
            coeffs = list(map(float, self.poly_coeffs.get().split()))
            poly = generate_polynomial(width, coeffs, self.poly_shift.get())
            total += np.tile(poly, (height, 1))

        if self.use_sinc2d.get():
            total += generate_sinc_2d(width, height, kind=self.sinc_mode.get(), extent_x=self.extent_x.get(), extent_y=self.extent_y.get(), a=self.sinc_a.get()*np.pi, b=self.sinc_b.get())

        if self.use_sinc1d_col.get():
            cwave = generate_sinc_1d(width, kind=self.sinc_mode.get(), extent=self.extent_x.get(), a=self.sinc_a.get()*np.pi, b=self.sinc_b.get())
            total += np.tile(cwave, (height, 1))

        if self.use_sinc1d_row.get():
            rwave = generate_sinc_1d(height, kind=self.sinc_mode.get(), extent=self.extent_y.get(), a=self.sinc_a.get()*np.pi, b=self.sinc_b.get())
            total += np.tile(rwave[:, None], (1, width))

        if self.baseline_on.get():
            baseline = np.polyval(self.baseline_coeffs, np.arange(width) - width // 2)
            total += np.tile(baseline, (height, 1))

        masked = apply_blank_mask(total, self.blank_mask)
        if self.wrap_output.get() and self.baseline_on.get():
            masked = (masked / (2 * np.pi)) * self.slm_params["effective_scale"]
            masked = np.mod(masked, self.slm_params["effective_scale"] + 1)

        self.mask = masked

        self.ax.clear()
        if self.cbar:
            self.cbar.remove()
        im = self.ax.imshow(self.mask, cmap='jet', aspect='auto')
        self.cbar = self.fig.colorbar(im, ax=self.ax)
        self.ax.set_title("Phase Mask")
        self.canvas.draw()

    def open_blanking(self):
        BlankingWindow(self.root, self.blank_mask.shape, self.blank_mask)

    def load_to_slm(self):
        if not messagebox.askyesno("Confirm", "Is RA off and SLM ready?"):
            return
        temp_path = Path("/tmp/slm_gui_mask.csv")
        np.savetxt(temp_path, self.mask.astype(int), delimiter=",", fmt="%d")
        slm = SantecSLM(**self.slm_params)
        slm.load_csv(str(temp_path))
        slm.close()
        messagebox.showinfo("Done", f"Mask loaded to SLM from {temp_path}")

    def save_mask(self):
        out_dir = filedialog.askdirectory()
        if not out_dir:
            return
        ident = simpledialog.askstring("Identifier", "Enter mask name:")
        if ident:
            config = {
                "poly": self.poly_coeffs.get(),
                "sinc_mode": self.sinc_mode.get(),
                "extent_x": self.extent_x.get(),
                "extent_y": self.extent_y.get(),
                "a*pi": self.sinc_a.get(),
                "b": self.sinc_b.get(),
                "baseline": self.baseline_on.get(),
                "wrapped": self.wrap_output.get()
            }
            save_mask_and_config(self.mask, out_dir, ident, config)

    def run_frog(self):
        frog = FROG(integration_time=0.5, averaging=1, central_motor_position=0.165, scan_range=(-0.05, 0.05), step_size=0.005)
        trace, positions = frog.run(close=False)
        frog.plot(trace, positions, wavelength_range=(250, 400), time_axis=True)
        frog.close_frog()

if __name__ == "__main__":
    root = tk.Tk()
    app = SLMGUI(root)
    root.mainloop()