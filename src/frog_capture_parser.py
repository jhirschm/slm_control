import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox

# File path to the FROG measurement data
file_path = '/Users/jhirschm/Documents/ShapingExperiment/FrogData/03062025/flint_1_n.05tp.05w.001_480t560_500ms.pyfrog'


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

# Initialize the plot
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.95)

# Initial plot
im = ax.imshow(
    spectral_array,
    aspect='auto',
    extent=[delay_data.min(), delay_data.max(), wavelength_data.min(), wavelength_data.max()],
    origin='lower',
    cmap='viridis'
)
cbar = plt.colorbar(im, ax=ax, label='Intensity (a.u.)')
ax.set_xlabel('Delay (fs)')
ax.set_ylabel('Wavelength (nm)')
plt.title('FROG Measurement Data')

# Define text box axes with better spacing and labels above
text_box_width = 0.15
text_box_height = 0.05
padding_x = 0.05
padding_y = 0.2

# Create text boxes with labels above
axbox_dmin = plt.axes([0.1, 0.15, text_box_width, text_box_height])
axbox_dmax = plt.axes([0.3, 0.15, text_box_width, text_box_height])
axbox_wmin = plt.axes([0.55, 0.15, text_box_width, text_box_height])
axbox_wmax = plt.axes([0.75, 0.15, text_box_width, text_box_height])

# Rounded default values with units in labels above
textbox_dmin = TextBox(axbox_dmin, '', initial=f"{delay_data.min():.2f}")
axbox_dmin.set_title('Delay Min (fs)', fontsize=10)

textbox_dmax = TextBox(axbox_dmax, '', initial=f"{delay_data.max():.2f}")
axbox_dmax.set_title('Delay Max (fs)', fontsize=10)

textbox_wmin = TextBox(axbox_wmin, '', initial=f"{wavelength_data.min():.2f}")
axbox_wmin.set_title('Wavelength Min (nm)', fontsize=10)

textbox_wmax = TextBox(axbox_wmax, '', initial=f"{wavelength_data.max():.2f}")
axbox_wmax.set_title('Wavelength Max (nm)', fontsize=10)

# Function to update plot based on text inputs
def update_plot(event=None):
    try:
        delay_min = float(textbox_dmin.text)
        delay_max = float(textbox_dmax.text)
        wavelength_min = float(textbox_wmin.text)
        wavelength_max = float(textbox_wmax.text)

        ax.set_xlim([delay_min, delay_max])
        ax.set_ylim([wavelength_min, wavelength_max])
        plt.draw()
    except ValueError:
        print("Invalid input. Please enter numeric values.")

# Bind the update function to the text boxes
textbox_dmin.on_submit(update_plot)
textbox_dmax.on_submit(update_plot)
textbox_wmin.on_submit(update_plot)
textbox_wmax.on_submit(update_plot)

# Add a reset button
resetax = plt.axes([0.85, 0.25, 0.1, 0.05])
button = Button(resetax, 'Reset', color='lightgray', hovercolor='0.975')

def reset(event):
    textbox_dmin.set_val(f"{delay_data.min():.2f}")
    textbox_dmax.set_val(f"{delay_data.max():.2f}")
    textbox_wmin.set_val(f"{wavelength_data.min():.2f}")
    textbox_wmax.set_val(f"{wavelength_data.max():.2f}")
    update_plot()

button.on_clicked(reset)

plt.show()
