# coding:utf-8

import ctypes
import time
import os
import _slm_win as slm
from ctypes import wintypes

# Constants for window positioning
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_SHOWWINDOW = 0x0040

# Get the handle of the display window
user32 = ctypes.WinDLL('user32', use_last_error=True)

######################################
# Detect SLM Display Number
######################################
def Get_SLM_DisplayNumber():
    width = ctypes.c_ushort(0)
    height = ctypes.c_ushort(0)
    DisplayName = ctypes.create_string_buffer(64)
    
    for DisplayNumber in range(1, 8):        
        ret = slm.SLM_Disp_Info2(DisplayNumber, width, height, DisplayName)
        if ret == slm.SLM_OK:
            Names = DisplayName.value.decode('mbcs').split(',')
            if 'LCOS-SLM' in Names[0]:
                print(f'Detected SLM on Display {DisplayNumber}: {Names}, Resolution: {width.value}x{height.value}')
                return DisplayNumber
    print('No SLM display detected')
    return -1

######################################
# Set SLM to DVI Mode
######################################
def Set_DVI_mode(DisplayNumber):
    # Open the SLM control
    ret = slm.SLM_Ctrl_Open(DisplayNumber)
    if ret != slm.SLM_OK:
        print(f'Failed to open SLM. Error code: {ret}')
        return False
    
    # Set the SLM to DVI mode (1 = DVI mode)
    ret = slm.SLM_Ctrl_WriteVI(DisplayNumber, 1)
    if ret == slm.SLM_OK:
        print('SLM set to DVI mode successfully.')
        time.sleep(0.5)  # Allow some time for the mode change
    else:
        print(f'Failed to set SLM to DVI mode. Error code: {ret}')
        return False

    # Open the display to make sure it is ready
    ret = slm.SLM_Disp_Open(DisplayNumber)
    if ret == slm.SLM_OK:
        print('SLM display opened successfully.')
        return True
    else:
        print(f'Failed to open SLM display. Error code: {ret}')
        return False

######################################
# Upload CSV File to SLM
######################################
def Upload_CSV_to_SLM(DisplayNumber, csv_file_path: str):
    Flags = slm.FLAGS_RATE120  # Use 120Hz DVI mode flag

    if not os.path.exists(csv_file_path):
        print(f'CSV file not found at: {csv_file_path}')
        return False

    print(f'Uploading CSV file: {csv_file_path}')
    ret = slm.SLM_Disp_ReadCSV(DisplayNumber, Flags, csv_file_path)
    if ret == slm.SLM_OK:
        print(f'CSV file {csv_file_path} successfully uploaded to SLM.')
        return True
    else:
        print(f'Failed to upload CSV file to SLM. Error code: {ret}')
        return False

######################################
# Set SLM Window Position and Size
######################################
def Set_SLM_Window_Position(x: int, y: int, width: int, height: int):
    hwnd = user32.FindWindowW(None, "LCOS-SLM")  # Replace with actual window title if known
    if hwnd:
        print(f"Found SLM window handle: {hwnd}")
        user32.SetWindowPos(hwnd, 0, x, y, width, height, SWP_SHOWWINDOW)
    else:
        print("SLM window not found. Unable to set position.")

######################################
# Main
######################################
def main():
    DisplayNumber = Get_SLM_DisplayNumber()
    if DisplayNumber < 0:
        return

    if not Set_DVI_mode(DisplayNumber):
        return

    csv_file = r'C:\Users\lasopr\Downloads\phase_28.csv'
    if not Upload_CSV_to_SLM(DisplayNumber, csv_file):
        return

    # Set the SLM window to a specific position and size
    Set_SLM_Window_Position(100, 100, 800, 600)

    print('Pattern should now be visible on the SLM.')
    print('Keeping the SLM display active...')

    # Keep the program running to maintain the display
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('Program interrupted. SLM display will remain active.')

if __name__ == '__main__': 
    print('start')
    main()
    print('end')
