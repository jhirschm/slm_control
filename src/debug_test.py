# coding:utf-8

import ctypes
import time
import numpy as np
import os
import _slm_win as slm

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

# ######################################
# # Set SLM to DVI Mode
# ######################################
# def Set_DVI_mode():
#     DisplayNumber = 1
    
#     # Open the SLM control
#     ret = slm.SLM_Ctrl_Open(DisplayNumber)
#     if ret != slm.SLM_OK:
#         print(f'Failed to open SLM. Error code: {ret}')
#         return
    
#     # Set the SLM to DVI mode (1 = DVI mode)
#     ret = slm.SLM_Ctrl_WriteVI(DisplayNumber, 1)
#     if ret == slm.SLM_OK:
#         print('SLM set to DVI mode successfully.')
#     else:
#         print(f'Failed to set SLM to DVI mode. Error code: {ret}')
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
# Check Current SLM Mode
######################################
def Check_SLM_Mode(DisplayNumber):
    mode = ctypes.c_uint32()
    ret = slm.SLM_Ctrl_ReadVI(DisplayNumber, ctypes.byref(mode))
    if ret == slm.SLM_OK:
        mode_str = 'DVI Mode' if mode.value == 1 else 'Memory Mode'
        print(f'Current SLM Mode: {mode_str} (Mode Code: {mode.value})')
    else:
        print(f'Failed to read SLM mode. Error code: {ret}')

######################################
# Check SLM Display
######################################
def Check_SLM_Display(DisplayNumber):
    Flags = 0  # Default flags, can be adjusted if needed
    ret = slm.SLM_Disp_Open(DisplayNumber)
    if ret == slm.SLM_OK:
        print('SLM display opened successfully.')
    else:
        print(f'Failed to open SLM display. Error code: {ret}')
        
    time.sleep(5)
    slm.SLM_Disp_Close(DisplayNumber)

# ######################################
# # Upload CSV File to SLM
# ######################################
# def Upload_CSV_to_SLM(csv_file_path: str):
#     DisplayNumber = 1
#     Flags = 0  # Default flags, can be adjusted if needed
#     ret = slm.SLM_Disp_Open(DisplayNumber)
#     if ret == slm.SLM_OK:
#         print('SLM display opened successfully.')
#     else:
#         print(f'Failed to open SLM display. Error code: {ret}')

#     if not os.path.exists(csv_file_path):
#         print(f'CSV file not found at: {csv_file_path}')
#         return
#     print(f'Uploading CSV file: {csv_file_path}')
#     ret = slm.SLM_Disp_ReadCSV(DisplayNumber, Flags, csv_file_path)
#     if ret == slm.SLM_OK:
#         print(f'CSV file {csv_file_path} successfully uploaded to SLM.')
#     else:
#         print(f'Failed to upload CSV file to SLM. Error code: {ret}')
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
# Main
######################################
def main():
    # DisplayNumber = Get_SLM_DisplayNumber()
    DisplayNumber = 1
    if DisplayNumber < 0:
        return
    if not Set_DVI_mode(DisplayNumber):
        return
    # Set_DVI_mode()
    Check_SLM_Mode(DisplayNumber)
    csv_file = 'C:\\Users\\lasopr\\Downloads\\phase_28.csv'
    Upload_CSV_to_SLM(DisplayNumber, csv_file)
    # print("Checking Display")
    i = 0
    while(i<40):
        time.sleep(1)
        print("here")
        i+=1
    print("Now going to check")
    slm.SLM_Disp_Close(DisplayNumber)
    time.sleep(5)
    Check_SLM_Display(DisplayNumber)

    
if __name__ == '__main__': 
    print('start')
    main()
    print('end')