import ftd3xx

import _slm_win as slm
import ctypes
import time


# set 120Hz Flag
Rate120 = True
if Rate120:
    Flags = slm.FLAGS_RATE120
else:
    Flags = 0

# slm device number connected to pc via usb
slm_number = 1

# open usb device
slm.SLM_Ctrl_Open(slm_number)
# set wavelength and phase
# set wavelength and phase range of SLM
# setWL = 1064  # nm
setWL = 1035
setPhaseRange = 200  # 200 means 2.00*pi

# set wavelength and phase range
res = slm.SLM_Ctrl_WriteWL(slm_number, setWL, setPhaseRange)
# wait about 38s for SLM controller to reload the lookup table.
# time.sleep(38)
# save wavelength and phase range to SLM controller
ret = slm.SLM_Ctrl_WriteAW(slm_number)
if res == 0:
    print(
        f"setting SLM{slm_number}'s wavelength to {setWL}nm; phase range to {setPhaseRange*1e-2:.2f}pi."
    )
else:
    print("setting error! The wavelength or phase might be out of the range.")

# print("Setting wavelength and phase range, it would take about 40s ...")
# read wavelength and phase range of SLM1

# define two varialbes to store wavelength and phase range
wavelengthRead = ctypes.c_uint32(0)
phaseRangeRead = ctypes.c_uint32(0)

# read wavelength and phase range of SLM
res = slm.SLM_Ctrl_ReadWL(slm_number, wavelengthRead, phaseRangeRead)
if res != 0:
    print(f"State code:{res}: Error!")
else:
    # get output result of wavelength and phase range setting
    print(f"SLM{slm_number}: read Wavelength, {wavelengthRead.value}nm")
    print(f"SLM{slm_number}: read phase Range, 0~{phaseRangeRead.value*1e-2:.2f}pi")

# change to memory mode
display_mode = ["memory_mode", "dvi_mode"]
memory_mode = 0
dvi_mode = 1
mode_set = memory_mode
ret = slm.SLM_Ctrl_WriteVI(slm_number, mode_set)
if ret != 0:
    print(ret, mode_set, display_mode[mode_set], "Error.")
else:
    print(ret, mode_set, display_mode[mode_set], "OK.")

# check display mode
mode_read = ctypes.c_uint32(0)
ret = slm.SLM_Ctrl_ReadVI(slm_number, mode_read)
if ret != 0:
    print(ret, "Error.", mode_read.value, display_mode[mode_read.value])
else:
    print(ret, "OK.", mode_read.value, display_mode[mode_read.value])

# test memory mode
# change grayscale
slm.SLM_Ctrl_WriteGS(slm_number, 0)
time.sleep(1)

# read phase data from csv files in folder ./phases/
# fn = r"./numbers/001.csv"
fn = r"C:/Users/lasopr/Downloads/phase_13.csv"
memory_id = 1
ret = slm.SLM_Ctrl_WriteMI_CSV(slm_number, memory_id, Flags, fn)
if ret != 0:
    print(ret, fn, "Error.")
else:
    print(ret, fn, "OK.")

# display phase in memory mode
memory_id = 1
ret = slm.SLM_Ctrl_WriteDS(slm_number, memory_id)
if ret != 0:
    print(f"{ret}: error.")
else:
    print(f"{ret}: OK.")

# wait 60s
time.sleep(60)

# close device connected via usb cable
ret = slm.SLM_Ctrl_Close(slm_number)
if ret != 0:
    print(ret, "Error.")
else:
    print(ret, "OK.")
