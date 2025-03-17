import ctypes
import time
import numpy as np
import sys
import os
import clr
from clr_loader import get_coreclr
# Spectrometer setup
slm_func_path = r"C:\slm_control\src"
if slm_func_path not in sys.path:
    sys.path.append(slm_func_path)

# # Load the .NET DLL
# dll_path = os.path.join(slm_func_path, "SLMFunc.dll")
# if os.path.exists(dll_path):
#     clr.AddReference(dll_path)
# else:
#     raise FileNotFoundError("DLL path does not exist!")


# slm_func_path = "C:\\slm_control\\src\\"
dll = ctypes.cdll.LoadLibrary("C:\\slm_control\\src\\SLMFunc.dll")
print(dll)

# Get function names
function_names = dir(dll)

# Print available functions
print("Available functions in SLMFunc.dll:")
for name in function_names:
    print(name)

print(dll.SLM_Disp_Open(ctypes.c_int32(1)))
print(dll.SLM_Ctrl_ReadSU(ctypes.c_int32(1))=="SLM_OK")
print(dll.SLM_Ctrl_ReadSU(ctypes.c_int32(1)))
print("here")

for i in range(10):
    # ret = dll.SLM_Disp_GrayScale(ctypes.c_int32(1),ctypes.c_int32(0), ctypes.c_int(i*10))
    # print(ret)
    # print(dll.SLM_Ctrl_ReadSU(ctypes.c_int32(1)))
    print(dll.SLM_Ctrl_WriteWL(ctypes.c_int32(1), ctypes.c_int32(1000), ctypes.c_int32(200)))
    


    time.sleep(1)

time.sleep(5)

# def get_gradation_2d(start, stop, width, height, is_horizontal):
#     if is_horizontal:
#         return np.tile(np.linspace(start, stop, width), (height, 1))
#     else:
#         return np.tile(np.linspace(start, stop, height), (width, 1)).T

# n = get_gradation_2d(0,1023,1920,1200,1)
# n1 = n.astype(np.int16)
# n_h, n_w = n1.shape # height, width
# for i in range(5):
#     n1 = np.roll(n1,10)
#     c = n1.ctypes.data_as(ctypes.POINTER((ctypes.c_int16 * n_h) * n_w)).contents # convert
#     ret = dll.SLM_Disp_Data(ctypes.c_int32(1),ctypes.c_int16(n_w),ctypes.c_int16(n_h),ctypes.c_int32(0),c)
#     if(ret != 0): print(ret)
#     time.sleep(0.05)
# time.sleep(0.5)

# csv_file = 'C:\\Users\\lasopr\\Downloads\\phase_13.csv'
# ret = dll.SLM_Disp_ReadCSV(ctypes.c_int32(1),ctypes.c_int32(0),csv_file)
# time.sleep(60)
# dll.SLM_Disp_Close(ctypes.c_int32(1))