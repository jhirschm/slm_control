import ctypes
import numpy as np
import warnings
import ftd3xx
import time
import os

import _slm_win as slm_funcs

class SantecSLM:
    def __init__(self, slm_number=1, bitdepth=10, wave_um=1.35, rate=120, phase_range=200, verbose=True):
        if slm_funcs is None:
            raise RuntimeError("Santec DLLs not installed. Check the runtime directory.")

        self.slm_number = int(slm_number)
        self.bitdepth = int(bitdepth)
        self.wave_nm = int(wave_um*1000)
        self.rate = rate
        self.verbose = verbose
        self.phase_range = phase_range
        self.device = None

        if verbose:
            print("Initializing Santec SLM...")

        self._check_ftd3xx_device()
        self._initialize_slm()

    def _check_ftd3xx_device(self):
        """Checks and initializes FTDI communication."""
        num_devices = ftd3xx.createDeviceInfoList()
        if num_devices == 0:
            raise RuntimeError("No FTDI devices detected.")
        
        dev_info = ftd3xx.getDeviceInfoDetail(0)
        if self.verbose:
            print(f"FTDI Device Info: {dev_info}")
        
        self.device = ftd3xx.create(0)
        if not self.device:
            raise RuntimeError("Failed to create FTDI device instance.")
        self.device.close()
        
    def _initialize_slm(self):
        """Initializes SLM communication and activates DVI mode."""
        status = slm_funcs.SLM_Ctrl_Open(self.slm_number)
        
        

        
        # define two varialbes to store wavelength and phase range
        wavelengthRead = ctypes.c_uint32(0)
        phaseRangeRead = ctypes.c_uint32(0)

        # read wavelength and phase range of SLM
        res = slm_funcs.SLM_Ctrl_ReadWL(self.slm_number, wavelengthRead, phaseRangeRead)
        if res != 0:
            print(f"State code:{res}: Error!")
        else:
            # get output result of wavelength and phase range setting
            print(f"SLM{self.slm_number}: read Wavelength, {wavelengthRead.value}nm")
            print(f"SLM{self.slm_number}: read phase Range, 0~{phaseRangeRead.value*1e-2:.2f}pi")

        if wavelengthRead.value != self.wave_nm or phaseRangeRead.value != self.phase_range:
            # set wavelength and phase range
            res = slm_funcs.SLM_Ctrl_WriteWL(self.slm_number, self.wave_nm, self.phase_range)

            ret = slm_funcs.SLM_Ctrl_WriteAW(self.slm_number)

            if res == 0:
                print(
                    f"setting SLM{self.slm_number}'s wavelength to {self.wave_nm}nm; phase range to {self.phase_range*1e-2:.2f}pi."
                )
            else:
                print("setting error! The wavelength or phase might be out of the range.")

            print("Checking updated SLM wavelenght and phase range...")
            if res != 0:
                print(f"State code:{res}: Error!")
            else:
                # get output result of wavelength and phase range setting
                print(f"SLM{self.slm_number}: read Wavelength, {wavelengthRead.value}nm")
                print(f"SLM{self.slm_number}: read phase Range, 0~{phaseRangeRead.value*1e-2:.2f}pi")


        memory_mode = 0
        mode_set = memory_mode
        display_mode = ["memory_mode", "dvi_mode"]

        ret = slm_funcs.SLM_Ctrl_WriteVI(self.slm_number, mode_set)
        if ret != 0:
            print(ret, mode_set, display_mode[mode_set], "Error.")
        else:
            print(ret, mode_set, display_mode[mode_set], "OK.")

        # check display mode
        mode_read = ctypes.c_uint32(0)
        ret = slm_funcs.SLM_Ctrl_ReadVI(self.slm_number, mode_read)
        if ret != 0:
            print(ret, "Error.", mode_read.value, display_mode[mode_read.value])
        else:
            print(ret, "OK.", mode_read.value, display_mode[mode_read.value])

        # test memory mode
        # change grayscale
        slm_funcs.SLM_Ctrl_WriteGS(self.slm_number, 0)
        time.sleep(1)

        print("-----------")
        print("SLM Initialization Complete from SLM Backend")
        print("-----------")
    
    def load_csv(self, file_path, memory_id=1):
        """Loads a CSV file onto the SLM."""
        
        if os.path.exists(file_path) != 1:
            print("File does not exist.")
        else:
            print("Loading File")
        if self.rate==120:
            Flags = slm_funcs.FLAGS_RATE120
        else:
            Flags = 0
        ret = slm_funcs.SLM_Ctrl_WriteMI_CSV(self.slm_number, memory_id, Flags, file_path)
        if ret != 0:
            print(ret, file_path, "Error.")
        else:
            print(ret, file_path, "OK.")

        # display phase in memory mode
        memory_id = 1
        ret = slm_funcs.SLM_Ctrl_WriteDS(self.slm_number, memory_id)
        if ret != 0:
            print(f"{ret}: error.")
        else:
            print(f"{ret}: OK.")

    
    def close(self):
        """Closes the SLM connection."""
        ret = slm_funcs.SLM_Ctrl_Close(self.slm_number)
        if ret != 0:
            print(ret, "Error.")
        else:
            print(ret, "SLM Closed.")




