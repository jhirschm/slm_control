import ctypes
import numpy as np
import warnings
import ftd3xx
import time

try:
    import _slm_win as slm_funcs  # Load Santec's functions
except Exception as e:
    warnings.warn(
        "Santec DLLs not installed. Install these to use Santec SLMs.\n"
        "Ensure the following DLLs are present in the runtime directory:\n"
        "  - SLMFunc.dll\n  - FTD3XX.dll\n"
        f"Original error: {e}"
    )
    slm_funcs = None

class SantecSLM:
    def __init__(self, slm_number=1, display_number=1, bitdepth=10, wav_um=1.0, verbose=True):
        if slm_funcs is None:
            raise RuntimeError("Santec DLLs not installed. Check the runtime directory.")

        self.slm_number = slm_number
        self.display_number = display_number
        self.bitdepth = bitdepth
        self.wav_um = wav_um
        self.verbose = verbose
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
        self._check_status(status, "SLM Open")

        # Activate DVI mode
        status = slm_funcs.SLM_Ctrl_WriteVI(self.slm_number, 1)  # 1 = DVI Mode
        self._check_status(status, "DVI Mode Activation")

        # Retrieve Display Information
        width = ctypes.c_ushort(0)
        height = ctypes.c_ushort(0)
        display_name = ctypes.create_string_buffer(128)
        
        status = slm_funcs.SLM_Disp_Info2(self.display_number, width, height, display_name)
        self._check_status(status, "SLM Display Info Retrieval")

        self.resolution = (width.value, height.value)
        self.display_name = display_name.value.decode("mbcs")
        
        if self.verbose:
            print(f"SLM Display Info:\n  Display Number: {self.display_number}\n  Resolution: {self.resolution[0]} x {self.resolution[1]}\n  Display Name: {self.display_name}")

        # Open Display
        status = slm_funcs.SLM_Disp_Open(self.display_number)
        self._check_status(status, "SLM Display Open")

        # Set the Wavelength
        self._set_wavelength()
    
    def _set_wavelength(self):
        """Sets the operating wavelength of the SLM."""
        wav_nm = int(self.wav_um * 1000)  # Convert to nanometers
        phase_max = 200  # Max phase in SLM internal units
        status = slm_funcs.SLM_Ctrl_WriteWL(self.slm_number, ctypes.c_uint32(wav_nm), phase_max)
        self._check_status(status, "Set Wavelength")

        if self.verbose:
            print(f"Wavelength successfully set to {wav_nm} nm.")
    
    def load_csv(self, file_path):
        """Loads a CSV file onto the SLM."""
        print("Loading")
        # slm_funcs.SLM_Disp_Close(self.display_number)
        status = slm_funcs.SLM_Disp_ReadCSV(self.display_number, 0, file_path)
        self._check_status(status, "CSV Upload")
        # Keep the SLM in memory mode after loading
        # slm_funcs.SLM_Ctrl_WriteVI(self.slm_number, 0)
        if self.verbose:
            print(f"CSV {file_path} successfully uploaded to SLM.")

    def _check_status(self, status, context):
        """Checks SLM status and raises errors if necessary."""
        if status != 0:
            raise RuntimeError(f"{context} Error: {status}")
    
    def close(self):
        """Closes the SLM connection."""
        slm_funcs.SLM_Disp_Close(self.display_number)
        slm_funcs.SLM_Ctrl_Close(self.slm_number)
        if self.verbose:
            print("SLM successfully closed.")

    def is_monitor_active(self):
        """Checks if the SLM monitor is active and receiving input."""
        width = ctypes.c_ushort(0)
        height = ctypes.c_ushort(0)
        display_name = ctypes.create_string_buffer(128)

        status = slm_funcs.SLM_Disp_Info2(self.display_number, width, height, display_name)
        
        if status == 0 and width.value > 0 and height.value > 0:
            if self.verbose:
                print(f"SLM Monitor Active: {display_name.value.decode('mbcs')}, Resolution: {width.value}x{height.value}")
            return True  # Monitor is active
        if self.verbose:
            print("SLM Monitor is NOT active. Please check physical connections.")
        return False

    def check_dvi_mode(self):
        """Checks if the SLM is in DVI mode."""
        status = slm_funcs.SLM_Ctrl_ReadVI(self.slm_number)
        if status == 1:
            if self.verbose:
                print("SLM is in DVI Mode.")
            return True  # DVI mode is enabled
        if self.verbose:
            print("SLM is NOT in DVI Mode.")
        return False

    def set_dvi_mode(self):
        """Activates DVI mode for the SLM."""
        status = slm_funcs.SLM_Ctrl_WriteVI(self.slm_number, 1)  # 1 = Enable DVI Mode
        self._check_status(status, "DVI Mode Activation")
        if self.verbose:
            print("DVI mode activated successfully.")
    
    def set_memory_mode(self):
        """Switches the SLM to Memory Mode to prevent live display output."""
        status = slm_funcs.SLM_Ctrl_WriteVI(self.slm_number, 0)  # 0 = Memory Mode
        self._check_status(status, "Memory Mode Activation")
        if self.verbose:
            print("SLM switched to Memory Mode (No Live Display).")


# # Example Usage
# if __name__ == "__main__":
#     slm = SantecSLM(slm_number=1, display_number=1, bitdepth=10, wav_um=1.0, verbose=True)
#     slm.load_csv("C:\\Users\\lasopr\\Downloads\\phase_29.csv")
    
#     slm.close()

if __name__ == "__main__":
    slm = SantecSLM(slm_number=1, display_number=1, bitdepth=10, wav_um=1.0, verbose=True)

    # # Ensure the monitor is receiving input
    # if not slm.is_monitor_active():
    #     print("Monitor is inactive! Attempting to activate DVI mode...")
    #     slm.set_dvi_mode()
    #     time.sleep(2)  # Wait a moment to allow changes to take effect
    #     if slm.is_monitor_active():
    #         print("SLM monitor is now active.")
    #     else:
    #         print("SLM monitor still inactive. Please check physical connections.")

    # slm.set_memory_mode()

    # Load a CSV file to the SLM
    slm.load_csv("C:\\Users\\lasopr\\Downloads\\phase_29.csv")

    # Allow time for visual inspection
    print("Pausing for 10 seconds for manual verification of SLM display...")
    time.sleep(10)

    # Load a CSV file to the SLM
    slm.load_csv("C:\\Users\\lasopr\\Downloads\\phase_13.csv")

    time.sleep(60)

    # Close connection
    slm.close()

