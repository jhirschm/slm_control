import ctypes
import time
import os
import numpy as np
import _slm_win as slm


class SLMController:
    def __init__(self):
        """
        Initialize SLM controller, detect display, and set it up.
        """
        self.display_number = self.get_slm_display_number()
        if self.display_number < 0:
            raise RuntimeError("No SLM display detected!")

        # Get device info
        # self.device_info = self.get_slm_info()

        # Set SLM to DVI mode
        if not self.set_dvi_mode():
            raise RuntimeError("Failed to set SLM to DVI mode.")

        print("SLM successfully initialized.")

    def get_slm_display_number(self):
        """
        Detect and return the SLM display number.
        """
        width = ctypes.c_ushort(0)
        height = ctypes.c_ushort(0)
        display_name = ctypes.create_string_buffer(64)

        for display_num in range(1, 8):
            ret = slm.SLM_Disp_Info2(display_num, width, height, display_name)
            if ret == slm.SLM_OK:
                names = display_name.value.decode('mbcs').split(',')
                if 'LCOS-SLM' in names[0]:
                    print(f"Detected SLM on Display {display_num}: {names}, Resolution: {width.value}x{height.value}")
                    self.width = width.value
                    self.height = height.value
                    return display_num

        print("No SLM display detected.")
        return -1

    def set_dvi_mode(self):
        """
        Set the SLM to DVI mode and open the display.
        """
        ret = slm.SLM_Ctrl_Open(self.display_number)
        if ret != slm.SLM_OK:
            print(f"Failed to open SLM. Error code: {ret}")
            return False
        
        ret = slm.SLM_Ctrl_WriteVI(self.display_number, 1)  # 1 = DVI Mode
        if ret == slm.SLM_OK:
            print("SLM set to DVI mode successfully.")
            time.sleep(0.5)  # Allow time for mode change
        else:
            print(f"Failed to set SLM to DVI mode. Error code: {ret}")
            return False

        ret = slm.SLM_Disp_Open(self.display_number)
        if ret == slm.SLM_OK:
            print("SLM display opened successfully.")
            return True
        else:
            print(f"Failed to open SLM display. Error code: {ret}")
            return False

    def get_slm_info(self):
        """
        Retrieve SLM system information.
        """
        Ver = ctypes.create_string_buffer(64)
        ProductID0 = ctypes.create_string_buffer(16)
        ProductID1 = ctypes.create_string_buffer(16)
        LCOSID0 = ctypes.create_string_buffer(32)
        LCOSID1 = ctypes.create_string_buffer(32)
        DisplayName = ctypes.create_string_buffer(32)

        slm.SLM_Ctrl_ReadVR(self.display_number, Ver)
        vers = Ver.value.decode('mbcs').split(',')

        ver_dll = vers[0].split(':')[1]
        ver_op = vers[2].split(':')[1]

        slm.SLM_Ctrl_ReadPS(self.display_number, 0, ProductID0)
        slm.SLM_Ctrl_ReadPS(self.display_number, 1, ProductID1)
        slm.SLM_Ctrl_ReadLS(self.display_number, 0, LCOSID0)
        slm.SLM_Ctrl_ReadLS(self.display_number, 1, LCOSID1)
        slm.SLM_Ctrl_ReadPN(self.display_number, DisplayName)

        slm_info = {
            "DLL Version": ver_dll,
            "Operation Version": ver_op,
            "Product ID": ProductID0.value.decode() + " " + ProductID1.value.decode(),
            "LCOS ID": LCOSID0.value.decode() + " " + LCOSID1.value.decode(),
            "Display Name": DisplayName.value.decode(),
        }

        print("SLM Device Info:", slm_info)
        return slm_info

    def upload_grayscale_csv(self, csv_file_path):
        """
        Upload a grayscale CSV phase profile to the SLM after cleaning headers.

        Parameters:
        - csv_file_path: str, path to the CSV file.
        """
        if not os.path.exists(csv_file_path):
            print(f"CSV file not found at: {csv_file_path}")
            return False

        print(f"Reading CSV file: {csv_file_path}")

        try:
            # Load CSV while skipping headers and labels
            df = pd.read_csv(csv_file_path, delimiter=',', header=None)

            # Remove first row and first column (axis labels)
            grayscale_data = df.iloc[1:, 1:].astype(float).to_numpy()

            # Ensure grayscale values are within 0-1023
            grayscale_data = np.clip(grayscale_data, 0, 1023).astype(np.uint16)

            # Verify dimensions
            h, w = grayscale_data.shape
            if h != self.height or w != self.width:
                print(f"CSV dimensions ({w}x{h}) do not match SLM resolution ({self.width}x{self.height})")
                return False

            # Convert NumPy array to ctypes format
            c_data = grayscale_data.ctypes.data_as(ctypes.POINTER((ctypes.c_ushort * w) * h)).contents

            print("Uploading grayscale phase data to SLM...")
            ret = slm.SLM_Disp_Data(self.display_number, ctypes.c_int16(w), ctypes.c_int16(h), ctypes.c_int32(0), c_data)

            if ret == slm.SLM_OK:
                print("Grayscale phase profile successfully uploaded to SLM.")
                return True
            else:
                print(f"Failed to upload grayscale data. Error code: {ret}")
                return False

        except Exception as e:
            print(f"Error processing CSV file: {e}")
            return False

    def set_uniform_grayscale(self, grayscale_value):
        """
        Set the entire SLM display to a uniform grayscale value.

        Parameters:
        - grayscale_value: int, value between 0 and 1023.
        """
        grayscale_value = np.clip(grayscale_value, 0, 1023)
        ret = slm.SLM_Disp_GrayScale(self.display_number, 0, grayscale_value)

        if ret == slm.SLM_OK:
            print(f"SLM set to uniform grayscale {grayscale_value}.")
        else:
            print(f"Failed to set grayscale. Error code: {ret}")

    def get_current_grayscale(self):
        """
        Read the grayscale value currently displayed on the SLM.
        """
        grayscale = ctypes.c_ushort()
        ret = slm.SLM_Ctrl_ReadGS(self.display_number, ctypes.byref(grayscale))

        if ret == slm.SLM_OK:
            print(f"Current SLM grayscale value: {grayscale.value}")
            return grayscale.value
        else:
            print(f"Failed to read grayscale. Error code: {ret}")
            return None

    def close_slm(self):
        """
        Close the SLM display.
        """
        slm.SLM_Disp_Close(self.display_number)
        print("SLM display closed.")

    def get_parameters(self):
        """
        Return all SLM settings and info as a dictionary.
        """
        return {
            "Display Number": self.display_number,
            **self.device_info
        }
