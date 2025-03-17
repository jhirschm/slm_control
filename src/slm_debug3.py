import os
import ctypes
import numpy as np
import cv2
import warnings
import ftd3xx

try:  # Load Santec's header file.
   import _slm_win as slm_funcs

except BaseException as e:  # Provide an informative error should something go wrong.
    warnings.warn(
        "Santec DLLs not installed. Install these to use Santec SLMs."
        "  Dynamically linked libraries from Santec (usually provided via USB) "
        "must be present in the runtime directory:\n"
        "  - SLMFunc.dll\n  - FTD3XX.dll\n"
        "  Check that these files are present and are error-free.\n"
        "Original error: {}".format(e)
    )
    slm_funcs = None

def _parse_status(status, raise_error=True):
        """
        Parses the meaning of a ``SLM_STATUS`` return from a Santec SLM.

        Parameters
        ----------
        status : int
            ``SLM_STATUS`` return.
        raise_error : bool
            Whether to raise an error (if True) or a warning (if False) when status is not ``SLM_OK``.

        Returns
        -------
        (int, str, str)
            Status in ``(name, note)`` form.
        """
        # Parse status
        status = int(status)

        if not status in slm_funcs.SLM_STATUS_DICT.keys():
            raise ValueError("SLM status '{}' not recognized.".format(status))

        # Recover the meaning of status
        (name, note) = slm_funcs.SLM_STATUS_DICT[status]

        status_str = "Santec error {}; '{}'".format(name, note)

        if status != 0:
            if raise_error:
                raise RuntimeError(status_str)
            else:
                warnings.warn(status_str)

        return (status, name, note)
slm_number = int(1)
display_number = int(2)

wav_design_um = 1
status = slm_funcs.SLM_Ctrl_ReadSU(1)
num_devices = ftd3xx.createDeviceInfoList()


print(num_devices)
print(f"Found {num_devices} device(s)")

if num_devices > 0:
    dev_info = ftd3xx.getDeviceInfoDetail(0)
    # dev = ftd3xx.create(0)
    # print(dev_info)
    # print(dev)
    # # dev.open(0)  # Try opening the first device
    # print("Device opened successfully!")
    # dev.close()
    status = slm_funcs.SLM_Ctrl_Open(slm_number)
    print(status)
    status = slm_funcs.SLM_Ctrl_ReadSU(slm_number)
    print(status)
while True:
    status = slm_funcs.SLM_Ctrl_ReadSU(slm_number)

    if status == 0:
        print("Ok")
        break  # SLM_OK (proceed)
    elif status == 2:
        continue  # SLM_BS (busy)
    else:
        _parse_status(status)
drive_error = ctypes.c_uint32(0)
option_error = ctypes.c_uint32(0)
slm_funcs.SLM_Ctrl_ReadEDO(slm_number, drive_error, option_error)
_parse_status(
            slm_funcs.SLM_Ctrl_ReadEDO(slm_number, drive_error, option_error),
            raise_error=True,
        )
