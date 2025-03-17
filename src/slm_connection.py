# coding:utf-8

import ctypes
import time
import numpy as np
import os
import _slm_win as slm

Rate120 = True

######################################
#  create２d gradation
######################################
def get_gradation_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T
    
    
def ChangeMode(SLMNumber, mode):
    
    if(mode == 0):
        print('Change Memory Mode ',mode)
    elif(mode == 1):
        print('Change DVI Mode',mode)
    else:
        print('No Mode',mode)
        return
    ret = slm.SLM_Ctrl_Open(SLMNumber)
    time.sleep(0.5)
    ret = slm.SLM_Ctrl_WriteVI(SLMNumber,mode)     # 0:Memory 1:DVI
    print('Done',mode)
    return ret

    
######################################
# Test DVI mode
######################################
def Test_DVI_mode():
    DisplayNumber = 1

    
    width = ctypes.c_ushort(0)
    height = ctypes.c_ushort(0)
    DisplayName =  ctypes.create_string_buffer(64)
    
    
    # Search LCOS-SLM
    for DisplayNumber in range(1,8):        
        ret = slm.SLM_Disp_Info2(DisplayNumber, width, height, DisplayName)
        if(ret == slm.SLM_OK):
            Names = DisplayName.value.decode('mbcs').split(',')
            if('LCOS-SLM' in Names[0]): # 'LCOS-SLM,SOC,8001,2018021001'
                print(DisplayNumber, Names, width, height)
                break

    if(DisplayNumber >= 8):
        print('No SLM')
        return
    

    if(Rate120):
        Flags = slm.FLAGS_RATE120
    else:
        Flags = 0

    slm.SLM_Disp_Open(DisplayNumber)
    count = 0
    # for i in range(100):
    #     gray = i * 10
    #     ret = slm.SLM_Disp_GrayScale(DisplayNumber, Flags, gray)
    #     if(ret != slm.SLM_OK):
    #         print(ret,count)
    #         count += 1
    #     time.sleep(0.05)

    
    
    n = get_gradation_2d(0,1023,1920,1200,1)
    n1 = n.astype(np.ushort)
    
    n_h, n_w = n1.shape  # nのサイズを取得
    
    time.sleep(0.1)
    
    # =============================================================================
    # Horizontal scroll
    # for i in range(50):
    #     n1 = np.roll(n1,10)
    #     c = n1.ctypes.data_as(ctypes.POINTER((ctypes.c_ushort * n_h) * n_w)).contents  # ctypesの3x4
    #     ret = slm.SLM_Disp_Data(DisplayNumber, n_w, n_h, Flags, c)
    #     if(ret != slm.SLM_OK): print(ret)
    #     time.sleep(0.1)
    # #    print(i)
    
    time.sleep(1)
    
    print('CSV File')
    # ret = slm.SLM_Disp_ReadCSV(DisplayNumber,Flags,'C:\santec\SLM-200\Files\santec_logo.csv')
    ret = slm.SLM_Disp_ReadCSV(DisplayNumber,Flags,'C:\\Users\\lasopr\\Downloads\\phase_28.csv')
    print(ret)

    
    time.sleep(10)
    
    
    slm.SLM_Disp_Close(DisplayNumber)


######################################
# Infomation
# 
######################################
def Infomation(SLMNumber):
    Ver = ctypes.create_string_buffer(64)
    ProductID0 = ctypes.create_string_buffer(16)
    ProductID1 = ctypes.create_string_buffer(16)
    LCOSID0 = ctypes.create_string_buffer(32)
    LCOSID1 = ctypes.create_string_buffer(32)
    DisplayName = ctypes.create_string_buffer(32)

    slm.SLM_Ctrl_ReadVR(SLMNumber, Ver)
    print(Ver.value)
    vers = Ver.value.decode('mbcs').split(',')

    ver_dll = vers[0].split(':')[1]
    ver_op = vers[2].split(':')[1]

    if(int(ver_dll) >= 250 and int(ver_op) >= 321):
        slm.SLM_Ctrl_ReadPS(SLMNumber, 0, ProductID0)
        slm.SLM_Ctrl_ReadPS(SLMNumber, 1, ProductID1)
        print(ProductID0.value, ProductID1.value)

        slm.SLM_Ctrl_ReadLS(SLMNumber, 0, LCOSID0)
        slm.SLM_Ctrl_ReadLS(SLMNumber, 1, LCOSID1)
        print(LCOSID0.value, LCOSID1.value)

        if(1):
            DN = b'LCOS-SLM-001'     # Display Name max 13byte
            slm.SLM_Ctrl_WritePN(SLMNumber, DN)

        slm.SLM_Ctrl_ReadPN(SLMNumber, DisplayName)
        print(DisplayName.value)


###############################################################################
# Main
###############################################################################
def main():

    Test_DVI_mode()
    
        
if __name__ == '__main__': 
    
    print('start')
    main()
    print('end')
  
    