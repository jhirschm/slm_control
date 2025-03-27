//
//  SlmCtrl.m
//  LcosSlm
//
//  Created by santec on 2022/06/01.
//  Copyright © 2022 SANTEC CORPORATION. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#import <Foundation/Foundation.h>
#import "SlmCtrl.h"
#import "SlmCommon.h"
#import "ftd3xxlib/ftd3xx.h"

@implementation SlmCtrl

static SlmCtrl *sharedSlmCtrl = nil;    // シングルトンのための静的インスタンス

// シングルトンアクセスメソッド
+ (SlmCtrl *)shared {
    @synchronized(self) {
        if(sharedSlmCtrl == nil) {
            sharedSlmCtrl = [[self alloc] init];
        }
    }
    return sharedSlmCtrl;
}

/**
 * initialize
 */
- (id)init {
    self = [super init];
    if (self) {
    }
    return self;
}

- (FT_STATUS)slm_Write:(FT_HANDLE)handle pucBuffer:(PUCHAR)pucBuffer ulBufferLength:(ULONG)ulBufferLength pulBytesTransferred:(PULONG)pulBytesTransferred {
    FT_STATUS ftStatus;
    
    ftStatus = FT_WritePipe(handle, 0x02, pucBuffer, ulBufferLength, pulBytesTransferred, NULL);
    [SlmCommon myPrintf:[NSString stringWithFormat:@"WR: ret %d, send %d\n", ftStatus, *pulBytesTransferred]];
    
    return ftStatus;
}

- (FT_STATUS)slm_Read:(FT_HANDLE)handle pucBuffer:(PUCHAR)pucBuffer ulBufferLength:(ULONG)ulBufferLength pulBytesTransferred:(PULONG)pulBytesTransferred {
    FT_STATUS ftStatus;
    
    ftStatus = FT_ReadPipe(handle, 0x82, pucBuffer, ulBufferLength, pulBytesTransferred, NULL);
    [SlmCommon myPrintf:[NSString stringWithFormat:@"RD: ret %d, read %d\n", ftStatus, *pulBytesTransferred]];

    return ftStatus;
}

/**
 * SlmNumber Check
 */
- (SLM_STATUS)CheckSLMNumber:(DWORD)slmNumber {
    if ((slmNumber >= 1) && (slmNumber <= MAX_SLM)) {
        if (ftHandle[slmNumber] != 0)
            return SLM_OK;
        else
            return SLM_NOT_OPEN_USB;
    }
    else {
        return SLM_NG;
    }
}

/**
 * Response Check
 */
- (SLM_STATUS)CheckRecv:(char*)Recv {
    if (strncmp(Recv, "NG", 2) == 0) return SLM_NG;
    if (strncmp(Recv, "BS", 2) == 0) return SLM_BS;
    if (strncmp(Recv, "ER", 2) == 0) return SLM_ER;
    return SLM_OK;
}

/**
 * SlmConnect Info
 */
- (void)FTDIList {
    DWORD numDevs = 0;
    FT_HANDLE ftHandle = NULL;
    FT_STATUS ftStatus = FT_CreateDeviceInfoList(&numDevs);

    if (!FT_FAILED(ftStatus) && numDevs > 0) {
        DWORD flags = 0;
        DWORD type = 0;
        DWORD iD = 0;
        char serialNumber[32] = {0};
        char description[64] = {0};
        for (DWORD i = 0; i < numDevs; i++) {
            ftStatus = FT_GetDeviceInfoDetail(i, &flags, &type, &iD, NULL, serialNumber, description, &ftHandle);
            if (!FT_FAILED(ftStatus)) {
                [SlmCommon myPrintf:[NSString stringWithFormat:@"Device[%d]\n", i]];
                [SlmCommon myPrintf:[NSString stringWithFormat:@"\tFlags: 0x%x %@ | Type: %d | ID: 0x%08X | ftHandle=0x%p\n",
                                     flags,
                                     flags & FT_FLAGS_SUPERSPEED ? @"[USB 3]" :
                                     flags & FT_FLAGS_HISPEED ? @"[USB 2]" :
                                     flags & FT_FLAGS_OPENED ? @"[OPENED]" : @"",
                                     type,
                                     iD,
                                     ftHandle]];
                [SlmCommon myPrintf:[NSString stringWithFormat:@"\tSerialNumber=%s\n", serialNumber]];
                [SlmCommon myPrintf:[NSString stringWithFormat:@"\tDescription=%s\n", description]];
            }
        }
    }
}

/**
 * Open USB interface
 */
- (SLM_STATUS)slm_Ctrl_Open:(DWORD)slmNumber {
    SLM_STATUS ret = [self CheckSLMNumber:slmNumber];
    //mod by passion Open済み時、ftcreateするとhandleがクリアされる為、Open済み時は処理を抜ける
    //if (ret != SLM_NOT_OPEN_USB && ret != SLM_OK)
    if (ret != SLM_NOT_OPEN_USB)
        return ret;

    [self FTDIList];
    unsigned long index = slmNumber - 1;

    //add by passion チップ情報&転送パラメータ設定の追加
    if (![self SetChipConfig:index])
        return SLM_NG;
    else
        [self SetFTTransferParams];

    FT_STATUS ftStatus = FT_Create((PVOID)index, FT_OPEN_BY_INDEX, &ftHandle[slmNumber]);
    
    if (ftStatus == FT_OK) {
        ftStatus = FT_SetPipeTimeout(ftHandle[slmNumber], 0x02, 3000);
        ftStatus = FT_SetPipeTimeout(ftHandle[slmNumber], 0x82, 3000);
        return SLM_OK;
    }
    else {
        return (SLM_STATUS)(SLM_FTDI_ERROR - ftStatus);
    }
}

/**
 * Close USB interface
 */
- (SLM_STATUS)slm_Ctrl_Close:(DWORD)slmNumber {
    SLM_STATUS ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK)
        return ret;
    
    FT_STATUS ftStatus = FT_Close(ftHandle[slmNumber]);
    ftHandle[slmNumber] = 0;

    if (ftStatus == FT_OK)
        return SLM_OK;
    else
        return (SLM_STATUS)(SLM_FTDI_ERROR - ftStatus);
}

/**
 * SlmFromReceiveRequest and ReceiveProc
 */
- (SLM_STATUS)RequestReceiveData:(FT_HANDLE)handle recv:(BYTE*)recv recv_len:(USHORT*)recv_len retry:(int)retry {
    FT_STATUS ftStatus;
    UCHAR acBuf[PACKET1_SIZE] = { 0xFF };
    ULONG ulBytesTransferred = 0;

    // Retry
    for (int loop = 0; loop < retry; loop++) {
        memset(acBuf, 0, sizeof(acBuf));
        acBuf[0] = 'S';
        acBuf[1] = 'E';
        acBuf[2] = 'N';
        acBuf[3] = 'D';

        // 0x02 Packet
        acBuf[5] = 0x00;
        acBuf[7] = 0x02;
        ulBytesTransferred = 0;
        ftStatus = [self slm_Write:handle pucBuffer:acBuf ulBufferLength:PACKET2_SIZE pulBytesTransferred:&ulBytesTransferred];
        
        // 0x03 Packet
        if (ftStatus == FT_OK) {
            ulBytesTransferred = 0;
            ftStatus = [self slm_Read:handle pucBuffer:acBuf ulBufferLength:PACKET3_SIZE pulBytesTransferred:&ulBytesTransferred];
            if (ftStatus == FT_OK) {
                memcpy(recv, &(acBuf[16]), ulBytesTransferred);
                *recv_len = (USHORT)ulBytesTransferred;
                [SlmCommon myPrintf:[NSString stringWithFormat:@"%s\n", recv]];
                if ((strncmp((char*)&(acBuf[16]), "NO RESPONSE", 11) != 0) &&
                    (strncmp((char*)&(acBuf[16]), "TO", 2) != 0))
                    return SLM_OK;;

            }
        }
        else {
            return (SLM_STATUS)(SLM_FTDI_ERROR - ftStatus);
        }
        if (loop < 5)       [NSThread sleepForTimeInterval:SETTING_WAIT1];   // 50
        else if (loop < 10) [NSThread sleepForTimeInterval:SETTING_WAIT2];   // 100
        else if (loop < 20) [NSThread sleepForTimeInterval:SETTING_WAIT3];   // 500
        else                [NSThread sleepForTimeInterval:SETTING_WAIT4];   // 1000

    } // loop

    return SLM_OTHER_ERROR;
}

/**
 * General Send and Receive
 */
- (SLM_STATUS)slm_Ctrl_WriteXX:(DWORD)slmNumber send:(BYTE*)send send_len:(USHORT)send_len recv:(BYTE*)recv recv_len:(USHORT*) recv_len retry:(DWORD)retry {

    FT_STATUS ftStatus;
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;
    
    UCHAR acBuf[PACKET1_SIZE] = { 0xFF };
    memset(acBuf, 0, sizeof(acBuf));

    acBuf[0] = 'S';
    acBuf[1] = 'E';
    acBuf[2] = 'N';
    acBuf[3] = 'D';
    acBuf[5] = 0xFF;
    acBuf[7] = 0x01;
    memcpy(&(acBuf[16]), send, send_len);
    
    //del by passion ReadPipeでFT_OPERATION_ABORTEDになるため削除
//    FT_FlushPipe(ftHandle[slmNumber], 0x02);
//    FT_FlushPipe(ftHandle[slmNumber], 0x82);

    // 0x01 Packet
    ULONG ulBytesTransferred = 0;
    ftStatus = [self slm_Write:ftHandle[slmNumber] pucBuffer:acBuf ulBufferLength:PACKET1_SIZE pulBytesTransferred:&ulBytesTransferred];

    if (ftStatus == FT_OK) {
        return [self RequestReceiveData:ftHandle[slmNumber] recv:recv recv_len:recv_len retry:retry];
    }
    else {
        return (SLM_STATUS)(SLM_FTDI_ERROR - ftStatus);
    }
}

/**
 * Transfer array data to SLM memory
 */
- (SLM_STATUS)slm_Ctrl_WriteMI:(DWORD)slmNumber memoryNumber:(DWORD)memoryNumber width:(USHORT)width height:(USHORT)height flags:(DWORD)flags data:(USHORT *)data {

    FT_STATUS ftStatus = FT_OK;
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;
    if (width > 1920 || height > 1200) return SLM_DATA_FORMAT_ERR;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    ret = [self slm_Ctrl_ReadSU:slmNumber];
    if (ret != SLM_OK) {
        return ret;
    }
    else {
        snprintf(Send, 16, "MI %d\n", memoryNumber);

        UCHAR *Buf = (UCHAR*)malloc(sizeof(char) * PACKET4_SIZE * 1200 + 4);
        memset(Buf, 0, PACKET4_SIZE * 1200 + 4);

        UCHAR *acBuf;

        UCHAR skipCol = 0;
        USHORT stWidth, edWidth, stHeight, edHeight;
        if (width < 1920) {
            stWidth = (USHORT)((1920 - width) / 2);
            edWidth = stWidth + width - 1;
        }
        else {
            stWidth = 0;
            edWidth = 1919;
        }
        if (height < 1200) {
            stHeight = (USHORT)((1200 - height) / 2);
            edHeight = stHeight + height - 1;
        }
        else {
            stHeight = 0;
            edHeight = 1199;
        }

        // Packt 1,2,3
        ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
        if (ret == SLM_OK) {
            ret = [self CheckRecv:Recv];
            if (ret == SLM_OK) {

                DWORD sum = 0;
                WORD *pos = data;

                for (int row = 0; row < 1200; row++) {
                    acBuf = &Buf[PACKET4_SIZE * row];

                    memset(acBuf, 0, PACKET4_SIZE);

                    acBuf[0] = 'S';
                    acBuf[1] = 'E';
                    acBuf[2] = 'N';
                    acBuf[3] = 'D';
                    acBuf[4] = 0x03;
                    acBuf[5] = 0xBF; // 0xFF;
                    acBuf[7] = 0x04; // Frame ID

                    ULONG ulBytesTransferred = 0;
                    // Sequence number
                    acBuf[11] = (UCHAR)((row >> 16) & 0xFF);
                    acBuf[10] = (UCHAR)((row >> 8) & 0xFF);
                    acBuf[9] = (UCHAR)(row & 0xFF);

                    UCHAR *dat = &acBuf[16];
                    for (int col = 0; col < 1920; col++) {
                        // Range Check
                        if (row >= stHeight && row <= edHeight && col >= stWidth && col <= edWidth) {
                            WORD d = *pos;
                            dat[(col * 2) + 1] = (d & 0x3FF) >> 8;          // higher data
                            dat[(col * 2) + 0] = (d & 0x3FF) & 0xFF;        // lower  data
                            sum += (DWORD)(d & 0x3FF);
                            pos++;
                        }
                        // Out of Range
                        else {
                            dat[(col * 2) + 1] = (skipCol & 0x3FF) >> 8;        // higher data
                            dat[(col * 2) + 0] = (skipCol & 0x3FF) & 0xFF;      // lower  data
                            sum += (DWORD)(skipCol & 0x3FF);
                        }

                    }
                    if (row == 1199) {
                        // Check Sum
                        acBuf[1027 * 4 + 3] = (UCHAR)((sum >> 24) & 0xFF);
                        acBuf[1027 * 4 + 2] = (UCHAR)((sum >> 16) & 0xFF);
                        acBuf[1027 * 4 + 1] = (UCHAR)((sum >> 8) & 0xFF);
                        acBuf[1027 * 4 + 0] = (UCHAR)(sum & 0xFF);

                        [SlmCommon myPrintf:[NSString stringWithFormat:@"sum:%d\n", sum]];
                        ftStatus = [self slm_Write:ftHandle[slmNumber] pucBuffer:Buf ulBufferLength:PACKET4_SIZE * 1200 pulBytesTransferred:&ulBytesTransferred];
                        if (ftStatus != FT_OK) {
                            return (SLM_STATUS)(SLM_FTDI_ERROR - ftStatus);
                        }
                    }
                    
                    // 0x04 Packet
                }
                [NSThread sleepForTimeInterval:SETTING_WAIT_P4];    // ここで待たないと何回りトライしてもダメ

                if (ftStatus == FT_OK) {
                    delete Buf;
                    SLM_STATUS ret;
                    // Packt 2,3
                    ret = [self RequestReceiveData:ftHandle[slmNumber] recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
                    if (ret == SLM_OK) {
                        return [self CheckRecv:Recv];
                    }
                }
                else return (SLM_STATUS)(SLM_FTDI_ERROR - ftStatus);

            }
            else return ret;

        }
        else return ret;
    }
    
    return SLM_OTHER_ERROR;
}

/**
 * Transfer BMP file(Unicode) to SLM memory
 */
- (SLM_STATUS)slm_Ctrl_WriteMI_BMP:(DWORD)slmNumber memoryNumber:(DWORD)memoryNumber flags:(DWORD)flags fileName:(LPCSTR)fileName {
    SLM_STATUS ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK)
        return ret;
    
    int width, height;
    uint16_t *data = (uint16_t*)malloc(MAX_WIDTH * MAX_HEIGHT * sizeof(WORD));

    ret = [SlmCommon ReadBMP:fileName flags:(flags | FLAGS_INCWORD) width:&width height:&height pPix:data];
    if (ret != SLM_OK) {
        free(data);
        return ret;
    }

    ret = [self slm_Ctrl_WriteMI:slmNumber memoryNumber:memoryNumber width:width height:height flags:flags data:data];
    free(data);
    
    return ret;
}

/**
 * Transfer CSV file(Unicode) to SLM memory
 */
- (SLM_STATUS)slm_Ctrl_WriteMI_CSV:(DWORD)slmNumber memoryNumber:(DWORD)memoryNumber flags:(DWORD)flags fileName:(LPCSTR)fileName {
    SLM_STATUS ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK)
        return ret;

    int width, height;
    uint16_t *data = (uint16_t*)malloc(MAX_WIDTH * MAX_HEIGHT * sizeof(WORD));

    ret = [SlmCommon ReadCSV:fileName flags:(flags | FLAGS_INCWORD) width:&width height:&height pPix:data];
    if (ret != SLM_OK) {
        free(data);
        return ret;
    }

    ret = [self slm_Ctrl_WriteMI:slmNumber memoryNumber:memoryNumber width:width height:height flags:flags data:data];
    free(data);

    return ret;
}

/**
 * Write video mode DVI or Memory mode
 */
- (SLM_STATUS)slm_Ctrl_WriteVI:(DWORD)slmNumber mode:(DWORD)mode {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;
    
    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "VI %d\n", mode);

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY_VI];
    if (ret == SLM_OK) {  // VI 1でガンマテーブル参照のため遅い
        return [self CheckRecv:Recv];
    }
    else if (ret <= SLM_OTHER_ERROR) {
        for (int i = 0; i < 40; i++) {
            [NSThread sleepForTimeInterval:1.0];
            if ([self slm_Ctrl_ReadSU:slmNumber] == SLM_OK) return SLM_OK;
        }
        return ret;
    }
    else return ret;
}

/**
 * Read display mode DVI or Memory mode
 */
- (SLM_STATUS)slm_Ctrl_ReadVI:(DWORD)slmNumber mode:(DWORD *)mode {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;
    
    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "VI\n");

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        ret = [self CheckRecv:Recv];
        if (ret == SLM_OK) {
            *mode = atoi(Recv);
            return SLM_OK;
        }
        else return ret;
    }
    else return ret;
}

/**
 * Write wavelength and phase value
 */
- (SLM_STATUS)slm_Ctrl_WriteWL:(DWORD)slmNumber wavelength:(DWORD)wavelength phase:(DWORD)phase {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;
    
    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "WL %d %1.1f\n", wavelength, ((double)phase)/ 100);

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY_WL];
    if (ret == SLM_OK) {
        return [self CheckRecv:Recv];
    }
    else return ret;
}

/**
 * Read wavelength and phase value
 */
- (SLM_STATUS)slm_Ctrl_ReadWL:(DWORD)slmNumber wavelength:(DWORD *)wavelength phase:(DWORD *)phase {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;
    
    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "WL\n");

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        ret = [self CheckRecv:Recv];
        if (ret == SLM_OK) {
            float p;
            NSString *str = [NSString stringWithCString:Recv encoding:NSUTF8StringEncoding];
            NSScanner *objScanner = [NSScanner scannerWithString:str];
            int length;
            [objScanner scanInt:&length];
            *wavelength = (DWORD)length;
            [objScanner scanFloat:&p];
            p *= 100;
            p += 0.5;            // 四捨五入
            *phase = int(p);

            return SLM_OK;
        }
        else return ret;
    }
    else return ret;
}

/**
 * Save wavelength and phase settings
 */
- (SLM_STATUS)slm_Ctrl_WriteAW:(DWORD)slmNumber {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "AW\n");

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        return [self CheckRecv:Recv];
    }
    else return ret;

}

/**
 * Write ON / OFF of trigger input value
 */
- (SLM_STATUS)slm_Ctrl_WriteTI:(DWORD)slmNumber onoff:(DWORD)onoff {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "TI %d\n", onoff);

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        return [self CheckRecv:Recv];
    }
    else return ret;

}

/**
 * Read ON / OFF of trigger input value
 */
- (SLM_STATUS)slm_Ctrl_ReadTI:(DWORD)slmNumber onoff:(DWORD *)onoff {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "TI\n");

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        ret = [self CheckRecv:Recv];
        if (ret == SLM_OK) {
            *onoff = atoi(Recv);
            return SLM_OK;
        }
        else return ret;
    }
    else return ret;
}

/**
 * Write ON / OFF of trigger output value
 */
- (SLM_STATUS)slm_Ctrl_WriteTM:(DWORD)slmNumber onoff:(DWORD)onoff {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "TM %d\n", onoff);

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        return [self CheckRecv:Recv];
    }
    else return ret;
}

/**
 * Read ON / OFF of trigger output value
 */
- (SLM_STATUS)slm_Ctrl_ReadTM:(DWORD)slmNumber onoff:(DWORD *)onoff {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "TM\n");

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        ret = [self CheckRecv:Recv];
        if (ret == SLM_OK) {
            *onoff = atoi(Recv);
            return SLM_OK;
        }
        else return ret;
    }
    else return ret;
}

/**
 * Write ascending / descending order of pattern display by trigger input
 */
- (SLM_STATUS)slm_Ctrl_WriteTC:(DWORD)slmNumber order:(DWORD)order {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "TC %d\n", order);

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        return [self CheckRecv:Recv];
    }
    else return ret;
}

/**
 * Read ascending / descending order of pattern display by trigger input
 */
- (SLM_STATUS)slm_Ctrl_ReadTC:(DWORD)slmNumber order:(DWORD *)order {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "TC\n");

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        ret = [self CheckRecv:Recv];
        if (ret == SLM_OK) {
            *order = atoi(Recv);
            return SLM_OK;
        }
        else return ret;
    }
    else return ret;
}

/**
 * Performs same operation as trigger input
 */
- (SLM_STATUS)slm_Ctrl_WriteTS:(DWORD)slmNumber {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "TS\n");

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        return [self CheckRecv:Recv];
    }
    else return ret;
}

/**
 * Transfer phase pattern input from the DVI input to internal memory
 */
- (SLM_STATUS)slm_Ctrl_WriteMC:(DWORD)slmNumber memoryNumber:(DWORD)memoryNumber {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "MC %d\n", memoryNumber);

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        return [self CheckRecv:Recv];
    }
    else return ret;
}

/**
 * Invalidates phase pattern stored in internal memory
 */
- (SLM_STATUS)slm_Ctrl_WriteME:(DWORD)slmNumber memoryNumber:(DWORD)memoryNumber {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "ME %d\n", memoryNumber);

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        return [self CheckRecv:Recv];
    }
    else return ret;
}

/**
 * Replace memory number set in display table
 */
- (SLM_STATUS)slm_Ctrl_WriteMT:(DWORD)slmNumber tableNumber:(DWORD)tableNumber memoryNumber:(DWORD)memoryNumber {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "MT %d %d\n", tableNumber, memoryNumber);

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        return [self CheckRecv:Recv];
    }
    else return ret;
}

/**
 * Read memory number set in display table
 */
- (SLM_STATUS)slm_Ctrl_ReadMS:(DWORD)slmNumber tableNumber:(DWORD)tableNumber memoryNumber:(DWORD *)memoryNumber {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "MS %d\n", tableNumber);

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        ret = [self CheckRecv:Recv];
        if (ret == SLM_OK) {
            *memoryNumber = atoi(Recv);
            return SLM_OK;
        }
        else return ret;
    }
    else return ret;
}

/**
 * Write effective range of display table
 */
- (SLM_STATUS)slm_Ctrl_WriteMR:(DWORD)slmNumber tableNumber1:(DWORD)tableNumber1 tableNumber2:(DWORD)tableNumber2 {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "MR %d %d\n", tableNumber1, tableNumber2);

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        return [self CheckRecv:Recv];
    }
    else return ret;
}

/**
 * Read effective range of display table
 */
- (SLM_STATUS)slm_Ctrl_ReadMR:(DWORD)slmNumber tableNumber1:(DWORD *)tableNumber1 tableNumber2:(DWORD *)tableNumber2 {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "MR\n");

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        ret = [self CheckRecv:Recv];
        if (ret == SLM_OK) {
            int number;
            NSString *str = [NSString stringWithCString:Recv encoding:NSUTF8StringEncoding];
            NSScanner *objScanner = [NSScanner scannerWithString:str];
            [objScanner scanInt:&number];
            *tableNumber1 = (DWORD)number;
            [objScanner scanInt:&number];
            *tableNumber2 = (DWORD)number;
            return SLM_OK;
        }
        else return ret;
    }
    else return ret;
}

/**
 * Write table number of display table to be displayed first
 */
- (SLM_STATUS)slm_Ctrl_WriteMP:(DWORD)slmNumber tableNumber:(DWORD)tableNumber {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "MP %d\n", tableNumber);

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        return [self CheckRecv:Recv];
    }
    else return ret;
}

/**
 * Set contents of display table to default settings
 */
- (SLM_STATUS)slm_Ctrl_WriteMZ:(DWORD)slmNumber {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "MZ\n");

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        return [self CheckRecv:Recv];
    }
    else return ret;
}

/**
 * Write interval for switching pattern display by number of frames
 */
- (SLM_STATUS)slm_Ctrl_WriteMW:(DWORD)slmNumber frames:(DWORD)frames {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "MW %d\n", frames);

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        return [self CheckRecv:Recv];
    }
    else return ret;
}

/**
 * Read interval for switching pattern display by number of frames
 */
- (SLM_STATUS)slm_Ctrl_ReadMW:(DWORD)slmNumber frames:(DWORD *)frames {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "MW\n");

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        ret = [self CheckRecv:Recv];
        if (ret == SLM_OK) {
            *frames = atoi(Recv);
            return SLM_OK;
        }
        else return ret;
    }
    else return ret;
}

/**
 * Specify memory number to display internal memory phase pattern
 */
- (SLM_STATUS)slm_Ctrl_WriteDS:(DWORD)slmNumber memoryNumber:(DWORD)memoryNumber {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "DS %d\n", memoryNumber);

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        return [self CheckRecv:Recv];
    }
    else return ret;
}

/**
 * Read displayed memory number
 */
- (SLM_STATUS)slm_Ctrl_ReadDS:(DWORD)slmNumber memoryNumber:(DWORD *)memoryNumber {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "DS\n");

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        ret = [self CheckRecv:Recv];
        if (ret == SLM_OK) {
            *memoryNumber = atoi(Recv);
            return SLM_OK;
        }
        else return ret;
    }
    else return ret;
}

/**
 * Display phase patterns stored in internal memory in order of display table
 */
- (SLM_STATUS)slm_Ctrl_WriteDR:(DWORD)slmNumber order:(DWORD)order {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "DR %d\n", order);

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        return [self CheckRecv:Recv];
    }
    else return ret;
}

/**
 * Stop continuous display by SLM_Ctrl_WriteDR function
 */
- (SLM_STATUS)slm_Ctrl_WriteDB:(DWORD)slmNumber {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "DB\n");

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        return [self CheckRecv:Recv];
    }
    else return ret;
}

/**
 * Display specified grayscale on the entire display
 */
- (SLM_STATUS)slm_Ctrl_WriteGS:(DWORD)slmNumber grayScale:(USHORT)grayScale {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "GS %d\n", grayScale);

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        return [self CheckRecv:Recv];
    }
    else return ret;
}

/**
 * Read grayscale on display
 */
- (SLM_STATUS)slm_Ctrl_ReadGS:(DWORD)slmNumber grayScale:(USHORT *)grayScale {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "GS\n");

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        ret = [self CheckRecv:Recv];
        if (ret == SLM_OK) {
            *grayScale = atoi(Recv);
            return SLM_OK;
        }
        else return ret;
    }
    else return ret;
}

/**
 * Read drive board and option board Celsius temperatures
 */
- (SLM_STATUS)slm_Ctrl_ReadT:(DWORD)slmNumber driveTemp:(INT32 *)driveTemp optionTemp:(INT32 *)optionTemp {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;
    float f;

    snprintf(Send, 16, "TD\n");

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY_TD];
    if (ret == SLM_OK) {
        ret = [self CheckRecv:Recv];
        if (ret == SLM_OK) {
            //static float a = -10;
            //snprintf(Recv, 16, "%f", a); a += 0.1;
            NSString *str = [NSString stringWithCString:Recv encoding:NSUTF8StringEncoding];
            [[NSScanner scannerWithString:str] scanFloat:&f];

            if(f > 0)    *driveTemp = (int)(f * 10 + 0.5);
            else         *driveTemp = (int)(f * 10 - 0.5);

            snprintf(Send, 16, "TO\n");
            // Packt 1,2,3
            ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
            if (ret == SLM_OK) {
                ret = [self CheckRecv:Recv];
                if (ret == SLM_OK) {
                    str = [NSString stringWithCString:Recv encoding:NSUTF8StringEncoding];
                    [[NSScanner scannerWithString:str] scanFloat:&f];
                    if (f > 0)      *optionTemp = (int)(f * 10 + 0.5);
                    else            *optionTemp = (int)(f * 10 - 0.5);

                    return SLM_OK;
                }
                else return ret;
            }
            else return ret;
        }
        else return ret;
    }
    else return ret;
}

/**
 * Read error flags of "Drive board" and "Option board"
 */
- (SLM_STATUS)slm_Ctrl_ReadEDO:(DWORD)slmNumber driveError:(DWORD *)driveError optionError:(DWORD *)optionError {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "ED\n");

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        ret = [self CheckRecv:Recv];
        if (ret == SLM_OK) {
            *driveError = atoi(Recv);

            snprintf(Send, 16, "EO\n");
            // Packt 1,2,3
            ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
            if (ret == SLM_OK) {
                ret = [self CheckRecv:Recv];
                if (ret == SLM_OK) {
                    *optionError = atoi(Recv);

                    return SLM_OK;
                }
                else return ret;
            }
            else return ret;
        }
        else return ret;
    }
    else return ret;
}

/**
 * Read serial number
 */
- (SLM_STATUS)slm_Ctrl_ReadSN:(DWORD)slmNumber serialNo:(LPSTR)serialNo {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "SN\n");

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        ret = [self CheckRecv:Recv];
        if (ret == SLM_OK) {
            memcpy(serialNo, Recv, 10);
            serialNo[10] = 0;
            return SLM_OK;
        }
        else return ret;
    }
    else return ret;
}

/**
 * Read status of SLM. Busy or Ready
 */
- (SLM_STATUS)slm_Ctrl_ReadSU:(DWORD)slmNumber {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;
    
    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;
    snprintf(Send, 16, "SU\n");

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        return [self CheckRecv:Recv];
    }
    else return ret;
}

/**
 * Read identification numbers of "Drive board" and "Option board"
 */
- (SLM_STATUS)slm_Ctrl_ReadSDO:(DWORD)slmNumber driveID:(LPSTR)driveID optionID:(LPSTR)optionID {
    SLM_STATUS ret;
    ret = [self CheckSLMNumber:slmNumber];
    if (ret != SLM_OK) return ret;

    char Send[16];
    char Recv[PACKET3_SIZE];
    USHORT rlen = 0;

    snprintf(Send, 16, "SD\n");

    // Packt 1,2,3
    ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
    if (ret == SLM_OK) {
        ret = [self CheckRecv:Recv];
        if (ret == SLM_OK) {
            memcpy(driveID, Recv, 8);
            driveID[8] = 0;

            snprintf(Send, 16, "SO\n");
            // Packt 1,2,3
            ret = [self slm_Ctrl_WriteXX:slmNumber send:(BYTE*)Send send_len:USHORT(strlen(Send)) recv:(BYTE*)Recv recv_len:&rlen retry:RETRY];
            if (ret == SLM_OK) {
                ret = [self CheckRecv:Recv];
                if (ret == SLM_OK) {
                    memcpy(optionID, Recv, 8);
                    optionID[8] = 0;

                    return SLM_OK;
                }
                else return ret;
            }
            else return ret;
        }
        else return ret;
    }
    else return ret;
}

- (SLM_STATUS)slm_Debug:(DWORD)onoff {
    [SlmCommon SetDebugMode:(bool)onoff];

    return SLM_OK;
}

/**
 * Set transfer parameters for each FIFO channel
 */
- (void)SetFTTransferParams {
    FT_TRANSFER_CONF conf;

    memset(&conf, 0, sizeof(FT_TRANSFER_CONF));
    conf.wStructSize = sizeof(FT_TRANSFER_CONF);
    conf.pipe[FT_PIPE_DIR_IN].bURBCount = 7;
    conf.pipe[FT_PIPE_DIR_OUT].bURBCount = 7;
    conf.pipe[FT_PIPE_DIR_IN].dwURBBufferSize = 16384;
    conf.pipe[FT_PIPE_DIR_OUT].dwURBBufferSize = 16384;
    FT_SetTransferParams(&conf, 0);
}

/**
 * Set chip configuration
 */
- (BOOL)SetChipConfig:(unsigned long)index {
    FT_HANDLE handle = NULL;
    DWORD dwType = FT_DEVICE_UNKNOWN;

    FT_Create((PVOID)index, FT_OPEN_BY_INDEX, &handle);
    
    if (!handle)
        return false;
    
    FT_GetDeviceInfoDetail(0, NULL, &dwType, NULL, NULL, NULL, NULL, NULL);
    if (dwType != FT_DEVICE_600 && dwType != FT_DEVICE_601)
        return false;

    union CHIP_CONFIGURATION {
        FT_60XCONFIGURATION ft600;
    } old_cfg, new_cfg;

    if (FT_OK != FT_GetChipConfiguration(handle, &old_cfg)) {
        [SlmCommon myPrintf:@"Failed to get chip conf\r\n"];
        return false;
    }
    memcpy(&new_cfg, &old_cfg, sizeof(union CHIP_CONFIGURATION));

    new_cfg.ft600.FIFOClock = CONFIGURATION_FIFO_CLK_66;
    new_cfg.ft600.FIFOMode = CONFIGURATION_FIFO_MODE_245;
    new_cfg.ft600.OptionalFeatureSupport = CONFIGURATION_OPTIONAL_FEATURE_DISABLEALL;
    
    if (FT_OK != FT_SetChipConfiguration(handle, &new_cfg)) {
        [SlmCommon myPrintf:@"Failed to set chip conf\r\n"];
        return false;
    } else {
        [NSThread sleepForTimeInterval:1.0];
    }
    
    FT_Close(handle);
    return true;
}

@end
