# -*- coding: utf-8 -*-
#!/usr/bin/python
import smbus
import math
import time
import pandas as pd
import numpy as np
from threading import Thread
import configparser


class AGV_Status:
    def __init__(self,Mode='Stop',AGV_Status=0):
        self.Mode=Mode
        self.AGV_Status=AGV_Status
        

        # Power management registers
        self.power_mgmt_1 = 0x6b
        self.power_mgmt_2 = 0x6c
        
        self.recordTime=0.015
        self.bus = smbus.SMBus(1) # or bus = smbus.SMBus(1) for Revision 2 boards
        self.address = 0x68       # This is the address value read via the i2cdetect command

        # Now wake the 6050 up as it starts in sleep mode
        self.bus.write_byte_data(self.address, self.power_mgmt_1, 0)
        self.config=configparser.ConfigParser()
        

    def read_byte(self,adr):
        return self.bus.read_byte_data(self.address, adr)
    
    def read_word(self,adr):
        high = self.bus.read_byte_data(self.address, adr)
        low = self.bus.read_byte_data(self.address, adr+1)
        self.val = (high << 8) + low
        return self.val
    
    def read_word_2c(self,adr):
        self.val = self.read_word(adr)
        if (self.val >= 0x8000):
            return -((65535 - self.val) + 1)
        else:
            return self.val
    
    def dist(a,b):
        return math.sqrt((a*a)+(b*b))
    
    def get_y_rotation(x,y,z):
        self.radians = math.atan2(x, self.dist(y,z))
        return -math.degrees(self.radians)
    
    def get_x_rotation(x,y,z):
        self.radians = math.atan2(y, self.dist(x,z))
        return math.degrees(self.radians)

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        while(True):            
            self.config.read("config_mpu6050.txt")
            viewpoint = int(self.config["DEFAULT"]["viewpoint"])#args. View point setting
            RollingAvg = int(self.config["DEFAULT"]["RollingAvg"])#args. Rolling value(Moving Average)
            CheckValue1 = float(self.config["DEFAULT"]["CheckValue1"])#args. 
            CheckValue2 = float(self.config["DEFAULT"]["CheckValue2"])#args. 
            waitTimeValue = int(self.config["DEFAULT"]["waitTimeValue"])#args. 
            StatusVALUE1 = int(self.config["DEFAULT"]["StatusVALUE1"])#args. 
            StatusVALUE2 = int(self.config["DEFAULT"]["StatusVALUE2"])#args.
            LoadingData = np.zeros(shape=(viewpoint,1))
            AccData=np.zeros(shape=(viewpoint,1))
            
            for i in range(viewpoint):
                runtime=time.time()
                timeDiff=time.time()-runtime
                if (timeDiff<self.recordTime):
                    time.sleep(self.recordTime-timeDiff)
                accel_xout = self.read_word_2c(0x3b)
                accel_xout_scaled = accel_xout / 16384.0
                LoadingData[i]= accel_xout_scaled
            AccData = pd.DataFrame(LoadingData).rolling(RollingAvg).mean()
            AccDataNew = np.zeros(shape=((len(AccData)+1)-RollingAvg,1))
            
            j=0
            for k in range(len(AccData)):
                if k >= (RollingAvg-1):
                    AccDataNew[j] = AccData[0][k]
                    j+=1
            
            AccDataNew = AccDataNew-np.mean(AccDataNew)           
           
            
            k=0
            waitTime=0
            
            for k in range(len(AccDataNew)):       
                if self.Mode =='Stop':
                                       
                    self.AGV_Status=0
                    if AccDataNew[k]>CheckValue1:
                        
                        self.AGV_Status=StatusVALUE1
                        self.Mode = 'Forward_Accelerate'
                        
                    if AccDataNew[k]<CheckValue2:
                        
                        self.AGV_Status=StatusVALUE2
                        self.Mode = 'Backward_Accelerate'
                                
                if self.Mode =='Forward_Accelerate':
                    
                    self.AGV_Status=StatusVALUE1
                    if AccDataNew[k]<=0:
                        
                        self.AGV_Status=StatusVALUE1
                        self.Mode = 'Forward'
                                     
                if self.Mode =='Forward':
                    
                    self.AGV_Status=StatusVALUE1
                    if AccDataNew[k]<CheckValue2:
                        
                        self.AGV_Status=StatusVALUE1
                        self.Mode = 'Forward_Decelerate'
                                        
                if self.Mode == 'Forward_Decelerate':
                    
                    self.AGV_Status=StatusVALUE1
                    if AccDataNew[k]>=0:
                        
                        self.AGV_Status=0
                        self.Mode = 'Delay'
                        waitTime = waitTimeValue                        
                        
                if self.Mode =='Backward_Accelerate':
                    
                    self.AGV_Status=StatusVALUE2
                    if AccDataNew[k]>=0:
                        
                        self.AGV_Status=StatusVALUE2
                        self.Mode ='Backward'
                                      
                if self.Mode =='Backward':
                                       
                    self.AGV_Status=StatusVALUE2
                    if AccDataNew[k]>CheckValue1:
                        
                        self.AGV_Status=StatusVALUE2
                        self.Mode ='Backward_Decelerate'
                            
                if self.Mode =='Backward_Decelerate':
                   
                    self.AGV_Status=StatusVALUE2
                    if AccDataNew[k]<=0:
                        
                        self.AGV_Status=0
                        self.Mode = 'Delay'
                        waitTime = waitTimeValue
                          
                if self.Mode =='Delay':
                    waitTime -=1
                    if waitTime==0:
                        self.AGV_Status=0
                        self.Mode ='Stop'
                        
                        
            
                '''
                with open('AGV_Status.txt', 'r+') as f:
                    f.write('[DEFAULT]\n')
                    f.write('AGV_Mode='+str(self.Mode)+'\n')
                    f.write('AGV_Status='+str(int(StatusVALUE[k]))+'\n')
                f.close()
                '''
        
                #print(self.Mode)
                #print(StatusVALUE[k])
                '''
                filepath = "./Datalog2.csv"
                    # 檢查檔案是否存在
                if os.path.isfile(filepath):
                    fileprint=open(filepath, "a")
                else:
                    fileprint=open(filepath, "a")
                    fileprint.write("DateTime,AccX,Mode,StatusVALUE\n")
                    
                Text=time.strftime("%Y%m%d%H%M%S", time.localtime())+","+str(AccDataNew[k])+","+str(Mode)+","+str(StatusVALUE[k])
                fileprint.write(Text+"\n")
                fileprint.close()
                '''
    def read(self):
    # Return the most recent agv mode , agv status 
        return self.Mode,self.AGV_Status

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True

    
    
            
