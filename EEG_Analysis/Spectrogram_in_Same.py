# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:40:19 2016

@author: KJJ
"""

import os
#import xlrd 
#import xlwt                                        ## xlrd : reading, xlwt : writing
import numpy as np
import re
#import math
import pandas as pd
#from pandas import Series, DataFrame, Panel
from pandas import ExcelWriter
#from mindwave import pyeeg
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from numpy.fft import fft
from numpy import zeros, floor, log10, log, mean, array, sqrt, vstack, cumsum, \
				  ones, log2, std



def bin_power(X, Band, Fs=256):

    C = fft(X)
    C = abs(C)
    Power =zeros(len(Band)-1);
    for Freq_Index in xrange(0,len(Band)-1):
        Freq = float(Band[Freq_Index])										
        Next_Freq = float(Band[Freq_Index+1])
        Power[Freq_Index] = sum(C[floor(Freq/Fs*len(X)):floor(Next_Freq/Fs*len(X))])

    Power_Ratio = Power/sum(Power)
    return C, Power, Power_Ratio	


def SEF90(X, band=[0,50], Fs=256):

    C= fft(X)
    C = abs(C)
    SEF_Freq = 0
    Power=0

    Power = sum(C[floor(band[0]/Fs*len(X)):floor(band[1]/Fs*len(X))])
    print Power
    for i in xrange(0,len(X)):
        print sum(C[0:i])
        if(sum(C[0:i]) > 0.9*Power): 
            
            SEF_Freq = i/len(X)*Fs
            break
    
    return SEF_Freq

def SEF90(X, band=[0,50], Fs=256):

    C= fft(X)
    C = abs(C)
    SEF_Freq = 0
    Power=0

    Power = sum(C[floor(band[0]/Fs*len(X)):floor(band[1]/Fs*len(X))])
    print Power
    for i in xrange(0,len(X)):
        print sum(C[0:i])
        if(sum(C[0:i]) > 0.9*Power): 
            
            SEF_Freq = i/len(X)*Fs
            break
    
    return SEF_Freq


## 학생 1명에 대해 같다 유형과 다르다 유형에 대한 뇌파 값의 차이 ##

def readLog(path):
    
    event = []                                                          ## event time list
    mouse = []                                                          ## mouse event time list
    mouseMode = False                                                   ## log 중 mouse event 시 True
    testMode = False                                                    ## log 중 test event 시 True
    stimTime = 0                                                        ## stimulate time 
    
    with open(path, 'r') as f:                                          ## log file open
        while True:
            new_line = f.readline()                                     ## log file reading
            
            if len(new_line) is 0: break                                ## log 파일이 끝나면 break
            new_line.rstrip()                                           ## \n 제거
            new_line = new_line.split('\t')                             ## tab으로 split
            if (len(new_line) is not 3) : continue                      ## 3개의 tab으로 나타나지 않는 line은 pass

            if new_line[2].startswith("New trial") and mouseMode:       ## mouse event에서 New trial 이 다시 나타나면
                mouse.append([mouseType, float(new_line[0])])           ## mouseType 값과 timestamp를 mouse에 입력
                mouseMode = False                                       ## mouse event를 읽었으므로 다시 mouseMode False
            
            if new_line[2].startswith("image_3: autoDraw = True")and testMode:                  ## test Event에서 image_3 : autoDraw = True 로 시작하면
                stimTime = float(new_line[0])                                                   ## stimTime을 timestamp에 저장
                testMode = False                                                                ## test event를 읽었으므로 다시 testMode False
            
            if new_line[2].startswith("New trial") and new_line[2].endswith("}\n")  :           ## New trial로 시작하고 {} 가 포함된 경우 : 각 rootine의 시작
                Type = re.search("u'Type': u'(\w+[\w\.]*)'",new_line[2])                        ## Type을 u'Type': u'(____)' 에서 (_____) 부분 reading (정규표현식)
                eventType = Type.groups()[0]                                                    ## eventType을 Type에서 reading
                
                if eventType.startswith('Mouse'):                                                       ## eventType이 Mouse로 시작하면
                    mouseMode = True                                                                    ## mouseMode를 True로 설정
                    mouseType = re.search("u'TestType': u'(\w+[\w\.]*)'",new_line[2]).groups()[0]       ## mouseType을 TestType : u'(______)' 에서 (_____) 부분 reading (정규표현식)
                    
                if eventType.startswith('Test'):                                                ## eventType이 Test로 시작하면
                    testMode = True                                                             ## testMode를 True로 
                                            
            if new_line[1].startswith("DATA") and new_line[2].startswith("Keypress") and eventType.startswith('Test'):  ## DATA log이고, KeyPress로 시작하고 Test event인 경우

#                Keypress = re.search("Keypress: (\w+[\w\.]*)",new_line[2])                                              ## KeyPress : (_____) 에서 (______) 부분 reading
#                PressedKey = Keypress.groups()[0]                                                                       ## PressedKey에 눌린 키 저장
                event.append([float(stimTime), float(new_line[0])])                               ## eventType과 stimulatetime, 그리고 현재 keypressed time 과 pressed key 저장 
    
    f.close()                                                               ## loop가 끝나면 log file close
    
    event = np.array(event)                                                 ## event를 numpy array로 변환
    mouse = np.array(mouse)                                                 ## mouse event를 numpy array로 변환
    
#    print event
    
    eventTime = pd.DataFrame(event, columns=("Stimulate","Response"))    ## event를 DataFrame에 저장
    mouseTime = pd.DataFrame(mouse, columns=("Type","Time"))                                ## mouse를 DataFrame에 저장

    eventTime[["Stimulate","Response"]] = eventTime[["Stimulate","Response"]].astype(float) ## event의 Stimulate와 Response를 float로 변환
    mouseTime[["Time"]] = mouseTime[["Time"]].astype(float)                                 ## mouse의 Time을 float로 변환

    return eventTime, mouseTime     

##################### slice EEG
    
def cropEEG(eeg,event):
    
    
    saved_eeg = eeg                                                                         ## 받은 eeg를 saved_eeg에 저장
    
    prob_eeg = []                                                                           ## prob_eeg list 생성
#    print event
        
    for p in xrange(0,len(event)):                                                          ## event list 개수만큼 p(문제)

        eeg = saved_eeg

#        print eeg['Time']
        StimTime = event['Stimulate'][p]
        RespTime = event['Response'][p]
#        print StimTime, RespTime                                                            ## event로부터 stimTime과 RespTime 받아서
        eeg = eeg[eeg['Time']> StimTime]                                                    ## StimTime부터
        eeg = eeg[eeg['Time']< RespTime]                                                    ## RespTime 까지 eeg data를 자른다.
        
        prob_eeg.append(eeg)                                                                ## 자른 eeg data를 추가한다.
        
    return prob_eeg

#############################

def cropEEG_for_Spec(eeg,event):
    
    
    saved_eeg = eeg                                                                         ## 받은 eeg를 saved_eeg에 저장
    
    prob_eeg = []                                                                           ## prob_eeg list 생성
#    print event
        
    for p in xrange(0,len(event)):                                                          ## event list 개수만큼 p(문제)

        eeg = saved_eeg

#        print eeg['Time']
        StimTime = event['Stimulate'][p]
        RespTime = event['Response'][p]
#        print StimTime, RespTime                                                            ## event로부터 stimTime과 RespTime 받아서
        eeg = eeg[eeg['Time']> (StimTime-2.5)]                                                    ## StimTime부터
        eeg = eeg[eeg['Time']< (RespTime+0.5)]                                                    ## RespTime 까지 eeg data를 자른다.
        
        prob_eeg.append(eeg)                                                                ## 자른 eeg data를 추가한다.
        
    return prob_eeg


###########################################


if __name__ == '__main__':

    CWD = os.path.abspath(os.getcwd())
    
    Subject_path = "Data/Subject_Info.xlsx"
    
    s = ['160319','S19','01']
               
    data_path = "Data/Data/%s_%s %s_data.txt" %(s[0],s[1],s[2])
    header_path = "Data/Data/%s_%s %s_header.txt" %(s[0],s[1],s[2])
    signal_path = "Data/Data/%s_%s %s_signals.txt" %(s[0],s[1],s[2])
    log_path = "Data/Logs/%s_%s %s_log.log" %(s[0],s[1],s[2])
    filename = "Report/%s_%s %s_report.xls" %(s[0],s[1],s[2])
    
    print "Data path : %s" %(data_path)
   
    eventTime, mouseTime = readLog(log_path)
    
#   start = int(math.ceil(float(mouse[1,1])))
    start = float(mouseTime['Time'][0])                                     ## Sync 상의 문제를 해결해야함!!!
    print "StartTime = ", start                                             ## start time plot
    
#   print "EventTime = ", eventTime   
    eventTime = eventTime -start
#       eventTime.sub(start)                 ## start 시점을 빼서 EEG data 에서의 time으로 eventTime 변경
#       print "EventTime = ", eventTime        

    signal = pd.read_csv(signal_path)                           ## signal file 읽기
    a = pd.Series(signal['Label'])                              ## signal['label'] 읽기
    b = pd.Series('Time')
    signals = pd.concat([b,a],ignore_index = True)  
#       print signals
    
    eeg = pd.read_csv(data_path)                                ## data file reading
    eeg.columns = signals                                       ## 
    eeg_in_prob = cropEEG(eeg, eventTime)                       ## eeg를 eventTime으로 자르기.
    eeg_in_prob_for_Spec = cropEEG_for_Spec(eeg, eventTime)                       ## eeg를 eventTime으로 자르기.
    
    p_list = xrange(0,len(eventTime['Stimulate']))
    ch_list = [a[0],a[2],a[4],a[6],a[9],a[11],a[13],a[15]]
#    ch_list = [a[0],a[9]]
    
    p_list_same = [0,2,3,7,8,9,12,14,16,17]
    p_list_diff = [1,4,5,6,10,11,13,15,18,19]  
    p_list_survey = [7,18]

#       for p in p_list:
#           plotEEG(eeg_in_prob[p],signals[1:])
    bands = [0.5,4,8,10,12,15,20,30,50,70]
    band_name = ['Delta','Theta','Low Alpha','High Alpha','Low Beta','Mid Beta','High Beta','Low Gamma', 'High Gamma']

    writer = ExcelWriter(filename)
        
#    for p in p_list_diff:
    for p in p_list_same: 
#    for p in p_list_survey: 
#    for p in xrange(0,20): 
        
        power_list = []    
        
        for ch in ch_list:
#               power, power_ratio = pyeeg.bin_power(eeg_in_prob[p][ch],xrange(0,70),256)
            C, power, power_ratio = bin_power(eeg_in_prob[p][ch],bands,256)
            
            theta_power = power[1]
            alpha_power = power[2]+power[3]
            SMR_power = power[4]
            beta_power = power[4]+power[5]+power[6]
            gamma_power = power[7]+power[8]
            total_power = sum(power)
            
            spec_PSDperHz, spec_freqs, spec_t = mlab.specgram(np.squeeze(eeg_in_prob_for_Spec[p][ch]),
                                       NFFT=256,
                                       window=mlab.window_hanning,
                                       Fs=256,
                                       noverlap= 225
                                       ) # returns PSD power per Hz
        # convert the units of the spectral data
            f_lim_Hz = [0, 50]   # frequency limits for plotting
            plt.figure(figsize=(10,5))
            ax = plt.subplot(1,1,1)
            plt.pcolor(spec_t-0.5, spec_freqs, 10*np.log10(spec_PSDperHz))  # dB re: 1 uV
            plt.clim([-25,26])
            plt.xlim(spec_t[0]-0.5, spec_t[-1]-0.5)
            plt.ylim(f_lim_Hz)
            plt.xlabel('Time (sec)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Channel '+str(ch)+' '+'\nSpectrogram '+ 'in Problem ' +str(p+1))
        # add annotation for FFT Parameters
            ax.text(0.025, 0.95,
                "NFFT = " + str(256) + "\nfs = " + str(int(256)) + " Hz"
                + "\nSMR ratio = " + str(SMR_power/total_power*100) + "  %"
                + "\nGamma ratio = " + str(gamma_power/total_power*100) + "  %"
                + "\nBrain Activities = " + str(beta_power/alpha_power)
                + "\nConcentration Index = " + str((power[4]+power[5])/theta_power)
                + "\nMeditation = " + str(alpha_power/power[6])
                + "\nSEF 90 = " + str(SEF90(eeg_in_prob[p][ch])),
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                backgroundcolor='w')
            plt.show()
            plt.close()


#               power_2, power_ratio_2 = pyeeg.bin_power(eeg_in_prob[p][signals[ch]],xrange(0,70),256)
#               print power 
            power_list.append(power)
        power_list = np.array(power_list)
#           print power_list
        
        power_spec = pd.DataFrame(power_list, index = ch_list , columns = band_name)
        power_spec.to_excel(writer,'Sheet%s'%(p+1))
        
    writer.save()
            