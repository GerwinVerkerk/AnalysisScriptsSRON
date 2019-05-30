#
#Import module                                                                                                                                                                       
# Pars for GSFC A7 in XFDM-setup
# 

import sys
import optparse
import numpy as np
import datetime
import os
import fileinput
import scipy as sp
import glob

#----------------------------------------------------------                                                                                                                          
# Inputs from user. Needed for the calibrations
#                                          

tesnr    = np.array([1,0,12,9,20,18,7,3,16,13,99,23,15,5,99,11,19,99])
npxs     = int(len(tesnr))
demuxpx  = np.arange(0,npxs)    # pixel numbering in the demux board                                                                                                  
tesname  =('120um narrow(3um) banks','-','-','-','-', '-','-','-','-','-','-','-','-','-','-','-','-','-','-')

freq = np.array([1.07,1.17,1.27,1.37,1.47,1.97,2.65,2.75,2.85,2.95,0.0,3.15,3.7,4.5,0.0,4.7,4.8,0.0])

#pxmsk1 = np.array([True,False,False,False,False,False,False,False,False])
#pxmsk2 = np.array([False,True,False,False,False,False,False,False,False])
#pxmsk3 = np.array([False,False,True,True,False,False,False,False,False])
#pxmsk4 = np.array([False,False,False,False,True,True,True,True,True])
#pxmask = (pxmsk1,pxmsk2,pxmsk3,pxmsk4)
#pxmask = np.array([True,True,True,True,True,True,True,True,True])


Rn       = np.array([35.,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35])*1e-3
Tc       = np.array([87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87])*1e-3    # K                                                         
n        = np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1,1,1,1,1,1,1,1.,1.])*3.2
G100mK   = np.array([112.9,112.9,112.9,112.9,112.9,112.9,112.9,112.9,112.9,112.9,112.9,112.9,112.9,112.9,112.9,112.9,112.9,112.9])*1.e-12 # W/K

G = G100mK*Tc/100.e-3

K        = G/n/Tc**(n-1)
Ctes     = 1.05/0.092*1.e-12*Tc # J/K 
Ns       = 8   # N points to fit superc. branch
Nn       = 4   # N points to fit norm. branch                                                           
Nsmooth  = np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1,1,1,1,1,1,1])*0.1 # N Points to smooth the phase data                                                                   

rfdivsh  = 0
rfdivser = 0
RFEEset  = 6064. #no FB-divider
rffee    = RFEEset #rfdivser*(RFEEset+rfdivsh)/((rfdivser*rfdivsh)/(rfdivser+rfdivsh))   # 5k*(RFEEset+2k)/1.428 

rfpx     = np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1,1,1,1,1,1,1])*rffee  # from FB scan                                                                                     

mfi      = 1.66
trafo    = np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1,1,1,1,1,1,1])*2*.96
calegse  = 4.16
gainvb   = np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1,1,1,1,1,1,1])
#gaini    = np.array([1.05,1.18,1.3,1.53,1.8,2.1,2.36,2.5,2.9])*trafo*calegse  #from SQUID noise roll-off                                                                  
   
gaini    = np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1,1,1,1,1,1,1])*trafo*calegse
calb     = 205*0.1 # uT/mA<== 205mG/mA                                                                                                                                  

imagfile   = 'Imag.txt'
freqfile   = 'BiasFreqList.txt'
tbathfile  = 'TemperatureList.txt'

###--------------------------------------------------------------------------------------------------   

