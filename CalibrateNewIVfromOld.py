#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:30:53 2019

@author: gerwinv
"""

"""Script used to calibrate a newly acquired IV-curve from new electronics. This script uses an old IV curve
to determine at which level in transition a measurement was done. It will then supply the AC-bias voltage to
pick in the new IV-curve to be at the previously same point in transition. The script expects the bias frequency
and pixel to match!"""

import matplotlib.pyplot as plt
import calivac_niceplot_pwr
import numpy as np
import pars_run78 as pars

plot = True

print("\nPickPoint IVscript:\n")
print("This script will calculate the percentage in transition that the TES was for a given bias point in a given IV-file. It will then look which bias point to pick in the a new IV-file to be at the same percentage in transition.\n")

oldcurve = input("Please supply the Old IV-curve for calibration: ")
oldbiaspoint = int(input("At which bias point (mV) did you previously measure?: "))
newcurve = input("Please supply the New IV-curve you want to calibrate: ")


#oldcurve = "IV_px0_101_1068.025kHz.txt"
#newcurve = "IV_px0_137_1068.475kHz.txt"
#oldbiaspoint = 110

def read_IV_data(file):
    data = np.genfromtxt(file, delimiter=" ")
    column1 = data[:,0]
    column2 = data[:,1]
    column3 = data[:,2]
    return {file: [column1, column2, column3]}

def translate_RT(IVdataFile, pixel, normalized = True):
        calibration = {
                'RfbMiMf'   : pars.rfpx[pixel] * pars.mfi,
                'Rn'        : pars.Rn[pixel],
                'n'         : pars.n[pixel],
                'K'         : pars.K[pixel],
                'vbcal'     : pars.gainvb[pixel],
                'voutcal'   : pars.trafo[pixel],
                'nsmooth'   : pars.Nsmooth[pixel],
                'Tc'        : pars.Tc[pixel],
                'nnorm'     : pars.Nn,
                'nsuper'    : pars.Ns
                }
        text_bias_column = 0
        text_amplitude_column = 1
            
        xtsave = calivac_niceplot_pwr.calivac(
                    ivfile     = IVdataFile, 
                    vbcol      = text_bias_column, 
                    ibcol      = text_amplitude_column,
                    RfbMiMf    = calibration['RfbMiMf'], 
                    Rn         = calibration['Rn'],
                    Tbath      = 60,
                    Tc         = calibration['Tc'],
                    n          = calibration['n'],
                    K          = calibration['K'],
                    vbcal      = calibration['vbcal'],
                    voutcal    = calibration['voutcal'],
                    nsuper     = calibration['nsuper'],
                    nnorm      = calibration['nnorm'],
                    nsmooth    = calibration['nsmooth'],
                    plottag    = None)
            
            
            
        if normalized == True:
            norm = [float(i)/max(xtsave[:,4]) for i in xtsave[:,4]]
            yas = norm
        else:
#            ax.plot(xtsave[:,0],xtsave[:,4])
            yas = xtsave[:,4]
        

        
        xas = xtsave[:,0]
        return [xas, yas]
            

oldIV = read_IV_data(oldcurve)
oldRV = translate_RT(oldcurve, 0)
closests_bias = min(enumerate(oldRV[0]), key=lambda x: abs(x[1]-oldbiaspoint)) 
oldbiasindex = closests_bias[0]
oldbiaspointcal = closests_bias[1]
oldtransitionprecentage = oldRV[1][oldbiasindex]


newIV = oldIV = read_IV_data(newcurve)
newRV = translate_RT(newcurve, 0)

closests_bias = min(enumerate(newRV[0]), key=lambda x: abs(x[1]-oldbiaspoint)) 
newbiasindex = closests_bias[0]
newbiaspointcal = closests_bias[1]
newtransitionprecentage = newRV[1][newbiasindex]

closests_percentage = min(enumerate(newRV[1]), key=lambda x: abs(x[1]-oldtransitionprecentage)) 
closeindex = closests_percentage[0]
closepercentage = closests_percentage[1]
calibratedbiaspoint = newRV[0][closeindex]

if plot == True:
    fig, ax = plt.subplots(1,1)
    ax.plot(oldRV[0],oldRV[1], label="Old Curve")
    ax.plot(newRV[0], newRV[1], label="New Curve")
    
    leg = ax.legend(loc='upper right', title="Frequency (kHz)", prop={'size': '40'})
    leg.set_title('Curve:',prop={'size':'40'})
    ax.set_xlabel("Bias Voltage (mV)", fontsize = '40')
    ax.set_ylabel("R/Rn (-)", fontsize = '40')
    
    
    
    plt.show()

print("\nCalibration finished, presenting result...\n")
print("Old IV-curve is in {} transition for {} mV bias".format(round(oldtransitionprecentage,2), round(oldbiaspointcal,2)))
print("New IV-curve is in {} transition for the old bias point of {} mV".format(round(newtransitionprecentage,2),round(oldbiaspointcal,2)))
print("Change new bias point to {} mV to be at {} in transition".format(round(calibratedbiaspoint,2),round(closepercentage,2)))


