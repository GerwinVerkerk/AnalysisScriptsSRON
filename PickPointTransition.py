#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:32:42 2019

@author: gerwinv
"""

import matplotlib.pyplot as plt
import calivac_niceplot_pwr
import numpy as np
import pars_run78 as pars

"""This script takes an IV-curve and returns the AC-bias voltage to pick to be at the by the user specified
percentage in transition."""

IVfile = input("Pleas supply an IV-file: ")
pixel = int(input("Which pixel for XFDM?: "))
Transition = float(input("At which percentage (scale 0-1) in transition would you like to measure?: "))

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
    
RV = translate_RT(IVfile, pixel)

closests_transition = min(enumerate(RV[1]), key=lambda x: abs(x[1]-Transition))
index = closests_transition[0]
percentage = round(closests_transition[1],2)
pick_bias = round(RV[0][index],2)

print("\nPick AC-bias voltage of {} mV to be at {} in transition\n".format(pick_bias, percentage))
 