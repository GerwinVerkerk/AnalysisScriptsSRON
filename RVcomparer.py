#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:31:50 2019

@author: gerwinv
"""

import matplotlib.pyplot as plt
import calivac_niceplot_pwr
import numpy as np
import pars_run78 as pars

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
            ax.plot(xtsave[:,0],xtsave[:,4])
            yas = xtsave[:,4]
            
        xas = xtsave[:,0]
        return [xas, yas]
    
    