#!/usr/bin/python


# Author: Luciano Gottardi                                                                                                                                                                
# Version: v0.1_20170111                                                                                                                                                                  
# History: 2017/01/11  clean up the script and add some comments..
    
"""
Calibrate the IV curve measured under ac bias using the powet in the normal state to derive the calibraition factor for the voltage. THe TES normal resistance is assumed to be known.
The script expects a file with three or more columns: 
(Vbias,Iamp,phase,Iamp2,phase2,Iamp3,phase3,...). 
The phase is assumed to be the column next to the Iamp column input  by the user

Usage: calivgetic.py ivfile.txt ... (see Example)

  ivfile    = text file with IVs 
  vbcol     = column number for bias voltage
  ibcol     = column number for bias current
  RfbMiMf   = feedback resistance X Min/Mfb coupling factor
  Rn        = TES normal resistance RN
  Tbath     = Bath temperature in mK
  Tc        = TES critical temperature
  n         = n (from G measurements)
  K         = K (from G measurements)
  vbcal     = calibration factor bias voltage
  voutcal   = voltage calibration factor in RT electronics
  nsuper    = number of points to fit superc. branch
  nnorm     = number of points to fit normal  branch
  nsmooth   = Between 0 and 1. The fraction of the data used when estimating each y-value.                       

Optional parameters:                                                          
                                                                            
  --plot Y   = change to Y(yes) if you want to plot the results  (default=N) 

Return: ic,Ptes and calibrated iv file

Example 1: calivac_power.py  iv-rt.txt 0 1  30000.0 0.100 55.0 105 3.6 5.52e-8 1.0 2.2  10 5 0.2 
Example 2: calivac_power.py  iv-rt.txt 0 1  30000.0 0.100 55.0 105 3.6 5.52e-8 1.0 2.2  10 5 0.2 --plot Y 
"""

#Import module
import sys
import optparse
#import xdf as egselib
import numpy as np
import datetime
import os.path
import fileinput

import scipy as sp
import cmath
import matplotlib.pyplot as plt
import matplotlib as mp
import time

from scipy import *


from statsmodels.nonparametric.smoothers_lowess import lowess
from pylab import detrend_linear, mean, hanning, norm, window_hanning, window_none,detrend_mean, detrend_none
from scipy import interpolate
from scipy import stats

#from sklearn.preprocessing import StandardScaler

# load modules with TES and lc parameters                  

from matplotlib import rc
from matplotlib.ticker import MultipleLocator
rc('text', usetex=True)


# LateX configuration                                                                                                                                                      
fig_width_pt = 254.0*1.2  # Get this from LaTeX using \showthe\colum                                           
inches_per_pt = 1.0/72.27               # Convert pt to inch                                                             
golden_mean = (sqrt(5)-1.0)/2.0*1.35         # Aesthetic ratio                          
fig_width = fig_width_pt*inches_per_pt  # width in inches                                                     
fig_height = fig_width*golden_mean      # height in inches                                           
                                                                                         
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 17,
          'text.fontsize': 14,
          'legend.fontsize': 11,
          'xtick.labelsize': 17,
          'ytick.labelsize': 17,
          'text.usetex': True,
          'figure.figsize': fig_size}

plt.rcParams.update(params)



def deriv(x, y):
    """Calculate the derivative dY/dX of a sequence of data points (X,Y).

    x                   -- 1D numpy array of X values
    y                   -- 1D numpy array of Y values, same length as xs

    Returns a 1D numpy array of same length as xs, containing the
    local derivatives dY/dX.

    The calculation is based on 3-point interpolation.
    """

    n = len(x)
    assert n >= 3
    assert len(y) == n

    x   = x.astype(np.float64)

    dx  = x[1:n] - x[0:n-1]
    ddx = x[2:n] - x[0:n-2]

    ret = np.zeros(n)

    ret[0]     = ( - y[0] * (dx[0] + ddx[0]) / (dx[0] * ddx[0])
                   + y[1] * ddx[0] / (dx[0] * dx[1])
                   - y[2] * dx[0] / (dx[1] * ddx[0]) )

    ret[1:n-1] = ( - y[0:n-2] * dx[1:n-1] / (dx[0:n-2] * ddx)
                   + y[1:n-1] / dx[0:n-2]
                   - y[1:n-1] / dx[1:n-1]
                   + y[2:n]   * dx[0:n-2] / (dx[1:n-1] * ddx) )

    ret[n-1]   = ( + y[n-3] * dx[n-2] / (dx[n-3] * ddx[n-3])
                   - y[n-2] * ddx[n-3] / (dx[n-3] * dx[n-2])
                   + y[n-1] * (dx[n-2] + ddx[n-3]) / (dx[n-2] * ddx[n-3]) )

    return ret


def calivac(ivfile,vbcol,ibcol,RfbMiMf,Rn,Tbath,Tc,n,K,vbcal,voutcal,nsuper,nnorm,nsmooth,plottag=None):

    """
    Calibrate Iv curve taken under ac bias using phase information

    Input:

    ivfile    = text file with IVs 
    vbcol     = column number for bias voltage
    ibcol     = column number for bias current
    RfbMiMf   = feedback resistance X Min/Mfb coupling factor
    Rn        = TES normal resistance RN
    Tbath     = Bath temperature in mK
    Tc        = TES critical temperature
    n         = n (from G measurements)
    K         = K (from G measurements)
    vbcal     = calibration factor bias voltage
    voutcal   = voltage calibration factor in RT electronics
    nsuper    = number of points to fit superc. branch
    nnorm     = number of points to fit normal  branch
    nsmooth   = Between 0 and 1. The fraction of the data used when estimating each y-value.  
    plottag   = change to Y(yes) if you want to plot the results  (default=N)

    Output:
    ic,P,P_half
    file: ivcaloutput_power.txt 
 
    """

    if plottag is None:
        plottag='N'


    # read input  file
    ivdata = np.loadtxt(ivfile)

    
    vraw    = ivdata[:,vbcol]   
    iraw    = ivdata[:,ibcol]    
    phase   = ivdata[:,ibcol+1]*np.pi/180  # convert phase to radiant    
        
    # calibrate signals
    vcal=vraw*vbcal
    ical=iraw*voutcal/RfbMiMf
       
    # ---------------------------------------------------------------------------------------
    # 1. Correct bias voltage using phase information assuming the current and the voltage are in phase when the  TES is normal

    #vraw_corr=iraw*(np.cos(0*(phase-15)*np.pi/180))/b #

    ph_norm     = mean(phase[:nnorm])
    ph_super    = mean(phase[-8:-2])
    delay = 0
    phAB  = ph_norm+delay*np.pi/180
    
    vlc_i  = vcal*np.cos(phase-phAB) 
    vlc_q  = vcal*np.sin(phase-phAB)
 
    # ----------------------------------------------------------------                                                                          
    # 2. Smooth the phase in the IV curve                                                                                                       
    
    # Use the tan(phase) to do the smoothing                                                            
    tphsmooth  = lowess(np.tan((phase-phAB)),vlc_i, frac=nsmooth, it=0,return_sorted=False)
    tph_tes    = np.tan((phase-phAB))-tphsmooth
  
    #plt.plot(vcal,tphsmooth,'b-')
    #plt.plot(vcal,tph_tes,'r-')
    #plt.plot(vcal,np.tan((phase-phAB)*np.pi/180),'g.')
    #plt.show()
    
    phasesmooth = np.arctan(tphsmooth)
    phase_tes   = np.arctan(tph_tes)

    # -------------------------------------------------------------------------------------------
    # 3. Calculate the power dissipated in the TES and the TES resistance (not calibrated yet)                                                                           

    ptesuncal = vcal*ical*np.cos((phase-ph_norm))
    rtesuncal = ptesuncal/(ical*ical)

    # get calibration factor assuming the normal resistance is known                 

    calr = Rn/mean(rtesuncal[:nnorm])


    # -----------------------------------
    # 4. Fit the normal part of the IV curve
        
    # Note: this assume Iv curve are taken from normal to super
    slope_norm,int_norm,r_value_n,p_value_n, std_err_n = stats.linregress(np.append(vlc_i[:nnorm],0),np.append(ical[:nnorm],0))
    
    # -----------------------------------------------------
    # 5. Fit the superconductive  part of the IV curve
        
    # Note: this assume IV curves are taken from normal to super
    slope_super,int_super,r_value_s,p_value_s, std_err_s = stats.linregress(np.insert(vlc_i[-1*nsuper:-2],0,0),np.insert(ical[-1*nsuper:-2],0,0))
    
    # -----------------------------------------------------------
    # 6. Get the absolute value of the shunt and series impedance  
    
    bn  = slope_norm  
    bs  = slope_super 
    bratio = bs/bn
    
    r   = Rn/(bratio-1)        # it represents  the total losses in the circuit assuming TES is superconducting at low current.
    
    # the following two lines needs some revision
    Rseries = 0                # to get the Rsh we need to now the calibration of the bias line and convert applied voltage to current. 
    Rsh     = r-Rseries        # for backward compatibility I set Rsh=r. R is anyway the important parameter for the calibration below

    # -----------------------------------------------------------
    # 7. Get TES parameters  
    #
    # Note: Estimation of the Real part of the TES impedance is done using the power calibration. This method works better
    #       when there are ac losses in the TES and the TES s not fully superconducting at the superc. branch of the IV curve
    #       The estimation of the Imaginary part is done following the standard ac bias calibration. 
    #       The assumption here is that any external rectance can be calibrated out using the superc and normal branch of the iv curve
  
    Ztes_Re  = rtesuncal*calr  # [Ohm].
    Ites = ical  # in Ampere

    #Vre = vlc_i*bs*Rn/(bratio-1)-Ites*r # this is the voltage in phase with the current (only resistive component are used to estimate the voltage)
    Vre = Ites*Ztes_Re

    Ptes = Ites*Vre    # in Watt. This is the power dissipated in the Re(Ztes)

    #From here we proceed as in calivac.py
    #Vim     = vlc_q*bs*Rn/(bratio-1)-Ites*(r+Ztes_Re)*np.tan(phasesmooth)  # this is the voltage 90 deg out-of-phase with the current. NOte #using phasesmooth may not be correct  
    Vim=Vre*np.tan(phase-phAB)
    
    Ztes_Im = Vim/Ites

    phtes_wl  = np.arctan(Vim/Vre)     # this is the electrical phase in the weaklink (=equivalent TES impedance) phtes_wl=atan(ImZtes/Re(Ztes)
    Vteswl    = Vre/np.cos((phtes_wl)) # np.sqrt(Vim*Vim+Vreal*Vreal)
    #Vteswl = Vtes/np.cos((phase_tes)*np.pi/180) #np.sqrt(Vim*Vim+Vreal*Vreal)
    
    Ttes   = (Ptes/K + (Tbath*1.e-3)**n)**(1/n)  # Kelvin
    
    #plt.plot(vcal_corr,Vtes,'b-',label='Vtes')
    #plt.plot(vcal_corr,Vtes/np.cos(phtes2),'r-',label='ph_tes2')
    #plt.plot(vcal_corr,Vteswl,'m-',label='phase_tes')
    #plt.plot(vcal_corr,Vim,'g-',label='Vim')
#    plt.plot(vcal_corr,(r+Rtes)*np.tan((phase-phAB)*np.pi/180),'b-')
    #plt.plot(vcal_corr,((phtes2)),'r-')
    #plt.plot(vcal_corr,((phase-phAB)*np.pi/180),'g-')
    #plt.plot(vcal_corr,((phasesmooth)*np.pi/180),'g--')
#    plt.legend(loc=1)
#    plt.show() 

    # -----------------------------------------------------------
    # 8. Get TES Ic and Power
  
    iminvrn=ical-vlc_i*slope_norm  # subtract normal slope from IV

    # find the maximum
    indmax  = np.where(iminvrn==np.max(iminvrn))
    indmaxr = np.where((iminvrn<iminvrn.max()*1.02) & (iminvrn>iminvrn.max()*0.98))
    ic = np.mean(ical[indmaxr])

    # search minumum of the IV curve after the Imax=Ic
    itrans  = ical[0:int(np.array(indmax))]
    indmin  = np.where(itrans==np.min(itrans))

    indminr = np.where((itrans<itrans.min()*1.02) & (itrans>itrans.min()*0.98))
    indhalfr = np.where((Ztes_Re<0.5*Rn*1.01) & (Ztes_Re>0.5*Rn*0.99))
    print('indmax,indmin,ic,ical_min:',indmax[0],indmin[0],ic,ical[indmin[0].max()]) 
    
    try:
        if indhalfr[0].max()>indminr[0].min()-5 and indmaxr[0].max()>indminr[0].min()-5 and indmaxr[0].min()<indminr[0].max()+5:
        
            rthrs = 0.03*Rn
            indr  = np.where(Ztes_Re>np.max(iminvrn))
            
            ic = np.mean(ical[indr[0].max()-1:indr[0].max()+1])
            print('ic from R threshold:', ic,indr[0].max())

            if plottag == 'Y':
                plt.figure('IV fit')
                plt.plot(vlc_i,ical,'b.')
                plt.plot(vlc_i[indr[0].max()-3:indr[0].max()+2],ical[indr[0].max()-3:indr[0].max()+2],'r.')
                plt.legend(loc=1)
                plt.show()
    except:
        pass

    Ptesmin  = np.mean(Ptes[indminr])
    Pteshalf = np.mean(Ptes[indhalfr])

    # -----------------------------------------------------------
    # 9. Calcualte the derivative of the R-t curve dR/dT

    ztrsmth= lowess(Ztes_Re,Ttes, frac=0.2, it=0,return_sorted=False)
    ztismth= lowess(Ztes_Im,Ttes, frac=0.2, it=0,return_sorted=False)
    dZRedT = deriv(Ttes,ztrsmth)
    dZImdT = deriv(Ttes,ztismth)

    dZRedTsmooth=lowess(dZRedT,Ttes, frac=nsmooth, it=0,return_sorted=False)
    dZImdTsmooth=lowess(dZImdT,Ttes, frac=nsmooth, it=0,return_sorted=False)

    # convert output to convenient units 
    
    VreuV       = Vre*1e6
    VtesuVwl    = Vteswl*1e6
    ItesuA      = Ites*1e6
    PtespW      = Ptes*1e12
    Ztes_RemO   = Ztes_Re*1e3
    TtesmK      = Ttes*1e3

    # Here some Plots
   
    if plottag == 'Y':

         plt.figure(1)
         
         #plt.subplots_adjust(wspace=0.9)
         plt.subplot(211)
         plt.plot(vcal,ical*1.e6, 'b-',label='Vbias_raw')
         plt.plot(vlc_i,ical*1.e6, 'r-',label='Vbias_corr')
         plt.plot(vlc_i,slope_norm*vlc_i*1.e6+int_norm*1e6, 'g-',label='fit')
         plt.plot(vlc_i,slope_super*vlc_i*1.e6, 'g-')
         #plt.plot(vlc_i[indr[0].max()-3:indr[0].max()+2],ical[indr[0].max()-3:indr[0].max()+2]*1.e6,'m.',label='Ic')
         #plt.plot(vlc_i[indmaxr],ical[indmaxr]*1.e6, 'm.',label='Ic')
         plt.xlabel('Voltage [a.u.]')
         plt.ylabel('Current [a.u.]')
         plt.ylim([0,300])
         plt.grid(True)
         plt.legend(loc=0)

         plt.subplot(212)
            
         plt.plot(vcal, phase, 'b-',label='phase')
         plt.plot(vlc_i, phasesmooth, 'r-',label='phase_smoothed')
         plt.plot(vlc_i, phase-phAB, 'g.',ms=3,label='phase_vcorr')
         plt.plot(vlc_i, phase_tes, 'm-',label='phase_tes')
         plt.plot(vlc_i, phtes_wl, 'b--',label='phase_tes wl')
         plt.xlabel('Voltage [a.u.]')
         plt.ylabel('phase [rad]')
         #plt.xlim([0.04,0.1])
         #plt.ylim([1e-15,1e-8])
         plt.grid(True)
         
         plt.legend(loc=0)

         plt.figure(2)
         
         #plt.subplots_adjust(wspace=0.9)
         plt.subplot(211)
         plt.plot(vcal,vcal, 'b-',label='Vbias_row')
         plt.plot(vcal,vlc_i, 'r-',label='V in-phase')
         plt.plot(vcal,vlc_q, 'm-',label='V out-phase')
         #plt.plot(vcal_corr,slope_norm*vcal_corr*1.e6+int_norm*1e6, 'g-',label='fit')
         #plt.plot(vcal_corr,slope_super*vcal_corr*1.e6, 'g-')
         #plt.plot(vcal_corr[indmaxr],ical[indmaxr]*1.e6, 'm.',label='Ic')
         plt.xlabel('Voltage [a.u.]')
         plt.ylabel('Voltage [a.u.]')
         #plt.ylim([0,300])
         plt.grid(True)
         plt.legend(loc=0)

         plt.subplot(212)
            
         
         plt.plot(vcal,tphsmooth,'b-',label='tan phsmooth')
         plt.plot(vcal,tph_tes,'r-',label='tan phtes')
         plt.plot(vcal,np.tan(phase-phAB),'g.',ms=3,label='tan phase')
         plt.xlabel('Voltage [a.u.]')
         plt.ylabel('phase [deg]')
         #plt.xlim([0.04,0.1])
         #plt.ylim([1e-15,1e-8])
         plt.grid(True)
         
         plt.legend(loc=0) 


         plt.figure(23)

         #plt.subplots_adjust(wspace=0.9)                                                                                            
         ax1=plt.subplot(211)
         plt.plot(VreuV, ItesuA*np.cos(phase_tes), color='blue',marker='.',ms=3,lw=0,label='TES I-V in-phase')
         #plt.plot(VtesuVwl, ItesuA*np.cos(phtes_wl), 'm--',label='TES I-V wl in-phase')                                             

         #plt.xlabel('$V_{TES}$ [$\mu V$]')                                                                                          
         plt.ylabel('$I_{I}$ [$\mu A$]')
         #plt.xlim([0.,1.5])                                                                                                         
         #plt.ylim([0,30])                                                                                                           
         plt.xlim([0.,0.3])
         plt.ylim([0,90])
         #plt.grid(True)                                                                                                             
         plt.legend(loc=0)
         plt.minorticks_on()
         plt.setp(ax1.get_xticklabels(), visible=False)

         ax2=plt.subplot(212,sharex=ax1)

         #plt.plot(VreuV, ItesuA*np.sin(phase_tes), 'r-',label='TES I-V out-phase')                                                  
         plt.plot(VreuV, np.tan(phase_tes), color='red',marker='.',ms=3,lw=0,label='$I_Q/I_I$')
         #plt.plot(VtesuVwl, ItesuA*np.sin(phtes_wl), 'g--',label='TES I-V wl out-phase')                                            
         plt.xlabel('Voltage [$\mu V$]')
        #plt.ylabel('Current [uA]')                                                                                                  
         plt.ylabel('$I_{Q}/I_{I}$')
         #plt.xlim([0.,1.5])                                                                                                         
         plt.xlim([0.,0.3])
         plt.ylim([-0.25,0.25])
         #plt.grid(True)                                                                                                             
         plt.minorticks_on()
         plt.legend(loc=0)

         plt.savefig('IiIqvsVtes'+'.png',bbox_inches='tight',dpi=300)

         plt.figure(24)

         
 #plt.subplots_adjust(wspace=0.9)                                                                                            
         ax1=plt.subplot(211)
         plt.plot(Ztes_RemO/Ztes_RemO[0], ItesuA*np.cos(phase_tes),color='blue',marker='.',ms=3,lw=0,label='$I_{TES}$ in-phase')

         #plt.xlabel('$R_{TES}/R_N$')                                                                             
         plt.ylabel('$I_{I}$ [$\mu A$]')
         plt.xlim([0.,1.01])
         #plt.ylim([0,30])                                                                                               
         plt.ylim([0,90])
         #plt.grid(True)                                                                                                
         plt.minorticks_on()
         plt.legend(loc=0)
         plt.setp(ax1.get_xticklabels(), visible=False)

         ax2=plt.subplot(212,sharex=ax1)
         #plt.plot(VreuV, ItesuA*np.sin(phase_tes), 'r-',label='TES I-V out-phase')                                                  
         plt.plot(Ztes_RemO/Ztes_RemO[0], np.tan(phase_tes), color='red',marker='.',ms=3,lw=0,label='$I_Q/I_I$')

         plt.xlabel('$R_{TES}/R_N$')
         plt.ylabel('$I_{Q}/I_{I}$')
         plt.xlim([0.,1.01])
         plt.ylim([-0.25,0.25])
         #plt.grid(True)                                                                                               

         plt.minorticks_on()
         plt.legend(loc=0)
         plt.savefig('IiIqvsRtesRn'+'.png',bbox_inches='tight',dpi=300)


         plt.figure(3)
         
         plt.subplots_adjust(wspace=0.9)
         plt.subplot(321)
         plt.plot(VreuV, ItesuA, 'b--',label='TES I-V curve')

         plt.plot(VreuV, ItesuA*np.cos(phase_tes), 'g-',label='TES I-V in-phase')
         plt.plot(VreuV, ItesuA*np.sin(phase_tes), 'r-',label='TES I-V out-phase')
         plt.plot(VtesuVwl, ItesuA*np.cos(phtes_wl), 'm--',label='TES I-V wl in-phase')
         plt.plot(VtesuVwl, ItesuA*np.sin(phtes_wl), 'g--',label='TES I-V wl out-phase')
         plt.xlabel('Voltage [uV]')
         plt.ylabel('Current [uA]')
         #plt.xlim([-0.04,1])
         #plt.ylim([1e-15,1e-8])
         plt.grid(True)
         plt.legend(loc=0,prop={'size':8})

         plt.subplot(322)
         plt.plot(VreuV, PtespW, 'b-',label='TES P-V curve')
         plt.plot(VreuV[indminr],Ptes[indminr]*1e12, 'r.',label='Ptes at IV min')
         plt.plot(VreuV[indhalfr],Ptes[indhalfr]*1e12, 'g.',label='Ptes at IV 50%')
         plt.xlabel('Voltage [uV]')
         plt.ylabel('Power [pW]')
         #plt.xlim([-0.04,1])
         #plt.ylim([1e-15,1e-8])
         plt.grid(True)
         plt.legend(loc=0,prop={'size':8})
  
         plt.subplot(323)
         plt.plot(VreuV, Ztes_RemO, 'b-',label='TES R-V curve')         
         plt.xlabel('Voltage [uV]')
         plt.ylabel('Rtes [mOhm]')
         #plt.xlim([-0.04,1])
         #plt.ylim([1e-15,1e-8])
         plt.grid(True)
         plt.legend(loc=0,prop={'size':8})

         plt.subplot(324)
         plt.plot(TtesmK, Ztes_RemO, 'b-',label='TES R-T curve')         
         plt.xlabel('Temperature [mK]')
         plt.ylabel('Rtes [mOhm]')
         #plt.xlim([45,130])
         #plt.ylim([1e-15,1e-8])
         plt.grid(True)
         plt.legend(loc=0,prop={'size':8})

         plt.subplot(325)
         plt.plot(VreuV,Ttes/Ztes_Re*dZRedT, 'b.',ms=2,label='TES alpha-T curve')
         plt.plot(VreuV,Ttes/Ztes_Re*dZRedTsmooth, 'r-',ms=2,label='smoothed')
         plt.xlabel('Voltage [uV]')
         plt.ylabel('$alpha_{IV}$')
         #plt.xlim([-0.04,1])                                                                                               
         plt.ylim([0,50])
         plt.grid(True)
         plt.legend(loc=0,prop={'size':8})

         plt.subplot(326)
         plt.plot(TtesmK,Ttes/Ztes_Re*dZRedT, 'b.',ms=2,label='TES alpha-T curve')
         plt.plot(TtesmK,Ttes/Ztes_Re*dZRedTsmooth, 'r-',ms=2,label='smoothed')
         plt.xlabel('Temperature [mK]')
         plt.ylabel('$alpha_{IV}$')
         #plt.xlim([0.04,0.1])                                                                                              
         #plt.xlim([45,130])                                                                                                
         plt.ylim([0,50])
         plt.grid(True)
         plt.legend(loc=0,prop={'size':8})

                  
         plt.figure(4)
         
         plt.plot(vlc_i,(ical-vlc_i*slope_norm)*1.e6, 'b-',label='Ites-Vlc_i/Rn')
         #plt.plot(Ztes_RemO,ical*1e6, 'b-',label='Ites-Vlc_i/Rn')
         plt.xlabel('Voltage [a.u.]')
         plt.ylabel('Current [a.u.]')
         plt.ylim([0,300])
         plt.grid(True)
         plt.legend(loc=0)
         plt.show()

        #-------------------------  
    
    #---------------------------------------------
    # Write to file
    
    #Xsave=[vraw,phase,Ites,Vtes,Rtes,Ptes,Ttes,vcal_corr,phase_tes,phtes2*180/np.pi]
    Xsave=[vraw,phase*180/np.pi,Ites,Vre,Ztes_Re,Ptes,Ttes,vlc_i,phtes_wl*180/np.pi,vlc_q,Ztes_Im,phase_tes*180/np.pi,dZRedT,dZImdT]
    Xsave=[vraw,phase*180/np.pi,Ites,Vre,Ztes_Re,Ptes,Ttes,vlc_i,phtes_wl*180/np.pi,vlc_q,Ztes_Im,phase_tes*180/np.pi,dZRedT,dZImdT,dZRedTsmooth,dZImdTsmooth]
    xtsave=np.transpose(Xsave)
    
    hdrtxt =  '# Command line: XDF-CalibrateIV_AC.py'+' '+str(ivfile)+' '+str(vbcol)+' '+str(ibcol)+' '+str(RfbMiMf)+' '+str(Rn)+' '+str(Tbath)+' '+str(Tc)+' '+str(n)+' '+str(K)+' '+str(vbcal)+' '+str(voutcal)+' '+str(nsuper)+' '+str(nnorm)+' '+str(nsmooth)+'\n'
    hdrtxt += '# Rsh='+str(Rsh)+'\n'
    hdrtxt += '# Rn='+str(Rn)+'\n'
    hdrtxt += '# Rseries='+str(Rseries)+'\n'
    hdrtxt += '# Tbath='+str(Tbath)+'\n'
    hdrtxt += '# n='+str(n)+'\n'
    hdrtxt += '# Tc='+str(Tc)+'\n'
    hdrtxt += '# K='+str(K)+'\n'
    hdrtxt += '# 1:Vraw 2:Phase[deg] 3:Ites[A] 4:Vtes[V] 5:Ztes_Re[ohm] 6:Ptes[W] 7:Ttes[K] 8:Vlc_i[V] 9:Phase_wl[deg] 10:Vlc_q[V] 11: Ztes_im[ohm] 12:Phasetes_smoothed[deg] 13:dZRedT[ohm/K] 14:dZImdT[ohm/K] 15:dZRedT_smooth[ohm/K] 16:dZImdT_smooth[ohm/K]'+'\n'

    #fid=open('ivcaloutput_power.txt','w')
    #fid.write(hdrtxt)
    #np.savetxt(fid, xtsave, fmt='%1.8e')
    #fid.close()

    return xtsave
    
def main():

    parser = optparse.OptionParser()
    parser.add_option('--plot', type='str', default='N', action='store')

    try:
        (options, args) = parser.parse_args()
    except optparse.OptionError:
        sys.exit(1)
    except TypeError:
        sys.exit(1)

    if len(args) != 14:
        sys.exit(1)

    # list of argument
    
    ivfile    = args[0]
    vbcol     = int(args[1])
    ibcol     = int(args[2]) 
    RfbMiMf   = float(args[3])
    Rn        = float(args[4])
    Tbath     = float(args[5])
    Tc        = float(args[6])
    n         = float(args[7])
    K         = float(args[8])
    vbcal     = float(args[9])
    voutcal   = float(args[10])
    nsuper    = int(args[11])
    nnorm     = int(args[12])
    nsmooth   = float(args[13])
    
    plottag    = getattr(options,'plot', None)    
 
    (ic,Ptesmin,Pteshalf)=calivac(ivfile,vbcol,ibcol,RfbMiMf,Rn,Tbath,Tc,n,K,vbcal,voutcal,nsuper,nnorm,nsmooth,plottag)

if __name__ == "__main__":
    main()
