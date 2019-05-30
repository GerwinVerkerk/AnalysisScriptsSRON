import os
import numpy
import pars_run78 as pars

def medfilt (x, k):
    """Applies k-length median filter
    
    Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    
    Args:
        x (array): 1D array which is to be filtered
        k (int): Length of the median filter
    
    Returns:
        filtered (array): median filtered array
    
    """
    k2 = (k - 1) // 2
    y = numpy.zeros ((len (x), k), dtype=numpy.float16)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
        
    filtered = numpy.median (y, axis=1)
    return filtered

def calibrate_data(v_raw, i_raw, phase, Tbath, calibration, px):
    """Calibrates the data
    
    Callibrates the measured data and creates 2 data files per pixel
    in which the result of the callibration is saved
    
    Args:
        data (array): data from the read_data() function
        pxs (list): list of measured pixels
        Num_Dsets (int): number of data sets in the HDF file
        attr_keys (list): list of argument names in order
        calibration (dict): dictionary of constants used in the calibration
        
    Returns:
        None: saves the callibration data to two .txt files
    """
    
    # Checks if path location exists and creates if not
    if not os.path.exists('data/output/'):
        os.makedirs('data/output/')
    if not os.path.exists('data/analyis_data/'):
            os.makedirs('data/analyis_data/')
    if not os.path.exists('data/Plots/Ic'):
        os.makedirs('data/Plots/Ic') 

    # median filters data
    i_raw_med = medfilt(i_raw, 7)
    phase_med = medfilt(phase, 7)

    # Looks for outliers in data
    bad_ind1 = numpy.where(numpy.abs((i_raw_med/i_raw) - 1) > .10)[0]
    bad_ind2 = numpy.where(numpy.abs((phase_med/phase) - 1) > .10)[0]
    bad_ind = list(set(numpy.append(bad_ind1, bad_ind2)))

    # removes outliers
    v_raw = numpy.delete(v_raw, bad_ind)
    i_raw = numpy.delete(i_raw, bad_ind)
    phase = numpy.delete(phase, bad_ind)

    # convert phase to radiant
    phase = phase * (numpy.pi/180)

    # Calibrate signals
    v_cal = numpy.float64(v_raw) * calibration['vbcal'][px]
    i_cal = numpy.float64(i_raw) * (calibration['voutcal'][px]/
        calibration['RfbMiMf'][px])

    #======================================================================
    # 1. Correct bias voltage using phase information assuming the current
    #    and the voltage are in phase when the TES is normal

    ph_norm = mean(phase[:10])
    ph_super = mean(phase[-10:-2])

    delay = 0
    phAB = ph_norm + delay * (numpy.pi/180)

    vlc_i = v_cal * numpy.cos(phase - phAB)
    vlc_q = v_cal * numpy.sin(phase - phAB)

    #======================================================================
    # 2. Smooth the phase in the IV curve

    tph_smooth = lowess(numpy.tan((phase-phAB)),vlc_i, frac=calibration['nsmooth'][px] * 4
        , it=0, return_sorted=False)
    tph_tes = numpy.tan((phase - phAB)) - tph_smooth

    phase_smooth = numpy.arctan(tph_smooth)
    phase_tes = numpy.arctan(tph_tes)

    #======================================================================
    # 3. Calculate the power dissipated in the TES and the TES 
    # (uncalibrated) resistance

    ptes_uncal = v_cal * i_cal * numpy.cos((phase - phAB))
    rtes_uncal = numpy.float32(ptes_uncal) / numpy.float32(i_cal**2)

    calr = calibration['Rn'][px] / mean(rtes_uncal[:10])

    #======================================================================
    # 4. Fit the normal and superconducting part of the IV curve

    slope_norm,int_norm,r_value_n,p_value_n, std_err_n = \
        stats.linregress(numpy.append(vlc_i[:20],0),numpy.append(i_cal[:20],0))

    fit_n = numpy.array([((x * slope_norm)+int_norm) for x in vlc_i])

    slope_super,int_super,r_value_s,p_value_s, std_err_s = \
        stats.linregress(numpy.insert(vlc_i[-1*10:-2],0,0),
            numpy.insert(i_cal[-1*10:-2],0,0))

    fit_s = numpy.array([((x * slope_super)+int_super) for x in vlc_i])

    #======================================================================
    # 5. Get the absolute value of the shunt and series impedance

    bn = slope_norm
    bs = slope_super
    bratio = bs/bn

    r = calibration['Rn'][px] / (bratio - 1) 

    Rseries = 0
    Rsh = r - Rseries

    #======================================================================
    # 6. Get TES parameters

    Ztes_Re = rtes_uncal * calr
    Ites = i_cal

    Vre = Ites * Ztes_Re

    Ptes = Ites * Vre

    Vim = Vre * numpy.tan(phase - phAB)

    Ztes_Im = Vim / Ites

    phtes_wl = numpy.arctan(Vim / Vre)
    Vteswl = Vre / numpy.cos((phtes_wl))

    Ttes = (Ptes/calibration['k'][px] + (Tbath*1.e-3)**calibration['n'][px])\
        **(1/calibration['n'][px])

    #======================================================================
    # Get TES IC

    log_ical = numpy.log10(i_cal)
    log_vr = numpy.log10(Vre)

    slope_log = (log_ical[0]-log_ical[-3])/(log_vr[0]-log_vr[-3])
    int_log = 0
    y = numpy.array([((x * slope_log)+int_log) for x in log_vr])
    y = y - (y[0]-log_ical[0])

    logiminy = numpy.array(log_ical - y)
    indmaxr = numpy.where((logiminy < numpy.nanmax(logiminy)*1.02) & \
        (logiminy > numpy.nanmax(logiminy)*0.98))
    Ic = numpy.mean(i_cal[indmaxr])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    filename = 'Ic_temp_' + str(Tbath)[:6] + 'mK_pixel_' + str(px)
    ax.plot(log_vr, logiminy, color='blue')
    ax.scatter(log_vr[indmaxr], logiminy[indmaxr], color='red')
    ax.plot(log_vr, numpy.array(logiminy*0.0))

    if numpy.count_nonzero(~numpy.isnan(logiminy)) > 0:
        logiminy[logiminy < 0] = numpy.nan
        ax.plot(log_vr, logiminy)

    ax.minorticks_on()
    ax.grid(True, which='both')
    ax.set_xlabel("log V [mV]")
    ax.set_ylabel("log I [uA]")
    ax.set_title(filename)
    fig.set_size_inches(9, 5)
    fig.savefig('data/Plots/Ic/'+filename+'.png', dpi=200)

    #======================================================================
    # Get TES Power

    iminvrn = i_cal - fit_n

    indmax = numpy.where(iminvrn == numpy.max(iminvrn))
    indmaxr = numpy.where((iminvrn < iminvrn.max()*1.02) & \
        (iminvrn > iminvrn.max()*0.98))

    itrans = i_cal[:indmax[0][-1]]

    indmin = numpy.where(itrans == numpy.min(itrans))
    indminr = numpy.where((itrans < itrans.min()*1.01) & \
        (itrans > itrans.min()*0.99))
    indhalfr = numpy.where((Ztes_Re < 0.51 * calibration['Rn'][px]) & \
        (Ztes_Re > 0.49 * calibration['Rn'][px]))

    Pmin = Ptes[indmin]
    Phalf = numpy.mean(Ptes[indhalfr])

    try:
        if indhalfr[0].max()>indminr[0].min()-5 and \
            indmaxr[0].max()>indminr[0].min()-5 and \
            indmaxr[0].min()<indminr[0].max()+5:

            rthrs = 0.03 * calibration['Rn'][px]
            indr = numpy.where(Ztes_Re > numpy.max(iminvrn))

            ic = numpy.mean(i_cal[indr[0].max()-1:indr[0].max()+1])
    except:
        pass

    Ptesmin = numpy.mean(Ptes[indminr])
    Pteshalf = numpy.mean(Ptes[indhalfr])

    #======================================================================
    # Calculate the derivative of the R-t curve (dR/dT)

    ztrsmth= lowess(Ztes_Re,Ttes, frac=0.2, it=0,return_sorted=False)
    ztismth= lowess(Ztes_Im,Ttes, frac=0.2, it=0,return_sorted=False)
    dZRedT = deriv(Ttes,ztrsmth)
    dZImdT = deriv(Ttes,ztismth)

    dZRedTsmooth=lowess(dZRedT,Ttes, frac=calibration['nsmooth'][px],
        it=0,return_sorted=False)
    dZImdTsmooth=lowess(dZImdT,Ttes, frac=calibration['nsmooth'][px],
        it=0,return_sorted=False)

    VreuV       = Vre*1e6
    VtesuVwl    = Vteswl*1e6
    ItesuA      = Ites*1e6
    PtespW      = Ptes*1e12
    Ztes_RemO   = Ztes_Re*1e3
    TtesmK      = Ttes*1e3

    output_filename = 'data/output/Caldata_original_temp_' + str(Tbath) + '_px' + \
        str(px) + '.txt'
    output_data = numpy.column_stack((v_raw, phase*180/numpy.pi, Ites, Vre, Ztes_Re, 
        Ptes, Ttes, vlc_i, phtes_wl*180/numpy.pi, vlc_q, Ztes_Im, 
        phase_tes*180/numpy.pi, dZRedT, dZImdT, dZRedTsmooth, dZImdTsmooth))
    output_header = '1:Vraw 2:Phase[deg] 3:Ites[A] 4:Vtes[V] 5:Ztes_Re[ohm] 6:Ptes[W] \
        7:Ttes[K] 8:Vlc_i[V] 9:Phase_wl[deg] 10:Vlc_q[V] 11: Ztes_im[ohm] \
        12:Phasetes_smoothed[deg] 13:dZRedT[ohm/K] 14:dZImdT[ohm/K] \
        15:dZRedT_smooth[ohm/K] 16:dZImdT_smooth[ohm/K]'
    numpy.savetxt(output_filename, output_data, header=output_header)

    IcTPmPh[j] = numpy.array([Tbath, Ic, Pmin[0], Phalf])

    filename = 'data/analyis_data/IcTPmPh_Px' + str(px) + '.txt'
    header = '|T (mK)|Ic (A)|Pmin (W)|Phalf (W)|'

    numpy.savetxt(filename, IcTPmPh, header=header)
    
    return IcTPmPh

def main():
    # import data from text file
    #data = numpy.genfromtxt('location_here.txt')

    # Assign data column to type
    #v_raw = data[:,0]
    #i_raw = data[:,1]
    #phase = data[:,2]

    # Pars data (check if imported pars file matches with data)
    calibration = {
    'RfbMiMf'   : pars.rfpx * pars.mfi,
    'Rn'        : pars.Rn,
    'n'         : pars.n,
    'k'         : pars.K,
    'vbcal'     : pars.gainvb,
    'voutcal'   : pars.trafo,
    'nsmooth'   : pars.Nsmooth
    }
    
    #return calibration

    # pixel number
    px = 1

if __name__ == '__main__':
    main()