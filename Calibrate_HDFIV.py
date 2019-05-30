"""
Calibration Script,

Reads the HDF5 file created by the Create_HDF_IV.py measurement script.

Creates and I-V and Phase-V graph of the measured data for each measured
pixel and saves the graph in the ./Data/Plots/IV_PhaseV folder for easy overview of
the measured data

Calibrates the measured data for each pixel and saves the result under a
output file in the ./Data/Output folder. During the callibration the program
also determines the most likely value of the Ic current and saves this value
at each temperature for each pixel together with the corresponding power to
an .txt file in the ./Data/Analyis_data folder

"""

# ==================================================================================
# Imports
import os
import numpy
import hdffileIV
import scipy.signal as signal
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from pylab import detrend_linear, mean, hanning, norm, window_hanning, window_none, detrend_mean, detrend_none
from scipy import interpolate
from scipy import stats

# ==================================================================================
# Import system parameters
import pars_run70 as pars

def read_data(filename, pxs, Num_Dsets, main_var):
    """Imports all data from HDF5 file
    
    Imports all data from the HDF5 measurement file and stores this data
    in a data array which is structured as:
    data[pixel_n[[variable], [ampl], [phase], [attr]],..]
    (variable is the data from the variable selected as the main
    variable in the measurement script)
    Saves list of attribute names in same ordeer as they are saved
    in the attr data list 
    
    Args:
        filename (str): name of the HDF5 file
        pxs (list): list of measured pixels
        Num_Dsets (int): the number of data sets in the HDF file
        main_var (str): main variable of the measurement
        
    Returns:
        data (array): All data from the HDF5 file (structured)
        attr_keys (list): names of the saved attributed
    
    """
        
    # Loads the HDF file
    hdf = hdffileIV.HdfFile(filename)

    # Creates data list in which all the data will be placed
    data = []
    # Creates four lists for each of the pixels which were measured, matching
    # with the four data types in the data (Variable data, Amplitude, Phase,
    # attribute data)
    for i, px in enumerate(pxs):
        var_col = []
        ampl_col = []
        phase_col = []
        attr_data = []

        # Creates data lists of the four data types for each of measurement set
        for j in range(Num_Dsets/len(pxs)):
            indx = (j * len(pxs)) + i
            indx += 1

            # Loads the measurement set data from the HDF5 file
            dset = hdf.get_iq_dset(0, px, main_var, indx)

            # Selects the data from the measurement set and loads it into the
            # corrosponding data type list
            var_col.append(dset[:,0])
            ampl_col.append(dset[:,1])
            phase_col.append(dset[:,2])
            attr_data.append(dset.attrs.values())

        # adds all the data from the pixel into the data list for later use
        data.append([var_col, ampl_col, phase_col, attr_data])

    # Saves the attribute keys which can be used to determine the location of
    # certain attribute data types in the attribute data list
    attr_keys = dset.attrs.keys()

    # Converts all the attribute keys to strings
    for i in range(len(attr_keys)):
        attr_keys[i] = str(attr_keys[i])

    return data, attr_keys

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

def deriv(x, y):
    """Calculates the derivative
    
    Calculate the derivative dY/dX of a sequence of data points (X,Y).
    The calculation is based on 3-point interpolation.
    
    Args:
        x (array): 1D numpy array of x values
        y (array): 1D numpy array of y values
        
    Returns:
        ret (array): 1D array of the dY/dX derivative of same length as 
            the original y array
    """

    n = len(x)
    assert n >= 3
    assert len(y) == n

    x   = x.astype(numpy.float64)

    dx  = x[1:n] - x[0:n-1]
    ddx = x[2:n] - x[0:n-2]

    ret = numpy.zeros(n)

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

def plot_mainvar_data(data, attr_keys, attr, pxs, show_plot=False):
    """Creates the IV and PhaseV plots of the measured data
    
    Takes the measured data and creates a plot of the main variable
    (usually the voltage) as function of the amplitude and the phase
    
    Args:
        data (array): data array from the read_data() function
        attr_keys (list): list of attribute names from read_data() 
            function
        attr (str): attribute name by which the plots are sorted 
            e.g. temperature
        pxs (list): list of measured pixels
        show_plot (bool): whether to open
        
    Returns:
        None: saves the created plots as a png file
        
    """

    # Checks if filepath to which data will be writen already exsists
    # If not then the filepath is created
    if not os.path.exists('data/Plots/IV_PhaseV'):
        os.makedirs('data/Plots/IV_PhaseV')

    # Looks for the index of the attribute by which the different data
    # sets are differentiated
    attr_indx = attr_keys.index(attr)

    # Loops through the different pixels which were measured
    for j, px in enumerate(pxs):
        # Selects the data from the pixel from the complete data list
        px_data = data[j]

        # Creates 2 figures with 2 plots each, the plots are positioned
        # On top of each other and the x axis between the 2 plots are
        # Linked
        fig, (ax11, ax21) = plt.subplots(2, sharex=True)
        fig2, (ax12, ax22) = plt.subplots(2, sharex=True)
        for k in range(len(px_data[0])):
            # dynamicly decides a color to the data so no two data sets
            # have the same color
            col_idx = (float(k+1)/float(len(px_data[0]))) * 255
            col_idx = int(numpy.floor(col_idx))

            color = plt.cm.jet(col_idx)

            # Creates an median filterd version of the data set
            fil_set1 = medfilt(px_data[1][k], 5)
            fil_set2 = medfilt(px_data[2][k], 5)

            label = px_data[3][k][attr_indx]
            if isinstance(label, float):
                if label < 95.:
                    label = float("{0:.3f}".format(label))

                    ax11.plot(px_data[0][k], px_data[1][k], linewidth=0.7, label=label,
                        color=color)
                    ax21.plot(px_data[0][k], px_data[2][k], linewidth=0.7, color=color)

                    ax12.plot(px_data[0][k], fil_set1, linewidth=0.7, label=label,
                        color=color)
                    ax22.plot(px_data[0][k], fil_set2, linewidth=0.7, color=color)

        ax11.legend(loc=1, ncol=4, framealpha=0.6, fontsize=7)
        ax12.legend(loc=1, ncol=4, framealpha=0.6, fontsize=7)

        file_name = 'data/Plots/data_px' + str(px) + '.png'
        file_name2 = 'data/Plots/data_px' + str(px) + '_medfilter.png'

        ax11.set_title('IV & phase measurement pixel' + str(px))
        ax12.set_title('IV & phase measurement pixel' + str(px))

        ax11.set_axisbelow(True)
        ax21.set_axisbelow(True)
        ax11.minorticks_on()
        ax11.grid(True, which='both')
        ax21.minorticks_on()
        ax21.grid(True, which='both')

        ax12.set_axisbelow(True)
        ax22.set_axisbelow(True)
        ax12.minorticks_on()
        ax12.grid(True, which='both')
        ax22.minorticks_on()
        ax22.grid(True, which='both')

        fig.set_size_inches(18.5, 10.5)
        fig.savefig(file_name, dpi=400)

        fig2.set_size_inches(18.5, 10.5)
        fig2.savefig(file_name2, dpi=400)

        if show_plot:
            plt.show()
        plt.show()
        plt.cla()

def calibrate_data(data, pxs, Num_Dsets, attr_keys, calibration):
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
    
    if not os.path.exists('data/output/'):
        os.makedirs('data/output/')
    if not os.path.exists('data/analyis_data/'):
            os.makedirs('data/analyis_data/')
    if not os.path.exists('data/Plots/Ic'):
        os.makedirs('data/Plots/Ic') 

    for i, px in enumerate(pxs):
        IcTPmPh = numpy.empty((Num_Dsets/len(pxs), 4,))
        IcTPmPh[:] = numpy.nan
        for j in range(Num_Dsets/len(pxs)):
            v_raw = data[i][0][j]
            i_raw = data[i][1][j]
            phase = data[i][2][j]

            i_raw_med = medfilt(i_raw, 7)
            phase_med = medfilt(phase, 7)

            bad_ind1 = numpy.where(numpy.abs((i_raw_med/i_raw) - 1) > .10)[0]
            bad_ind2 = numpy.where(numpy.abs((phase_med/phase) - 1) > .10)[0]
            bad_ind = list(set(numpy.append(bad_ind1, bad_ind2)))

            v_raw = numpy.delete(v_raw, bad_ind)
            i_raw = numpy.delete(i_raw, bad_ind)
            phase = numpy.delete(phase, bad_ind)


            Tbath = data[i][3][j][attr_keys.index('Measured_temp')]
            if isinstance(Tbath, float):
                if Tbath < 95.:

                    phase = phase * (numpy.pi/180) # convert phase to radiant

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

        print('==========================DONE===PX' +str(px) + '===========================')

def main():
    filename = '' 
    pxs = []
    Num_Dsets = 
    main_var = ''
    attr = ''

    calibration = {
    'RfbMiMf'   : pars.rfpx * pars.mfi,
    'Rn'        : pars.Rn,
    'n'         : pars.n,
    'k'         : pars.K,
    'vbcal'     : pars.gainvb,
    'voutcal'   : pars.trafo,
    'nsmooth'   : pars.Nsmooth
    }


    data, attr_keys = read_data(filename, pxs, Num_Dsets, main_var)

    plot_mainvar_data(data, attr_keys, attr, pxs)

    calibrate_data(data, pxs, Num_Dsets, attr_keys, calibration)

main()
