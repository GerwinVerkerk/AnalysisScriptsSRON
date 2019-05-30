#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:42:18 2019
Place this script in the folder where
there IV curve data is located.

@author: gerwinv
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import scipy
from scipy import optimize
import time
import os
import re
from scipy.interpolate import interp1d
import itertools
#import Calibrate_HDFIV
import calivac_niceplot_pwr
#import Calibrate_IV_TXT
import pars_run78 as pars

import pylab as plb
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

def find_nearest(array, value): 
    array = np.asarray(array); 
    idx = (np.abs(array - value)).argmin(); 
    return idx;


class UberCalibration():
    def __init__(self, pixel, text_phase_column, text_amplitude_column, text_bias_column):      
        self.make_plot_canvas()
        IVdataFiles = self.search_files()
        ResolutionData = self.getPixelResolution()
        
#        freq_transition = self.determineTransitionRegion(IVdataFiles, text_phase_column, 
#                                                         text_amplitude_column, text_bias_column,
#                                                         frequency_key = "freq", pixel = pixel)
#        
        freq_dico = self.determineDirectionCoefficient(IVdataFiles, self.ax4, text_bias_column, 
                                                       text_amplitude_column, text_phase_column,
                                                       pixel)
#        
#        
#
        SlopeResDat = self.slopeHeatmap(ResolutionData, freq_dico, axes = self.ax3, 
                              frequency_key = "freq(khz)",
                              voltage_key = "V_bias",
                              resolution_key = "fwhm(err)(eV)")
#
#==============================================================================
#         self.slopeVSresolution(data = SlopeResDat, axes = self.fig5ax)
#==============================================================================

        
#==============================================================================
#         print(freq_transition[1068.25])
#         transition_region = freq_transition[1068.25][0]-freq_transition[1068.25][1]
#         
#         
#         print("10% at:", transition_region*0.1+freq_transition[1068.25][1])
#         print("30% at:", transition_region*0.3+freq_transition[1068.25][1])
#==============================================================================

        self.make_IV_plot(IVdataFiles, 
                       X = text_bias_column,
                       Y = text_amplitude_column,
                       pixel=pixel,
                       axes = self.ax1,
                       plottype = 'IV',
                       remove_normal_offset=False
                       )
#        
        self.make_IV_plot(IVdataFiles, 
                   X = text_bias_column,
                   Y = text_phase_column,
                   pixel=pixel,
                   axes = self.ax2,
                   plottype = 'PHASE',
                   remove_normal_offset= True,
                   SlopeResData = SlopeResDat
                   )
        
#==============================================================================
#         self.resolutionHeatmap(ResolutionData,
#                                    axes = self.fig3ax,
#                                    frequency_key = "freq(khz)",
#                                    voltage_key = "V_bias",
#                                    resolution_key = "fwhm(err)(eV)")
#==============================================================================
    
#        self.make_RT_plot(IVdataFiles, ResolutionData, pixel, self.fig2ax, text_bias_column, text_amplitude_column, text_phase_column)  
        

        
        self.plot_nwa_data(self.read_nwa('nwaACB2.dat'), self.fig2ax, "blue")
        self.plot_nwa_data(self.read_nwa('nwaFB3.dat'), self.figduoax2, "green")


        self.frequencyVSresolution(ResolutionData, self.figduoax1, residualaxes = self.residualax ,fit = "filtered_polynomial")
        
#        self.temporarily(ResolutionData)
        
#        self.gainVSresolution(ResolutionData, nwaScanFB1, self.fig3ax, residualaxes = self.residualax3)
        
        
        plt.show()
        
    def chisqr(self, obs, exp, error):
        chisqr = 0
        for i in range(len(obs)):
            chisqr = chisqr + ((obs[i]-exp[i])**2)/(error[i]**2)
        return chisqr
        
    def plot_nwa_data(self, nwadata, axes, colour):
        axes.set_xlabel(r'$f$ (kHz)', fontsize = '40')
        axes.set_ylabel(r"$G$ (mV/mV)", fontsize = '40')
        axes.tick_params(labelsize = '40')
        axes.get_xaxis().get_major_formatter().set_useOffset(False)
        
        top_index = np.where(nwadata[1]==max(nwadata[1]))[0][0]
        top_value = max(nwadata[1])
        frequency_at_top = round(nwadata[0][top_index],2)
        
        xavg = sum(nwadata[0])/len(nwadata[0])
        yavg = sum(nwadata[1])/len(nwadata[1])
        
#==============================================================================
#         xdiffs = [abs(e[1] - e[0]) for e in itertools.permutations(nwadata[0], 2)]
#         ydiffs = [abs(e[1] - e[0]) for e in itertools.permutations(nwadata[1], 2)]
#         
#         xavg = sum(xdiffs)/len(xdiffs)
#         yavg = sum(ydiffs)/len(ydiffs)
#==============================================================================
        
        
        
        axes.axvline(x=frequency_at_top, color='k', linestyle='--')
        
        
#        xdist = frequency_at_top + xavg/2
#        ydist = max(nwadata[1]) - yavg*0
        xdist = frequency_at_top + 0.025
        ydist = max(nwadata[1])

        
        
        axes.annotate(r'$f_\textup{res}$ = '+str(frequency_at_top)+' (kHz)', 
            xy=(frequency_at_top, max(nwadata[1])), 
            xytext=(xdist, ydist), fontsize = '35',
            arrowprops = dict(facecolor='black', shrink=0.01))
        
        
        axes.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axes.grid(True)
                
        axes.plot(nwadata[0], nwadata[1], color=colour)



    def make_plot_canvas(self):
        self.fig2, self.fig2ax = plt.subplots(1,1)
        
        
#==============================================================================
#         self.fig3, self.axs3 = plt.subplots(2,1, sharex=True, gridspec_kw = {'height_ratios': [5,1]})
#         self.fig3.subplots_adjust(hspace=0)
#         self.fig3ax = self.axs3[0]
#         self.residualax3 = self.axs3[1]
#==============================================================================
        
        
        
#==============================================================================
#         self.fig4, self.fig4ax = plt.subplots(1,1)
#         self.fig5, self.fig5ax = plt.subplots(1,1)
#         self.fig6, self.fig6ax = plt.subplots(1,1)
#==============================================================================
        
        
        
        self.figduo, self.axs = plt.subplots(2,1, sharex=True, gridspec_kw = {'height_ratios': [5,1]})
        self.figduo.subplots_adjust(hspace=0)
        self.figduoax1 = self.axs[0]
        self.residualax = self.axs[1]
        self.figduoax2 = self.figduoax1.twinx()
        
        
        
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2,2)
        self.fig.subplots_adjust(hspace=0.5)
        
    def temporarily(self, ResolutionData):
        freqs = []
        bias = []
        data = []
        datarows = []
        
        for file in ResolutionData:
            bias_voltage = round(float(ResolutionData[file]["V_bias"]),2)
            bias_frequency = round(float(ResolutionData[file]["freq(khz)"]),2)
            res_and_error = ResolutionData[file]["fwhm(err)(eV)"].split('(')
            res = res_and_error[0]
            error = res_and_error[1]
            error = error.replace(')', "")
            try:
                res = float(res)
                error = float(error)
            except:
                res = None
                error = None
            
            
            if isinstance(res, float) and isinstance(error, float):
                data.append([bias_frequency, bias_voltage, res, error])
                
            
        
        sorted_data = sorted(data, key = lambda x: (float(x[0]), float(x[1])))
        
        
#        print(sorted_data)

        df = pd.DataFrame(sorted_data, columns = ['Freq', 'Vbias', 'Res', 'Err'])
        print(df)
    
    def checkPolynomial(self, xsort, ysort, order, axes, colour):
        p, res, _, _, _ = np.polyfit(xsort, ysort, order, full=True)
        
        f = np.poly1d(p)
        x_new = np.linspace(xsort[0], xsort[-1], 1000)
        y_new = f(x_new)
        
        
        res = []
        for idx, val in enumerate(ysort):
            xvalue = xsort[idx]
            
            closest_x = min(enumerate(x_new), key=lambda x: abs(x[1]-xvalue))
            index = closest_x[0]
            
            y_fit = y_new[index]
            
            difference = y_fit - val
            
            res.append(difference)
        
        if axes != None:
            if colour != None:
                axes.plot(x_new, y_new, color = colour)
            else:
                axes.plot(x_new, y_new)
        
        return (p,res)
        
    
    def checkPearsonCorrelation(self, xvalues, yvalues, axes, colour = None):
        pearson = scipy.stats.pearsonr(xvalues, yvalues)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(xvalues,yvalues)
        
        print("xvalues: ", xvalues)
        print("yvalues:", yvalues)
        print("slope: ", slope)
        print("intercept: ", intercept)
        line = slope*xvalues + intercept
        print(line)
        if colour != None:
            axes.plot(xvalues,line, color = colour)
        else:
            axes.plot(xvalues,line)
        
        return pearson

    def search_files(self, filetype = ".txt"):
        """Search for files that end with .txt and that contain the word 'IV' in the directory this script
        is located. After that, read out all the files and append their names and data to a dictionary.
        """
        files = [f for f in os.listdir('.') if os.path.isfile(f)]
        IVdataFiles = {}
        if filetype == ".txt":
            for file in files:
                if file.endswith(('.txt')) and "IV" in file:
                    IVdataFiles[file] = self.read_txt_IV_data_numpy(file)
            return IVdataFiles
        else:
            pass
    
    def read_txt_IV_data_numpy(self, file):
        data = np.genfromtxt(file, delimiter=" ")
        
        column1 = data[:,0]
        column2 = data[:,1]
        column3 = data[:,2]
        parameters = self.getPixelSettings(file)
        return {'columns': [column1,column2,column3], 'parameters': parameters}

    def read_nwa(self, file):
        data = np.genfromtxt(file, delimiter=" ")
        column1 = data[:,0]
        column2 = data[:,1]
        column3 = data[:,2]
        
        return [column1, column2, column3]

    def getPixelSettings(self, file):
        """This function reads all of the parameters contained in the IV curve file. These include the
        pixel frequency, bias, enz...."""
        
        textfile = open(file, "r")
        lines = textfile.readlines()
        lines = [x.strip() for x in lines]
        settings = []
        parameters = {}
        
        for line in lines:
            if "##" in line:
                line = line.replace("##", "")
                settings.append(line)
        
        headers = settings[0].split()

        for header in headers:
            x = headers.index(header)
            templist = []
            
            for setting in settings[1:]: 
                columns = setting.split()
                templist.append(columns[x])
                
            parameters[header] = templist
                
        return parameters
    
    def make_IV_plot(self, IVdataFiles, X, Y, pixel, axes, plottype, remove_normal_offset=True, SlopeResData = None):

        
        for filename in IVdataFiles.keys():
            data = IVdataFiles[filename]['columns']
            
            if remove_normal_offset == True:
                data[Y] = self.remove_offset(variable = data[Y], bias = data[X])
            else:
                pass
                
            header = filename
            
            
            axes.plot(data[X],data[Y], label=IVdataFiles[filename]['parameters']['freq'][pixel][:-3])
            
            
            if SlopeResData != None:
                Xslope = []
                Yslope = []
                freq = round(float(IVdataFiles[filename]['parameters']['freq'][pixel]), 2)
                for file in SlopeResData:
                    if SlopeResData[file]['freq(kHz)'] == freq:
                        
                        bias = SlopeResData[file]['V_bias']
                        closests_voltage = min(enumerate(data[X]), key=lambda x: abs(x[1]-bias)) 
                        indexbias = closests_voltage[0]
                        valuebias = closests_voltage[1]
                                               
                        Yvalue = data[Y][indexbias]
                        
                        Xslope.append(valuebias)
                        Yslope.append(Yvalue)
                        
                    else:
                        pass
                axes.scatter(Xslope, Yslope, marker = 'x', s = 350)
            
        leg = axes.legend(loc='upper right', title="Frequency (kHz)", prop={'size': '40'})
        leg.set_title('Frequency (kHz)',prop={'size':'40'})
        axes.set_xlabel("Bias Voltage (mV)", fontsize = '40')
        axes.tick_params(labelsize = '40')
        
        if plottype == 'IV':
            axes.set_ylabel("Fullscale (-)", fontsize = '40')
        elif plottype == 'PHASE':
            axes.set_ylabel("Phase (deg)", fontsize = '40')

        
    def make_RT_plot(self, IVdataFiles, ResolutionData, pixel, axes, text_bias_column, text_amplitude_column, text_phase_column, normalized=True):
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
        
        
        for file in IVdataFiles:
            for data in IVdataFiles[file]:
                xtsave = calivac_niceplot_pwr.calivac(
                                             ivfile     = file, 
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
                axes.plot(xtsave[:,0],norm, label = round(float(IVdataFiles[file]['parameters']['freq'][pixel]),2))
                yas = norm
                axes.set_ylabel("R/Rn (-)", fontsize = '40')
            else:
                axes.plot(xtsave[:,0],xtsave[:,4], label = round(float(IVdataFiles[file]['parameters']['freq'][pixel]),2))
                yas = xtsave[:,4]
                axes.set_ylabel("Ztes (Ohm)", fontsize = '40')
            
            leg = axes.legend(loc='upper right', title="Frequency (kHz)", prop={'size': '40'})
            leg.set_title('Frequency (kHz)',prop={'size':'40'})
            axes.set_xlabel("Bias Voltage (mV)", fontsize = '40')
            
            axes.tick_params(labelsize = '40')
            
            xas = xtsave[:,0]
            
            Xlist = []
            Ylist = []
            
            for file in ResolutionData:
                print('ResolutionData expects IVfile to have double values. Pleas fix line 272')
                
                bias = float(ResolutionData[file]['V_bias'])
#                biasindex = (xas >= bias-0.1)*(xas <= bias+0.1)
                closests_bias = min(enumerate(xas), key=lambda x: abs(x[1]-bias)) 
                biasindex = closests_bias[0]
                
                valueY = yas[biasindex]
                valueX = xas[biasindex]
            
                Xlist.append(valueX)
                Ylist.append(valueY)
                
            axes.axvspan(min(Xlist), max(Xlist), color='red', alpha=0.2)    
            

            axes.plot(Xlist, Ylist)
        
    def remove_offset(self, variable, bias, newoffset=0, begin_normal=300, end_normal=600):
        normal_offset = []
        variable = [float(i) for i in variable]
        bias = [float(i) for i in bias]
        
        for value in bias:
            if value > begin_normal and value < end_normal:
                index = bias.index(value)
                normal_offset.append(variable[index])
            else:
                pass
        average_offset = sum(normal_offset)/len(normal_offset)
        
        variable[:] = [value - (average_offset-newoffset) for value in variable]
        calibrated_variable = variable
        
        return calibrated_variable
        
        
    def getPixelResolution(self):
        PATH = "."
        files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PATH) for f in filenames if os.path.splitext(f)[1] == '.txt']
        files = [file for file in files if "Eventpar_Run" in file]
        data = []
        
        for file in files:
            temp = []
            f = open(file)
            for x,line in enumerate(f):
                if x > 2:
                    temp.append(line)
            temp = [x.strip() for x in temp]
            temp.remove('')
            for row in temp:
                index = temp.index(row)
                temp[index] = row.split()
            
            temp.append(file)
            temp = [x for x in temp if x != []]
            data.append(temp)
            
        dicts = []
        ResolutionData = {}
        for results in data:
            performance = dict(zip(results[0],results[1]))
            ResolutionData[results[-1]] = performance
            
            dicts.append(performance)
            
        return ResolutionData
    
    def resolutionHeatmap(self, ResolutionData, axes, frequency_key, voltage_key, resolution_key):
        
        super_dict = {}
        for file in ResolutionData:
            for k, v in ResolutionData[file].items():
                if super_dict.get(k) is None:
                    super_dict[k] = []
                super_dict[k].append(v)
        
        frequency = super_dict[frequency_key]
        frequency = [float(i) for i in frequency]
        
        voltage = super_dict[voltage_key]
        voltage = [float(i) for i in voltage]
        
        resolution = super_dict[resolution_key]
        resolution = [x[:4] for x in resolution]
        resolution = [float(i) for i in resolution]
        
        x = voltage
        y = frequency
        z = resolution
        
        df = pd.DataFrame.from_dict(np.array([x,y,z]).T)
        df.columns = ['X_value','Y_value','Z_value']
        df['Z_value'] = pd.to_numeric(df['Z_value'])
        pivotted= df.pivot('Y_value','X_value','Z_value')

        xlabels = np.unique(x)
        ylabels = np.unique(y)
        
        im = axes.imshow(pivotted,cmap='inferno',vmin=min(z), vmax=3)
        
        axes.figure.colorbar(im, ax=axes, label="Resolution (eV)")
        
        
        
        axes.set_xticks(np.arange(len(xlabels)))
        axes.set_yticks(np.arange(len(ylabels)))
        axes.set_xticklabels(xlabels)
        axes.set_yticklabels(ylabels)
        
        
        
        for tick in axes.xaxis.get_major_ticks():
            tick.label.set_fontsize(9)
                
        for tick in axes.yaxis.get_major_ticks():
            tick.label.set_fontsize(9)
        
        for i in range(pivotted.shape[0]):
            for j in range(pivotted.shape[1]):
                axes.text(j,i, pivotted.iat[i,j] ,ha="center", va="center", color="w", fontsize=12)
        

    def slopeHeatmap(self, ResolutionData, freq_dico, axes, frequency_key, voltage_key, resolution_key):
        """This"""
        
        
        x = []
        y = []
        z = []
        
        SlopeResData = {}
        
        for file in ResolutionData:
            frequency = float(ResolutionData[file][frequency_key])
            bias = float(ResolutionData[file][voltage_key])
            res = ResolutionData[file][resolution_key]
            
            closests_frequency = min(enumerate(freq_dico.keys()), key=lambda x: abs(x[1]-frequency))        

            info = freq_dico[closests_frequency[1]]
            
            closests_voltage = min(enumerate(info[0]), key=lambda x: abs(x[1]-bias)) 
            
            slope = info[1][closests_voltage[0]]
            
            
            
            SlopeResData[file] = {'freq(kHz)': frequency, 'V_bias': bias, 'slope': slope, 'res': res}
            
            x.append(bias)
            y.append(frequency)
            z.append(slope)

        df = pd.DataFrame.from_dict(np.array([x,y,z]).T)
        df.columns = ['X_value','Y_value','Z_value']
        df['Z_value'] = pd.to_numeric(df['Z_value'])
        pivotted= df.pivot('Y_value','X_value','Z_value')

        xlabels = np.unique(x)
        ylabels = np.unique(y)
        
        im = axes.imshow(pivotted,cmap='inferno',vmin=min(z), vmax=max(z))
        
        axes.figure.colorbar(im, ax=axes, label="Slope (rad)")
        
#        axes.figure.colorbar.set_label('# of contacts', rotation=270)
        
        axes.set_xticks(np.arange(len(xlabels)))
        axes.set_yticks(np.arange(len(ylabels)))
        axes.set_xticklabels(xlabels)
        axes.set_yticklabels(ylabels)
        
        
        
        for tick in axes.xaxis.get_major_ticks():
            tick.label.set_fontsize(9)
                
        for tick in axes.yaxis.get_major_ticks():
            tick.label.set_fontsize(9)
        
        for i in range(pivotted.shape[0]):
            for j in range(pivotted.shape[1]):
                axes.text(j,i, round(pivotted.iat[i,j],2) ,ha="center", va="center", color="w", fontsize=10)
        
        axes.set_xlabel("Bias Voltage (mV)")
        axes.set_ylabel("Bias Frequency (kHz)")
        
        return SlopeResData

    def slopeVSresolution(self, data, axes):
        x = []
        y = []
        
        combined_dict = {}
        
        for file in data:
            freq = data[file]['freq(kHz)']
            if combined_dict.get(freq) is None:
                combined_dict[freq] = {}
                
            V_bias = float(data[file]['V_bias'])
            slope = data[file]['slope']
            
            res_and_error = data[file]['res'].split('(')
            res = res_and_error[0]
            error = res_and_error[1]
            error = error.replace(')', "")
            
            try:
                res = float(res)
                error = float(error)
            except:
                res = None
                error = None
                
            combined_dict[freq][V_bias] = [res, error, slope]
        
        
        
        for frequency in combined_dict.keys():
            x = []
            y = []
            z = []
            for bias in combined_dict[frequency]:
                y.append(combined_dict[frequency][bias][0])
                x.append(combined_dict[frequency][bias][2])
                z.append(combined_dict[frequency][bias][1])
                
            if not (None in x or None in y or None in z):
                xsort = np.sort(x)
                ysort = []
                zsort = []
                
                for value in xsort:
                    index = x.index(value)
                    ysort.append(y[index])
                    zsort.append(z[index])

                
                axes.errorbar(xsort, ysort, zsort, 0.005, fmt='o', ls='', label = frequency, markersize=5, capsize=3)

        for value in zsort:
            print(value)
                
        axes.set_xlabel("Slope (deg/mV)", fontsize = '40')
        axes.set_ylabel("Resolution (eV)", fontsize = '40')
        axes.tick_params(labelsize = '40')
        leg = axes.legend(loc='upper right', prop={'size': '40'})
        leg.set_title('Frequency (kHz)',prop={'size':'40'})
        
        
        pearson = self.checkPearsonCorrelation(xsort, ysort, axes)
        print("------------")
        string = "Pearson Correlation", pearson
        print(string)


    def gainVSresolution(self, data, nwadata, axes, residualaxes = None, fit = None):
        maxResolution = 4   #Define maximum resolution to remove shootouts
        super_dict = {}
        for file in data:
            res_and_error = data[file]['fwhm(err)(eV)'].split('(')
            res = res_and_error[0]
            error = res_and_error[1]
            error = error.replace(')', "")
            
            try:
                res = float(res)
                error = float(error)
            except:
                res = None
                error = None
                
            frequency = float(data[file]['freq(khz)'])
            bias = float(data[file]['V_bias'])
            
            base = 10   #Bin size for bias voltages
            
            rounded = base * round(bias/base)
            
            if rounded in super_dict:
                super_dict[rounded].append([frequency, res, error])
            elif rounded not in super_dict:
                super_dict[rounded] = []
                super_dict[rounded].append([frequency, res, error])
            
        print(super_dict)
        
        for bias_voltage in super_dict:
            check_bias = bias_voltage
            x = []
            y = []
            z = []
            for fre in super_dict[bias_voltage]:
                frequency_match = min(enumerate(nwadata[0]), key=lambda x: abs(x[1]-fre[0])) 
                gain = nwadata[1][frequency_match[0]]
                
                if fre[1] != None:
                    if fre[1] < maxResolution:
                        x.append(gain)
                        y.append(fre[1])
                        z.append(fre[2])
                    else:
                        pass
                
            if None in x or None in y or None in z:
                print("None value found!")
            elif check_bias == bias_voltage:
                xsort = np.sort(x)
                ysort = []
                zsort = []
                for value in xsort:
                    index = x.index(value)
                    ysort.append(y[index])
                    zsort.append(z[index])
                
                plot = axes.errorbar(xsort, ysort, yerr=zsort, xerr=0, fmt='o', ls='', label = bias_voltage, markersize=5, capsize=3)
                colour = plot[0].get_color()
                    
                
                
                polynomial = self.checkPolynomial(xsort, ysort, 1, axes, colour)
                residuals = polynomial[1]
                    
                if residualaxes != None:
                    measured_values = ysort
                    expected_values = [ysort[i]+residuals[i] for i in range(len(ysort))]
                    error = zsort
                    
                    chisqr = self.chisqr(measured_values, expected_values, error)
                    
#                    print('----')
#                    print(colour)
#                    print(chisqr)
                    deg_frd = len(measured_values)-2
                    print(deg_frd)
#                    print('----')
                    
                    print(chisqr/deg_frd)
                    
                    residualaxes.plot(xsort,residuals,'or', color = colour)
                    residualaxes.set_xlabel(r'$G$ (mV/mV)', fontsize = '40')
                    residualaxes.set_ylabel(r"$\zeta E$ (eV)", fontsize = '40')
                    residualaxes.tick_params(labelsize = '40')
                    residualaxes.grid(True)
                    
                    chi_squared = scipy.stats.chisquare(measured_values, expected_values, 0)
                    
                    
                    pearson =  self.checkPearsonCorrelation(xsort,ysort, axes, colour)
                    print("The pearson correlation coefficient is {}".format(pearson))
                    print("Chi squared is {}".format(chi_squared))
#                    
#                    print(xsort)
#                    print(ysort)
            
            
        
        
            
            
            
            
        axes.set_xlabel(r'$G$ (mV/mV)', fontsize = '40')
        axes.set_ylabel(r"$\Delta E$ (eV)", fontsize = '40')
        axes.tick_params(labelsize = '40')
        leg = axes.legend(loc='upper right', prop={'size': '40'})
        # bbox_to_anchor=(0.20, 1.0),
        leg.set_title(r"$V_\textup{{bias}}$ (mV)",prop={'size':'40'})
        
        axes.grid(True)
        
        
        
        
    def gaus(self, x, a, x0, sigma, C):
        return C + (a*exp(-(x-x0)**2/(2*sigma**2)))
    

    def frequencyVSresolution(self, data, axes, residualaxes = None, fit = None):
        maxResolution = 4   #Define maximum resolution to remove shootouts
        super_dict = {}
        for file in data:
            res_and_error = data[file]['fwhm(err)(eV)'].split('(')
            res = res_and_error[0]
            error = res_and_error[1]
            error = error.replace(')', "")
            
            try:
                res = float(res)
                error = float(error)
            except:
                res = None
                error = None
                
            frequency = float(data[file]['freq(khz)'])
            bias = float(data[file]['V_bias'])
            
            base = 10   #Bin size for bias voltages
            
            rounded = base * round(bias/base)
            
            if rounded in super_dict:
                super_dict[rounded].append([frequency, res, error])
            elif rounded not in super_dict:
                super_dict[rounded] = []
                super_dict[rounded].append([frequency, res, error])
            
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        markers = ["o","x","s"]
        set_marker_size = 10
        
        
        i = 0
        for bias_voltage in super_dict:
            
            if bias_voltage != 120 and False:
                continue
            
            check_bias = bias_voltage #Assigning a specific value to check_bias allows to only plot 1 bias voltage.
            x = []
            y = []
            z = []
            for fre in super_dict[bias_voltage]:
                
                if fre[1] != None:
                    if fre[1] < maxResolution:
                        x.append(fre[0])
                        y.append(fre[1])
                        z.append(fre[2])
                    else:
                        pass
                
            if None in x or None in y or None in z:
                print("None value found!")
            elif check_bias == bias_voltage:
                xsort = np.sort(x)
                ysort = []
                zsort = []
                for value in xsort:
                    index = x.index(value)
                    ysort.append(y[index])
                    zsort.append(z[index])
                marker = markers[i]
                if fit != "filtered_polynomial" and fit != "gaussian":
                    plot = axes.errorbar(xsort, ysort, yerr=zsort, xerr=0, fmt=marker, ls='', label = bias_voltage, markersize=set_marker_size, capsize=3)
                    colour = plot[0].get_color()
                else:
                    colour = colours[i]

                i = i + 1
                    
                if fit == "pearson":
                    pearson =  self.checkPearsonCorrelation(xsort,ysort,axes, colour)
                    plot.set_label(str(bias_voltage) + " " + str(round(pearson[0],2)))
                elif fit == "polynomial":
                    polynomial = self.checkPolynomial(xsort, ysort, 2, axes, colour)
                    
                    residuals = polynomial[1]
                    residuals = [ -x for x in residuals]
                    
                    if residualaxes != None:

                        measured_values = ysort
                        expected_values = [ysort[i]+residuals[i] for i in range(len(ysort))]
                        error = zsort
                        
                        chisqr = self.chisqr(measured_values, expected_values, error)

                        deg_frd = len(measured_values)-3

                        residualaxes.plot(xsort,residuals,'or', color = colour)
                        residualaxes.set_xlabel(r'$f$ (kHz)', fontsize = '40')
                        residualaxes.set_ylabel(r"$\zeta E$ (eV)", fontsize = '40')
                        residualaxes.tick_params(labelsize = '40')
                        residualaxes.grid(True)
                        
                        chi_squared = scipy.stats.chisquare(measured_values, expected_values, 0)
                        
                        print("-------")
                        print(chisqr)
                        print(deg_frd)
                        print(colour)
                        print("-------")
                elif fit == "data_fit":
                    x1 = xsort
                    y1 = ysort
                    fit1 = np.polyfit(x1, y1, 1)  # linear
                    fit2 = np.polyfit(x1, y1, 2)  # quadratic
                    fit3 = np.polyfit(x1, y1, 3)  # cubic
                    
                    v1 = np.polyval(fit1, x1)
                    v2 = np.polyval(fit2, x1)
                    v3 = np.polyval(fit3, x1)
                    
                    axes.plot(xsort, v1, label = "linear fit", color = colour)
                    axes.plot(xsort, v2, label = "quadratic fit", color = colour)
                    axes.plot(xsort, v3, label = "cubic fit", color = colour)
                    
                elif fit == "filtered_polynomial":
                    polynomial = self.checkPolynomial(xsort, ysort, 2, None, colour)
                    residuals = polynomial[1]
                    
                    residuals = [ -x for x in residuals]
                    
                    measured_values = ysort
                    expected_values = [ysort[i]-residuals[i] for i in range(len(ysort))]
                    error = zsort
                    
                    chisqr = self.chisqr(measured_values, expected_values, error)
                    deg_frd = len(measured_values)-3
                    
                    residualaxes.errorbar(xsort,residuals, yerr=zsort, color = colour, 
                                       fmt = marker, ls = "", markersize=set_marker_size)
#                    residualaxes.plot(xsort,residuals,'or', color = colour)
                    residualaxes.set_xlabel(r'$f$ (kHz)', fontsize = '40')
                    residualaxes.set_ylabel(r"$\zeta E$ (eV)", fontsize = '40')
                    residualaxes.tick_params(labelsize = '40')
                    residualaxes.grid(True)
                    residualaxes.set_ylim(-0.5, 0.5)
                    
                    standard_deviation = np.std(residuals)
                    sigma = standard_deviation*3
                    
                    """Obtaind values bigger than one sigma"""
                    print(residuals)
                    
                    residuals = [abs(number) for number in residuals]
                    
                    print(residuals)
                    
                    indices = np.nonzero(residuals>abs(sigma))
                    filtered_measured_values = np.delete(measured_values, indices)
                    removed_measured_values = np.take(measured_values, indices)
                    
                    x_filtered = np.delete(xsort, indices)
                    y_filtered = filtered_measured_values
                    z_filtered = np.delete(zsort, indices)
                    
                    x_removed = np.take(xsort, indices)
                    y_removed = removed_measured_values
                    z_removed = np.take(zsort, indices)
                    
                    
                    """Plot the filtered values"""
                    axes.errorbar(x_filtered, y_filtered, yerr=z_filtered, xerr=0, fmt=marker, 
                                  ls='', label = bias_voltage, markersize=set_marker_size, 
                                  capsize=3, color = colour)
                    
                    """Plot the removed values in grey"""
                    if x_removed and y_removed:
                        axes.errorbar(x_removed, y_removed, yerr=z_removed, xerr=0, fmt=marker, 
                                      ls='', label = "rm {}".format(bias_voltage), markersize=set_marker_size, 
                                      capsize=3, ecolor='grey', color = 'grey')
                    
                    """Make fit through filtered data"""
                    polynomial = self.checkPolynomial(x_filtered, y_filtered, 2, axes, colour)
                    residuals = polynomial[1]
                    
                    """Check chi-squared value of filtered data"""
                    measured_values = y_filtered
                    expected_values = [y_filtered[i]+residuals[i] for i in range(len(y_filtered))]
                    error = z_filtered
                    
                    chisqr = self.chisqr(measured_values, expected_values, error)
                    deg_frd = len(measured_values)-3
                    
                    print("-------")
                    print(chisqr)
                    print(deg_frd)
                    print(colour)
                    print("-------")
                elif fit == "gaussian":
                    popt, pcov = curve_fit(self.gaus, xsort, ysort, p0=[100,1068.25,0.5,min(ysort)])
                    
#                    axes.errorbar(xsort, ysort, 
#                                  yerr=zsort, 
#                                  xerr=0, fmt=marker, ls='', 
#                                  label = bias_voltage, 
#                                  markersize=set_marker_size, 
#                                  capsize=3, color = colour)
        
                    residuals = ysort - self.gaus(xsort, *popt)
                    
                    measured_values = ysort
                    expected_values = [ysort[i]-residuals[i] for i in range(len(ysort))]
                    error = zsort
                    
#                    chisqr = self.chisqr(measured_values, expected_values, error)
#                    deg_frd = len(measured_values)-4

                    """remove large outliers"""
                    standard_deviation = np.std(residuals)
                    sigma = standard_deviation*3
                    
                    indices = np.nonzero(residuals>abs(sigma))
                    filtered_measured_values = np.delete(measured_values, indices)
                    removed_measured_values = np.take(measured_values, indices)
                    
                    x_filtered = np.delete(xsort, indices)
                    y_filtered = filtered_measured_values
                    z_filtered = np.delete(zsort, indices)
                    
                    x_removed = np.take(xsort, indices)
                    y_removed = removed_measured_values
                    z_removed = np.take(zsort, indices)
                    
                    print("removed!! \n\n")
                    print(y_removed)
                    
                    """Plot the filtered values"""
                    axes.errorbar(x_filtered, y_filtered, yerr=z_filtered, xerr=0, fmt=marker, 
                                  ls='', label = bias_voltage, markersize=set_marker_size, 
                                  capsize=3, color = colour)
                    
                    """Plot the removed values in grey"""
                    if x_removed and y_removed:
                        axes.errorbar(x_removed, y_removed, yerr=z_removed, xerr=0, fmt=marker, 
                                      ls='', label = "rm {}".format(bias_voltage), markersize=set_marker_size, 
                                      capsize=3, ecolor='grey', color = 'grey')
                    
                    """Make fit through filtered data"""
                    popt, pcov = curve_fit(self.gaus, xsort, ysort, p0=[100,1068.25,0.5,min(ysort)])
                    residuals = ysort - self.gaus(xsort, *popt)
                    
                    axes.plot(xsort, self.gaus(xsort,*popt), 'ro:',
                              color = colour, marker = "")
                    
                    residualaxes.errorbar(xsort,residuals, yerr=zsort, color = colour, 
                                       fmt = marker, ls = "", markersize=set_marker_size)
                    residualaxes.set_xlabel(r'$f$ (kHz)', fontsize = '40')
                    residualaxes.set_ylabel(r"$\zeta E$ (eV)", fontsize = '40')
                    residualaxes.tick_params(labelsize = '40')
                    residualaxes.grid(True)
                    residualaxes.set_ylim(-0.5, 0.5)
                    
                    
                    
                    """Check chi-squared value of filtered data"""
                    measured_values = y_filtered
                    expected_values = self.gaus(x_filtered, *popt)
                    
                    error = z_filtered
                    
                    chisqr = self.chisqr(measured_values, expected_values, error)
                    deg_frd = len(measured_values)-4
                    
                    
                    print(residuals)
                    
                    print("-------")
                    print(chisqr)
                    print(deg_frd)
                    print(colour)
                    print("-------")
                    
                else:
                    pass
                        
        
            
                        
        
                    
                
            
            
            
        axes.set_xlabel(r'$f$ (kHz)', fontsize = '40')
        axes.set_ylabel(r"$\Delta E$ (eV)", fontsize = '40')
        axes.tick_params(labelsize = '40')
#        leg = axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),&nbsp; shadow=True, ncol=2)
        leg = axes.legend(loc='upper left', bbox_to_anchor=(0.15, 1.0), prop={'size': '40'})
#        leg = axes.legend(loc='upper right', prop={'size': '40'})
        leg.set_title(r"$V_\textup{{bias}}$ (mV)",prop={'size':'40'})
        axes.grid(True)
        
        
        
        
        
        
        
#        axes.get_xaxis().get_major_formatter().set_useOffset(False)

        
                
    def determineTransitionRegion(self, IVdataFiles, text_phase_column, text_amplitude_column, text_bias_column,
                                  frequency_key, pixel):
        
        freq_transition = {}
        
        for file in IVdataFiles:
            
            amplitude = IVdataFiles[file]['columns'][text_amplitude_column]
            bias = IVdataFiles[file]['columns'][text_bias_column]
            freq = round(float(IVdataFiles[file]['parameters']['freq'][pixel]),2)
            
            
            amplitude = [float(i) for i in amplitude]
            bias = [float(i) for i in bias]

            end_index = amplitude.index(max(amplitude))
            start_index = amplitude[:end_index].index(min(amplitude[:end_index]))
            
            start = bias[start_index]
            end = bias[end_index]

            freq_transition[freq] = [start, end]
            
        return freq_transition

        
        

    def determineDirectionCoefficient(self, IVdataFiles, axes, text_bias_column, text_amplitude_column, text_phase_column,
                       pixel):
        
        freq_dico = {}
        
        for file in IVdataFiles:
            
            x =  bias = IVdataFiles[file]['columns'][text_bias_column]
            y = phase = IVdataFiles[file]['columns'][text_phase_column]
            
            uniqueXnump = np.unique(bias, True)
            uniqueY = []
            
            for indice in uniqueXnump[1]:
                uniqueY.append(phase[indice])
            
            uniqueX = uniqueXnump[0]
            
            dy = np.diff(uniqueY)
            dx = np.diff(uniqueX)
            
            a = dy/dx
            
            freq = round(float(IVdataFiles[file]['parameters']['freq'][pixel]),2)
            
            axes.plot(uniqueX[:-1],a, label=freq)
            
            
            freq_dico[freq] = [uniqueX, a]
        axes.legend(loc='upper right')
        axes.legend(loc='upper right', title="Frequency (kHz)")
        
        return freq_dico
        
        
        
calibrate = UberCalibration(pixel = 0, text_phase_column = 2, text_amplitude_column = 1, text_bias_column = 0)
