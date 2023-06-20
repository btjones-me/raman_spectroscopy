import glob, os
import numpy as np
import csv
from matplotlib import pyplot
import scipy.signal
from scipy.optimize import curve_fit
import peakutils
from lmfit.models import PseudoVoigtModel
import math
import traceback
from collections import OrderedDict 

path_to_folder = "..\\CZTS_data\\CZTS_111116\\B"
samples_to_analyse = ['21', '22', '23', '24', '25', '26', '27']


def import_data(samples_to_analyse, path_to_folder=path_to_folder):
    dict_container = OrderedDict()
    for sample_id in samples_to_analyse:
        dict_container[sample_id] = []
    # print(dict_container)

    for sample_id in samples_to_analyse:
        path = path_to_folder + sample_id + "\\*.txt"
        # print(path)
        for filepath in glob.glob(path):
            # print(filepath)
            data_from_file = np.genfromtxt(filepath)
            dict_container[sample_id].append(data_from_file)
    # print(np.shape(dict_container['21'][0]))  # First data set from sample B21 file 0
    # print(np.shape(data_container['21'][1][:,0]))  # xs from file 1 of the sample B21
    # print(np.shape(data_container['21'][0][:,1]))  # ys from file 0
    return dict_container

def gaussian(x, height, center, width, offset):
    ##Not in use
    return height*np.exp(-(x - center)**2/(2*width**2)) + offset

def lorentzian(x, amp, ctr, wid):
    return amp*wid**2/((x-ctr)**2+wid**2)

def func(x, *params):
    ''' *params of the form [center, amplitude, width ...] '''
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        # y = y + amp * np.exp( -((x - ctr)/wid)**2)
        y = y + lorentzian(x, amp, ctr, wid)
    return y

def fit_lorentzians(guess, func, x, y):
    # Uses scipy curve_fit to optimise the lorentzian fitting
    sigma = [2] * len(y)
    popt, pcov = curve_fit(func, x, y, p0=guess, maxfev=14000, sigma=sigma)
    print('popt:', popt)
    fit = func(x, *popt)
    # pyplot.plot(x, y)
    # pyplot.plot(x, fit , 'r-')
    return (popt, fit)
    # pyplot.show()


def find_peaks_peakutils(xs, ys, y_threshold=0.05, ys_min_separation=10, maxfev=20000, print=False):
    # Obsolete, not used
    indexes = peakutils.indexes(ys, thres=y_threshold, min_dist=ys_min_separation)  # Threshold is a percentage, min dist is minimum distance between peaks. I'm guessing it's in units of points, ie 3 points
    peaks_x = peakutils.peak.interpolate(xs, ys, ind=indexes, width=2)
    if(print):
        print('*************PEAKUTILS**************')
        print('Peak_pos= ', indexes, '\nPeaks= ', xs[indexes], ys[indexes])
        print('peaks_x: ', peaks_x)
        print('************************************')


    # pyplot.plot(xs[indexes], ys[indexes], linestyle='', marker='x', markersize='12', color='blue') #Plot peaks
    # plot vertical lines where peaks have been identified
    for peak in peaks_x:
        pyplot.axvline(x=peak, linestyle='--')

    return peaks_x

def remove_baseline(xs, ys, return_xs=False, degree=5): # Returns just ys unless otherwise specified
    # Remove baseline with polynomial
    y2 = ys + np.polyval([0.001,-0.08,degree], xs)
    # y2 = ys
    base = peakutils.baseline(y2, 2)
    if return_xs:
        return (xs, y2-base)
    else:
        return y2-base

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def find_peaks_scipy(xs, ys, widths=np.arange(1,50, 0.5)):
    # Not perfect but seems to find the most obscure ones, I use this one
    peaks = scipy.signal.find_peaks_cwt(ys, widths)#Problem
    # pyplot.plot(xs[peaks], ys[peaks], marker='+', linestyle='', markersize='12')
    print('*************PEAKSCIPY**************')
    print('From scipy: \nPeak_pos= ', peaks, '\nPeaks= ', xs[peaks], '\nIntensities= ',ys[peaks])
    print('************************************')

    return peaks 

def find_peaks_sri(d, th):
    # Too sensitive to noise - not used
    '''
    returns bool array with the same shape as `d` with elemets at the position of local maxima in `d` are set to `True`
    this function simply checks if the neighbouring elements are smaller or equal and is __sensitive to noise__
    '''
    np.r_[True, d[1:] >= d[:-1]] & np.r_[d[:-1] > d[1:], True]

    # th = 0 # use threshold of 0 mV for AP detection. the AP waveform is not very noise so the below works
    V_maxima = (np.r_[True, d[1:] >= d[:-1]] & np.r_[d[:-1] > d[1:], True] & (d>th))
    indices = V_maxima.nonzero()[0]

    return indices

def get_highest_n_from_list(a, n):
    # Returns the highest peaks found by the peak finding algorithms
    return sorted(a, key=lambda pair: pair[1])[-n:]


def get_highest_n_peaks_scipy(xs, ys, n, th=0.25):
    smooth_indexes_scipy = find_peaks_scipy(xs, ys)
    ##take the highest 'n' smooth peaks 
    peak_indexes_xs_ys = np.asarray([list(a) for a in list(zip(xs[smooth_indexes_scipy], ys[smooth_indexes_scipy]))])
    return get_highest_n_from_list(peak_indexes_xs_ys, n)


def predict_and_plot_lorentzians(xs, ys, n_peaks_to_find=5):
    n_peaks = np.asarray(get_highest_n_peaks_scipy(xs, ys, n_peaks_to_find))

    pyplot.plot(n_peaks[:,0], n_peaks[:,1], ls='', marker='x', markersize=10) ##This is plotting the peak positions, the xs
    # pyplot.ylabel("Counts")
    # pyplot.xlabel("Raman Shift/ Wavenumber ($cm^{-1}$)")

    print(n_peaks)
    guess = []

    for idx, xs_ys in enumerate(n_peaks):
        guess.append(xs_ys[0]) #ctr
        guess.append(xs_ys[1]) #amp
        guess.append(10) #width ###This could be improved by estimating the width first for a better fit
    print('Fit Guess: ', guess)
    # guess.append(500); guess.append(0.25); guess.append(500) #Broad lorenztian

    
    params, fit = fit_lorentzians(guess, func, xs, ys) ###params is the array of gaussian stuff, fit is the y's of lorentzians

    return (params, fit, ys, n_peaks)


def main_predict_fit():
    # The following code runs through each repeated measurement from all of the samples
    # and attempts to fit lorentzians to the data.
    dict_container = import_data(samples_to_analyse) #global variable at present

    # print(np.shape(dict_container['21'][0]))#this is the first one etc

    ###THIS CODE GOES THROUGH ALL THE DATA 
    for sample_id, sample_data in dict_container.items(): # 21, 22 etc
        # print(sample_data)
        for idx, data_set in enumerate(sample_data):
            xs = data_set[:,0]
            ys = data_set[:,1]
            data_set[:,1] = remove_baseline(xs, data_set[:,1])
            data_set[:,1] = 9.5*data_set[:,1]/np.max(data_set[:,1])
            pyplot.figure(figsize=(8,6))
            pyplot.title("B" + sample_id + " Raman Scattering - #" + str(idx+1))

            try:
                params, fit, ys, n_peaks = predict_and_plot_lorentzians(xs,ys, 5) # 5 = number of peaks to fit to ##Returns modified ys for y axis scaling
                for j in range(0, len(params), 3): 
                    ctr = params[j] 
                    amp = params[j+1]
                    width = params[j+2]
                    pyplot.plot(xs, lorentzian(xs, amp, ctr, width), ls='-')
                pyplot.plot(xs,ys, lw=1, label='data', c='black')
                pyplot.plot(xs, fit, 'r-', label='fit', c='red', lw=2, ls='--')
                pyplot.legend()
                pyplot.ylim([0,10])
                pyplot.xlim([250,450])
                pyplot.ylabel("Counts")
                pyplot.xlabel("Raman Shift/ Wavenumber ($cm^{-1}$)")

            except RuntimeError:
                print(traceback.format_exc())
                pyplot.plot(xs,ys, lw=1, label='data- no fit found', c='black')

                pyplot.legend()
                pyplot.ylabel("Counts")
                pyplot.xlabel("Raman Shift/ Wavenumber ($cm^{-1}$)")


            print('n_peaks: ', n_peaks)
    pyplot.show()


##Removes baseline of each individually and plots and fits to the sample averages
def main_predict_fit_averages():

    dict_container = import_data(samples_to_analyse) #global variable at present

    # print(np.shape(dict_container['21'][0]))#this is the first one etc

    ###THIS CODE GOES THROUGH ALL THE DATA 
    for sample_id, sample_data in dict_container.items():#21, 22 etc
        # print(sample_data)
        pyplot.figure(figsize=(8,6))
        pyplot.title("B" + sample_id + " Raman Scattering")
        ##should remove the baseline of each here

        for idx, data_set in enumerate(sample_data):
            xs = data_set[:,0]
            ys = data_set[:,1]
            data_set[:,1] = remove_baseline(xs, data_set[:,1])
            data_set[:,1] = 9.5*data_set[:,1]/np.max(data_set[:,1]) ##Make intensity arbitrary units (normalise)
            ys = data_set[:,1]

            # pyplot.plot(xs, ys,  linestyle='--') #Plot individual lines

        xs0 = sample_data[0][:,0]
        avg = np.average(sample_data, axis = 0)

        try:
            params, fit, ys , _ = predict_and_plot_lorentzians(avg[:,0], avg[:,1], 4) #5 = number of peaks to fit to ##Returns modified ys for y axis scaling
            for j in range(0, len(params), 3): 
                    ctr = params[j] 
                    amp = params[j+1]
                    width = params[j+2]
                    pyplot.plot(avg[:,0], lorentzian(avg[:,0], amp, ctr, width), ls='-')
            pyplot.plot(avg[:,0],ys, lw=1, label='data', c='blue')
            pyplot.plot(avg[:,0], fit, 'r-', label='fit', c='red', lw=2, ls='--')
            pyplot.ylabel("Counts")
            pyplot.xlabel("Raman Shift/ Wavenumber ($cm^{-1}$)")
            pyplot.legend()
        except RuntimeError:
            print(traceback.format_exc())
            pyplot.plot(avg[:,0],ys, lw=1, label='data- no fit found', c='black')
            pyplot.legend()
            pyplot.ylabel("Counts")
            pyplot.xlabel("Raman Shift/ Wavenumber ($cm^{-1}$)")

        pyplot.plot(avg[:,0], avg[:,1], label = 'average', c='black')##plots the average
        pyplot.ylabel("Counts")
        pyplot.xlabel("Raman Shift/ Wavenumber ($cm^{-1}$)")
        pyplot.legend()

    pyplot.show()


if __name__ == "__main__":
    main_predict_fit()
    # main_predict_fit_averages()
    # main_demo()
