import glob, os
import numpy
import csv
from matplotlib import pyplot
import scipy.signal
import peakutils

samples_to_analyse = ['21']
samples_to_analyse = ['21', '22', '23', '24', '25', '26', '27']



def find_peaks_peakutils(xs, ys, y_threshold=0.04, ys_min_separation=4):
    indexes = peakutils.indexes(ys, thres=y_threshold, min_dist=ys_min_separation) #Threshold is a percentage, min dist is minimum distance between peaks. I'm guessing it's in units of points, ie 3 points
    print('From peakutils: \nPeak_pos= ', indexes, '\nPeaks= ', xs[indexes], ys[indexes])
    pyplot.plot(xs[indexes], ys[indexes], linestyle='', marker='x', markersize='12')
    return indexes

def find_peaks_scipy(xs, ys, widths=numpy.arange(1,50, 0.05)):
    peaks = scipy.signal.find_peaks_cwt(ys, widths)#Problem
    pyplot.plot(xs[peaks], ys[peaks], marker='+', linestyle='', markersize='12')
    print('From scipy: \nPeak_pos= ', peaks, '\nPeaks= ', xs[peaks], ys[peaks])
    return peaks 

for sample_id in samples_to_analyse:
    data_container = []

    path = "C:\\Users\\BenSurface_i5\\OneDrive\\Project\\Raman Spectroscopy\\CZTS_data\\CZTS_111116\\B" + sample_id + "\\*.txt"
    for filename in glob.glob(path):
        data_set = []
        with open(filename, 'r') as f: #for each file
            xs,ys = [], []
            for l in f:
                row = l.split()
                xs.append(row[0])
                ys.append(row[1])
            data_set.append(xs)
            data_set.append(ys)
        data_container.append(data_set)
        
    data_container = numpy.array(data_container).astype(numpy.float)

    print(numpy.shape(data_container[0]))
    print(numpy.shape(data_container))

    avg = numpy.average(data_container, axis = 0)
    print(avg)
    print(numpy.shape(avg))


    pyplot.figure(figsize=(8,6))
    for i in range(len(data_container)):
        # pyplot.plot(data_container[i][0], data_container[i][1],  linestyle='--') #Plot individual lines
        pyplot.title("B" + sample_id + " Raman Scattering")
    pyplot.ylabel("Counts")
    pyplot.xlabel("Raman Shift/ Wavenumber ($cm^{-1}$)")





    
    pyplot.plot(data_container[0][0], avg[1], color='black')

    find_peaks_peakutils(data_container[i][0], avg[1])
    find_peaks_scipy(data_container[i][0], avg[1])

 

pyplot.show()



# i want a list of file data from each file [file1, file2, file3..] etc
# where file1 = [ [xs], [ys] ]
#and where xs = [0 , 1, 2, 3] etc