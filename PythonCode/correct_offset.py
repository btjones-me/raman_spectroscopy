import glob, os
import numpy
import csv
from matplotlib import pyplot
import scipy.signal
import peakutils
from lmfit.models import PseudoVoigtModel
import math
import raman_analysis as ra


path_to_folder = "C:\\Users\\BenSurface_i5\\OneDrive\\Project\\Raman Spectroscopy\\CZTS_data\\CZTS_291116\\CZTS-Ben488\\B"
# path_to_folder = "C:\\Users\\BenSurface_i5\\OneDrive\\Project\\Raman Spectroscopy\\CZTS_data\\CZTS_111116\\B"
samples_to_analyse = ['28']

# def import_data():
#     for sample_id in samples_to_analyse:
#         data_container = []

#         path = path_to_folder + sample_id + "\\*.txt"
#         for filename in glob.glob(path):
#             data = numpy.genfromtxt("C:\\Users\\BenSurface_i5\\OneDrive\\Project\\Raman Spectroscopy\\CZTS_data\\CZTS_291116\\CZTS-Ben488\\B" + samples_to_analyse + ".txt")
#             print(numpy.shape(data))
#             # print(data)
#             pyplot.plot(data[:,0], data[:,1], marker='.')
#             pyplot.show()

#     return data_container

def import_data(samples_to_analyse):
    data_container = []
    for sample_id in samples_to_analyse:
        path = path_to_folder + sample_id + "\\*.txt"
        # print(path)
        for filepath in glob.glob(path):
            print(filepath)
            data_from_file = numpy.genfromtxt(filepath)
            data_container.append(data_from_file)
    # print(numpy.shape(data_container[0])) ##First data set from file 0
    # print(numpy.shape(data_container[0][:,0])) ##xs from file 0
    # print(numpy.shape(data_container[0][:,1])) ##ys from file 0
    return data_container

def import_reference_data(path):
    ''' Import reference data + remove baseline
    '''
    data = numpy.genfromtxt(path)
    xs, ys = ra.remove_baseline(data[:,0],data[:,1], True)
    result =  [list(a) for a in list(zip(xs,ys))]
    # result =  list(map(list, list(zip(xs,ys))))
    return numpy.asarray(result)

def show_reference_graph(ref_data):
    peak_indexes = ra.find_peaks_sri(ref_data[:,1], 100)
    print("Peaks identified at: ", ref_data[:,0][peak_indexes])

    pyplot.plot(ref_data[:,0], ref_data[:,1], linestyle='-', marker='.')
    pyplot.plot(ref_data[:,0][peak_indexes], ref_data[:,1][peak_indexes], linestyle='', marker='x')
    pyplot.show()

# sulfur 473.14 is peak 

reference_path = "C:\\Users\\BenSurface_i5\\OneDrive\\Project\\Raman Spectroscopy\\CZTS_data\\CZTS_291116\\CZTS-Ben488\\Reference_Sulfur_30_2 488.txt"
# import_data(['28'])
# print(numpy.shape(import_reference_data("C:\\Users\\BenSurface_i5\\OneDrive\\Project\\Raman Spectroscopy\\CZTS_data\\CZTS_291116\\CZTS-Ben488\\Reference_Sulfur_30_2 488.txt")))
show_reference_graph(import_reference_data(reference_path))