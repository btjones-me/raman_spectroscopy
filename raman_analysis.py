import glob, os
import numpy
import csv
from matplotlib import pyplot
import scipy.signal
import peakutils
from lmfit.models import PseudoVoigtModel
import math

# samples_to_analyse = ['21']
samples_to_analyse = ['21', '22', '23', '24', '25', '26', '27']

raman_literature_values = {'KS-E(TO)':(253,0.4), 'KS-E(LO)': (265.1,1), 'ST or KS(dis) - A1': (274.8,0.5), 'KS-A':(288.93,0.02), 'KS-(TO)':(305,2), 'KS-B(LO)':(318,1),
                            'ST or KS(dis) - A1': (334.4,0.3), 'KS - A': (339.41,0.01), 'ST or KS(dis) -E(TO)': (348.8,0.1), 'KS-E(TO)': (353.6,0.1), 'KS-E(TO)': (368.21,0.05)}

def find_peaks_peakutils(xs, ys, y_threshold=0.05, ys_min_separation=10, maxfev=20000):
    indexes = peakutils.indexes(ys, thres=y_threshold, min_dist=ys_min_separation) #Threshold is a percentage, min dist is minimum distance between peaks. I'm guessing it's in units of points, ie 3 points
    peaks_x = peakutils.peak.interpolate(xs, ys, ind=indexes, width=2)
    print('*************PEAKUTILS**************')
    print('Peak_pos= ', indexes, '\nPeaks= ', xs[indexes], ys[indexes])
    print('peaks_x: ', peaks_x)
    print('************************************')


    pyplot.plot(xs[indexes], ys[indexes], linestyle='', marker='x', markersize='12', color='blue') #Plot peaks
    ###plot vertical lines where peaks have been identified
    for peak in peaks_x:
        pyplot.axvline(x=peak, linestyle='--')

def remove_baseline(xs, ys):
    ###Remove baseline with polynomial
    y2 = ys + numpy.polyval([0.001,-0.08,5], xs)
    base = peakutils.baseline(y2, 2)
    return y2-base


    return indexes

def find_peaks_scipy(xs, ys, widths=numpy.arange(1,50, 0.5)):
    peaks = scipy.signal.find_peaks_cwt(ys, widths)#Problem
    pyplot.plot(xs[peaks], ys[peaks], marker='+', linestyle='', markersize='12')
    print('*************PEAKUTILS**************')
    print('From scipy: \nPeak_pos= ', peaks, '\nPeaks= ', xs[peaks], ys[peaks])
    return peaks 


def experiment_voigt(xs, ys): ###PRODUCES NAN VALUES FOR NO REASON
    # xs = numpy.arange(0, 160)
    # ys = numpy.array([3.3487290833206163, 3.441076831745743, 7.7932863251851305, 7.519064207516034, 7.394406511652473, 11.251458210206666, 4.679476113847004, 8.313048016542345, 9.348006472917458, 6.086336477997078, 10.765370342398741, 11.402519337778239, 11.151689287913552, 8.546151698722557, 8.323886291540909, 7.133249200994414, 10.242189407441712, 8.887686444395982, 10.759444780127321, 9.21095463298772, 15.693160143294264, 9.239683298899614, 9.476116297451632, 10.128625585058783, 10.94392508956097, 10.274287987647595, 9.552394167463973, 9.51931115335406, 9.923989117054466, 8.646255122559495, 12.207746464070603, 15.249531807666745, 9.820667193850705, 11.913964012172858, 9.506862412612637, 15.858588835799232, 14.918486963658015, 15.089436171053094, 14.38496801289269, 14.42394419048644, 15.759311758218061, 17.063349232010786, 12.232863723786215, 10.988245956134314, 19.109899560493286, 18.344353100589824, 17.397232553539542, 12.372706600456558, 13.038720878764792, 19.100965014037367, 17.094480819566147, 20.801679461435484, 15.763762333448557, 22.302320507719728, 23.394129891315963, 19.884812694503303, 22.09743700979689, 16.995815335935077, 24.286037929073284, 25.214705826961016, 25.305223543285013, 22.656121668613896, 30.185701748800568, 28.28382587095781, 35.63753811848088, 35.59816270398698, 35.64529822281625, 36.213428394807224, 39.56541841125095, 46.360702383473075, 55.84449512752349, 64.50142387788203, 77.75090937376423, 83.00423387164669, 111.98365374689226, 121.05211901294848, 176.82062069814936, 198.46769832454626, 210.52624393366017, 215.36708238568033, 221.58003148955638, 209.7551225151964, 198.4104196333782, 168.13949002992925, 126.0081896958841, 110.39003569380478, 90.88743461485616, 60.5443025644061, 71.00628698937221, 61.616294708485384, 45.32803695045095, 43.85638472551629, 48.863070901568086, 44.65252243455522, 41.209120125948104, 36.63478075990383, 36.098369542551325, 37.75419965137265, 41.102019290969956, 26.874409332756752, 24.63314900554918, 26.05340465966265, 26.787053802870535, 16.51559065528567, 19.367731289491633, 17.794958746427422, 19.52785218727518, 15.437635249660396, 21.96712662378481, 15.311043443598177, 16.49893493905559, 16.41202114648668, 17.904512123179114, 14.198812322372405, 15.296623848360126, 14.39383356078112, 10.807540004905345, 17.405310725810278, 15.309786310492559, 15.117665282794073, 15.926377010540376, 14.000223621497955, 15.827757539949431, 19.22355433703294, 12.278007446886507, 14.822245428954957, 13.226674931853903, 10.551237809932955, 8.58081654372226, 10.329123069771072, 13.709943935412294, 11.778442391614956, 14.454930746849122, 10.023352452542506, 11.01463585064886, 10.621062477382623, 9.29665510291416, 9.633579419680572, 11.482703531988037, 9.819073927883121, 12.095918617534196, 9.820590920621864, 9.620109753045565, 13.215701804432598, 8.092085538619543, 9.828015669152578, 8.259655585415379, 9.424189583067022, 13.149985946123934, 7.471175119197948, 10.947567075630904, 10.777888096711512, 8.477442195191612, 9.585429992609711, 7.032549866566089, 5.103962051624133, 9.285999577275545, 7.421574444036404, 5.740841317806245, 2.3672530845679])
    # ys = numpy.array([10, 13, 16, 21, 29, 17, 14, 5, 15, 18, 13, 3, 7, 10, 12, 10, 10, 10, 10, 10]) #test later
    # xs = numpy.linspace(0,numpy.max(ys), len(ys))
    mod = PseudoVoigtModel()
    pars = mod.guess(ys, x=xs)
    out = mod.fit(ys, pars, x=xs)
    print(out.fit_report(min_correl=0.25))
    out.plot()

def predict_structures(peak_positions, sample_id):
    tolerance = 15
    for p in peak_positions:
        for key in raman_literature_values:
            value, error = raman_literature_values.get(key)
            if (abs(p-value) < tolerance*error):
                print(key, 'detected in sample ', sample_id,  '  \n peak, literature: ', p, value, '\n   Percentage confidence: ', ((p-value)/p)*100, '%')
                
            #In future, give coefficient of how close it is/ confidencde interval


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


    y_data = remove_baseline(data_container[i][0],avg[1])
    
    pyplot.plot(data_container[i][0], y_data, color='black')

    # find_peaks_peakutils(data_container[i][0], y_data)
    peak_pos = find_peaks_scipy(data_container[i][0], y_data, numpy.arange(1,50,0.5))

    predict_structures(data_container[i][0][peak_pos], sample_id)

    # experiment_voigt(data_container[i][0], y_data)

pyplot.show()



# i want a list of file data from each file [file1, file2, file3..] etc
# where file1 = [ [xs], [ys] ]
#and where xs = [0 , 1, 2, 3] etc