from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.fftpack
import h5py
from scipy.io import loadmat
from scipy.signal import welch

from pyrqa.image_generator import ImageGenerator
from pyrqa.computation import RPComputation
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.analysis_type import Cross
from pyrqa.neighbourhood import FixedRadius, RadiusCorridor
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.neighbourhood import Unthresholded



from nolitsa import dimension
import pywt
import pickle

# CONFIGURE RUN
PlotTimeAndFreq = False # set True to plot Frequency Domain and Time Domain
ComputeResult = False # set True to plot FNN/TT
ComputeSingleElectrode = True

def computeWelchMethod(signal,npnts,showPlot=False):
    # "dynamic" FFT via welch's function
    dynamicHz, dynamicX1 = welch(signal, nfft=npnts)
    dynamicX2 = welch(signal, nfft=npnts)[1]

    # And plot
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))

    ax[0].plot(dynamicHz, np.abs(dynamicX1[:len(dynamicHz)]), '-o', label='Signal 1')
    ax[0].plot(dynamicHz, np.abs(dynamicX2[:len(dynamicHz)]), '-o', label='Signal 2')
    ax[0].legend()
    ax[0].set_xlim([0, .01])
    ax[0].set_title('Dynamic spectrum via Welch')
    if showPlot:
        plt.show()
    return dynamicX1

def plot_Time_Freq_Domain(data):
  fig,ax = plt.subplots(1,2,figsize=(20,6))
  ax[0].plot(time,data)
  ax[0].set_xlabel('Time (s)')
  ax[0].set_title('Time domain')

  dataX = scipy.fftpack.fft(data/npnts)
  ax[1].plot(hz,np.abs(dataX[:len(hz)]))
  ax[1].set_xlim([0,150])
  ax[1].set_xlabel('Frequency (Hz)')
  ax[1].set_title('Frequency domain')

def loadUnicornData(filename):
    f = h5py.File(filename)
    data = np.array(f['y']).T
    df = data[1:]
    return np.array(df)

def loadData(filename):
    data = loadmat(filename)
    electrodes = data['chanlocs']['labels']
    return np.array(data['EEGdata']),[electrodes[0][i][0] for i in range(len(electrodes[0]))]

def setDataSlice(df,fromrange, torange):
    ndata = np.zeros((len(df), torange-fromrange))
    for i in range(len(df)):
        ndata[i] = df[i][fromrange:torange]
    return ndata

def computeRQA(j,embedding,timedel,el_idx,selected_band,interval,counter,srate,subject):
    time_series = TimeSeries(j,
                             embedding_dimension=embedding,
                             time_delay=timedel)

    settings = Settings(time_series,
                        analysis_type=Classic,
                        neighbourhood=FixedRadius(radius=10.0),
                        #neighbourhood=Unthresholded(),
                        #neighbourhood=RadiusCorridor(inner_radius=0.1,outer_radius=5.0),
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=1)

    computation = RPComputation.create(settings,
                                       verbose=True)

    result = computation.run()
    nbrtype = 'neighbourhood=FixedRadius(radius=10.0)'
    fig = plt.figure(figsize=(8, 8))
    grid = plt.GridSpec(6, 6, hspace=0.6, wspace=0.6)
    plt.title(electrodeName[el_idx] + ', ' + bands[selected_band] + ' band, ' + 'emb = ' + str(embedding) + ' td = ' + str(timedel) + ' timestamp ' + str((interval * counter) / srate) )

    plt.axis('off')

    main_ax = fig.add_subplot(grid[:-1, 1:])
    # y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
    x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)
    main = main_ax.imshow(result.recurrence_matrix_reverse_normalized[::-1], cmap='gray', interpolation='none',origin='upper')
    x_hist.plot(j, color='gray')
    x_hist.set_xlabel(nbrtype)
    # x_hist.invert_yaxis()
    # y_hist.plot(np.array(j).T, color='gray')
    # y_hist.invert_xaxis()
    cbar_ax = fig.add_axes([0.04, 0.10, 0.05, 0.7])
    fig.colorbar(main, cax=cbar_ax)
    main_ax.invert_yaxis()

    plt.savefig(
        "../RR_plots/Dist_" + subject + '_' + electrodeName[el_idx] + '_' + bands[selected_band] + '_emb_' + str(
            embedding) + '_td_' + str(
            timedel) + '_tstamp_' + str((interval * counter) / srate) + '_v4s_'+ nbrtype + '.png', dpi=500)


def computeFNN_TT(j,embedding,timedel,interval,counter,srate):
    #ImageGenerator.save_recurrence_plot(result.recurrence_matrix_reverse_normalized,'fz_tests.png')

    time_series = TimeSeries(j,
                             embedding_dimension=embedding,
                             time_delay=timedel)
    settings = Settings(time_series,
                        analysis_type=Classic,
                        neighbourhood=FixedRadius(0.65),
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=1)

    computation = RQAComputation.create(settings,
                                        verbose=True)
    result = computation.run()
    result.min_diagonal_line_length = 2
    result.min_vertical_line_length = 2
    result.min_white_vertical_line_length = 2

    rqaArray = result.to_array()
    tt = rqaArray[10]
    fnn_dic[str((interval * counter) / srate)] = fnn
    tt_dic[str((interval * counter) / srate)] = tt
    timestamps.append((interval * counter) / srate)
    fnn_list.append(fnn)
    tt_list.append(tt)


    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    ax.scatter(timestamps, fnn_list, color="red", alpha=0.3)
    ax2.scatter(timestamps, tt_list, color="blue", alpha=0.3)
    ax.set_ylabel("FNN", color="red")
    ax2.set_ylabel("TT", color="blue", rotation=270)

    ax.set_yticks(np.round(np.linspace(np.min(fnn_list), np.max(fnn_list), 10), 2))
    ax2.set_yticks(np.round(np.linspace(np.min(tt_list), np.max(tt_list), 10), 2))
    plt.xticks(timestamps)
    ax.set_xlabel('Timestamps')
    plt.title("Electrode C6 " + "beta" + " band")

    plt.savefig("FNN_TT" + "_C6_beta_" + '.png')
    plt.close()

    plt.scatter(tt_list, fnn_list)
    plt.xlabel("Trapping times")
    plt.ylabel("Fnns")
    plt.show()


if __name__ == '__main__':

    df, electrodeName = loadData('../DataSets/teData.mat')
    dane = setDataSlice(df, 2000, 3000)
    srate = 250

    #RQA settings
    timedel =  1 #1 2 3 4 8 16 32 64
    embedding =  4  # 2 4 8 16 32 64
    interval = 1000
    el_idx = 16
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high gamma']
    selected_band = 3
    subject = 'SubjectName'

    fnn_list = []
    tt_list = []
    fnn_dic = {}
    tt_dic = {}
    timestamps = []
    time = np.arange(len(dane[el_idx]))
    npnts = len(time)

    hz = np.linspace(0, srate / 2, int(npnts / 2) + 1)
    if PlotTimeAndFreq:
        plot_Time_Freq_Domain(np.array(dane[el_idx]))

    db4 = pywt.Wavelet('db4')

    if ComputeSingleElectrode:
        coeffs = pywt.wavedec(dane[el_idx], db4, mode='periodic', level=5)

        f = signal.resample(coeffs[selected_band], npnts)
        f = dane[el_idx]
        #fw = computeWelchMethod(dane[el_idx],(npnts*2)-1,showPlot=True)
        #f = fw

        if interval:
            list_of_slices = zip(*(iter(f),) * interval)

            counter = 1
        for j in list_of_slices:

            computeRQA(j, embedding, timedel, el_idx, selected_band, interval, counter, srate, subject)


            fnn = dimension.fnn(j, dim=[embedding], tau=timedel, metric='euclidean')[2].item()
            # fnn = np.asscalar(dimension.fnn(j, dim=[embedding], tau=timedel, metric='euclidean')[2])

            if ComputeResult:
                computeFNN_TT(j, embedding, timedel, interval, counter, srate)

            counter += 1
    else:
        for electrode in range(len(electrodeName)):

            coeffs = pywt.wavedec(dane[electrode], db4, mode='periodic', level=5)
            f = signal.resample(coeffs[selected_band], npnts)

            if interval:
                list_of_slices = zip(*(iter(f),) * interval)
                counter = 1
            for j in list_of_slices:

                computeRQA(j, embedding, timedel, electrode, selected_band, interval, counter, srate, subject)

                fnn = dimension.fnn(j, dim=[embedding], tau=timedel, metric='euclidean')[2].item()
                # fnn = np.asscalar(dimension.fnn(j, dim=[embedding], tau=timedel, metric='euclidean')[2])

                if ComputeResult:
                    computeFNN_TT(j, embedding, timedel, interval, counter, srate)

                counter += 1


