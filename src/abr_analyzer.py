from pathlib import Path
import os.path
from re import S
from termios import NL1
from tkinter import NONE

import numpy as np
import scipy.signal
import matplotlib.cm
import matplotlib.pyplot as mpl

from . import peakdetect  # from Brad Buran's project, but cloned and modified here
from . ABR_Datasets import ABR_Datasets # just the dict describing the datasets

class Analyzer(object):
    """
    Provide analysis functions for ABRs.
    """

    def __init__(self, sample_frequency:float=1e5):
        self.ppioMarker = "s"
        self.rmsMarker = "o"
        self.psdMarker = "*"
        self.baselineMarker = "+"
        self.sample_freq = sample_frequency

    def analyze(self, timebase, waves):

        # self.sample_rate = 0.001 * (timebase[1] - timebase[0])
        # self.sample_freq = 1.0 / self.sample_rate
        self.sample_rate = 1./self.sample_freq
        self.waves = waves
        self.timebase = timebase

        response_range = [1.0, 5]  # window, in msec...
        baseline = [20, 25] # [0.0, 0.8]

        self.get_triphasic(min_lat=response_range[0], dev=2.5)
        self.ppio = self.peaktopeak(response_range)
        self.rms_response = self.measure_rms(response_range)
        self.rms_baseline = self.measure_rms(baseline)
        # self.specpower(waves)

    def peaktopeak(self, tr):
        tx = self.gettimeindices(tr)
        pp = np.zeros(self.waves.shape[0])
        for i in range(self.waves.shape[0]):
            pp[i] = np.max(self.waves[i][tx]) - np.min(self.waves[i][tx])
        return pp

    def get_triphasic(self, min_lat:float=1.0, dev:float=2.5):
        """
        Use Buran's peakdetect routine to find the peaks and returan a list
        of peaks. Works 3 times - first run finds all the positive peaks, and
        the second run finds the negative peaks that follow the positive peaks.
        The last run finds the next positive peak after the negative peak.
        This yields P1-N1-P2, which is the returned value.
        Note that the peaks from peakdetect may not be "aligned" in the sense that it is possible
        to find two positive peaks in succession without a negative peak.
        """
        p1 = {}
        n1 = {}
        p2 = {}
        for j in range(self.waves.shape[0]):
            p1[j] = peakdetect.find_np(
                self.sample_freq,
                self.waves[j, :],
                nzc_algorithm_kw={"dev": 1.0},
                guess_algorithm_kw={"min_latency": min_lat},
            )

            if len(p1[j]) > 0:
                n1[j] = peakdetect.find_np(
                    self.sample_freq,
                    -self.waves[j, :],
                    nzc_algorithm_kw={"dev": dev},
                    guess_algorithm_kw={"min_latency": self.timebase[p1[j][0]]},
                )  # find negative peaks after positive peaks
            else:
                n1[j] = np.nan
            if len(n1[j]) > 0:
                p2[j] = peakdetect.find_np(
                    self.sample_freq,
                    self.waves[j, :],
                    nzc_algorithm_kw={"dev": dev},
                    guess_algorithm_kw={"min_latency": self.timebase[n1[j][0]]},
                )  # find negative peaks after positive peaks
            else:
                p2[j] = np.nan
        self.p1n1p2 = (p1, n1, p2)

    def measure_rms(self, tr):
        tx = self.gettimeindices(tr)
        rms = np.zeros(self.waves.shape[0])
        for i in range(self.waves.shape[0]):
            rms[i] = np.std(self.waves[i][tx])
        return rms

    def gettimeindices(self, tr):
        (x,) = np.where((tr[0] <= self.timebase) & (self.timebase < tr[1]))
        return x

    def specpower(self, waves, spls, fr=[500.0, 1500.0], win=[0, -1], ax = None, ax2=None, lt='-', cindex=0):
        fs = 1.0 / self.sample_rate
        psd = [None] * waves.shape[0]
        psdwindow = np.zeros(waves.shape[0])
        print("win: ", win)
        cmap = matplotlib.cm.get_cmap('tab20')
        for i in range(waves.shape[0]):
            freqs, psd[i] = scipy.signal.welch(
                1e6 * waves[i][win[0] : win[1]],
                fs,
                nperseg=256,
                nfft=8192,
                scaling="density",
            )
            (frx,) = np.where((freqs >= fr[0]) & (freqs <= fr[1]))
            psdwindow[i] = np.nanmax(psd[i][frx[0] : frx[-1]])
            if ax is not None:
                ax.semilogx(freqs, psd[i], linestyle = lt, label=f"{spls[i]:.1f}", color=cmap(i/20.0))
                # ax.set_ylim([0.1e-4, 0.1])
                ax.set_xlim([10., 2500.])
                ax.set_xlabel('F (Hz)')
                ax.set_ylabel(r'PSD ($\mu V^2/Hz$)')
            if ax2 is not None:
                tb = fs*np.arange(0, len(waves[i][win[0]: win[1]]))
                ax2.plot(tb, waves[i][win[0]: win[1]], linestyle=lt, color=cmap(i/20.))
        self.fr = freqs
        self.psd = psd
        self.psdwindow = psdwindow
        return psdwindow

    def thresholds(self, waves, spls, tr=[1.0, 8.0], reftimes=[20, 25], SD=65.0):
        """
        Auto threshold detection:
        BMC Neuroscience200910:104  DOI: 10.1186/1471-2202-10-104
        Use last 10 msec of 25 msec window for SD estimates
        Computes SNR (max(abs(signal))/reference SD) for a group of traces
        The reference SD is the MEDIAN SD across the intensity run.
        

        """
        refwin = self.gettimeindices(reftimes)
        sds = np.std(waves[:, refwin[0] : refwin[-1]], axis=1)
        self.median_sd = np.nanmedian(sds)
        tx = self.gettimeindices(tr)
        self.max_wave = np.max(np.fabs(waves[:, tx[0] : tx[-1]]), axis=1)
        true_thr = np.max(spls)
        # if len(thr) > 0:
        for i, s in enumerate(spls):
            j = len(spls)-i-1
            if self.max_wave[j] >= self.median_sd*SD:
                true_thr = spls[j]
            else:
                break
        # else:
        #     t = len(spls)-1
        print("thr: ", true_thr)
 
        return true_thr # spls[thr[0]]
        # (thr,) = np.where(
        #     self.max_wave >= self.median_sd * SD
        # )  # find criteria threshold
        # (thrx,) = np.where(
        #     np.diff(thr) == 1
        # )  # find first contiguous point (remove low threshold non-contiguous)
        # if len(thrx) > 0:
        #     return spls[thr[thrx[0]]]
        # else:
        #     return np.nan

    def threshold_spec(self, waves, spls, tr=[1.0, 8.0], reftimes=[20, 25], SD=4.0):
        """
        Auto threshold detection:
        BMC Neuroscience200910:104  DOI: 10.1186/1471-2202-10-104
        Use last 10 msec of 15 msec window for SD estimates
        Computes SNR (max(abs(signal))/reference SD) for a group of traces
        The reference SD is the MEDIAN SD across the intensity run.

        MODIFIED version: criteria based on power spec

        """
        showspec = False
        # print('max time: ', np.max(self.timebase))
        refwin = self.gettimeindices(reftimes)
        if showspec:
            fig, ax = mpl.subplots(1,2, figsize=(10, 5))
        else:
            ax = [None, None]
        sds = self.specpower(waves, spls, fr=[1900., 2200.0], win=[refwin[0], refwin[-1]], ax=ax[0], ax2=ax[1], lt = '--')
        self.median_sd = np.nanmedian(sds)
        tx = self.gettimeindices(tr)
        self.max_wave = self.specpower(waves, spls, fr=[1900., 2200.0], win=[tx[0], tx[-1]], ax=ax[0], ax2=ax[1], lt = '-')
        # (thr,) = np.where(
        #     self.max_wave >= self.median_sd * SD
        # )  # find all spls that meet the criteria threshold
        # then find the lowest one for whcih ALL higher SPLs are also
        # above threshold
        # print("thr: ", thr)
        print("spl: ", spls)
        true_thr = np.max(spls)
        # if len(thr) > 0:
        for i, s in enumerate(spls):
            j = len(spls)-i-1
            if self.max_wave[j] >= self.median_sd*SD:
                true_thr = spls[j]
            else:
                break
        # else:
        #     t = len(spls)-1
        print("thr: ", true_thr)

        if showspec:
            mpl.show()
        return true_thr # spls[thr[0]]
        # else:
        #     return np.nan
