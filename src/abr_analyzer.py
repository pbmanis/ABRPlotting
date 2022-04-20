from pathlib import Path
import os.path

import numpy as np
import scipy.signal

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

        response_range = [1.0, 7]  # window, in msec...
        baseline = [10, 24] # [0.0, 0.8]

        self.P1N1(min_lat=response_range[0])
        self.ppio = self.peaktopeak(response_range)
        self.rms_response = self.measure_rms(response_range)
        self.rms_baseline = self.measure_rms(baseline)
        self.specpower()

    def peaktopeak(self, tr):
        tx = self.gettimeindices(tr)
        pp = np.zeros(self.waves.shape[0])
        for i in range(self.waves.shape[0]):
            pp[i] = np.max(self.waves[i][tx]) - np.min(self.waves[i][tx])
        return pp

    def P1N1(self, min_lat:float=2.5, dev:float=2.5):
        """
        Use Buran's peakdetect routine to find the peaks and returan a list
        of peaks. Works twice - first run finds all the positive peaks, and
        the second run finds the negative peaks that follow the positive peaks
        Note that the peaks may not be "aligned" in the sense that it is possible
        to find two positive peaks in succession without a negative peak.
        """
        r = {}
        n = {}
        for j in range(self.waves.shape[0]):
            r[j] = peakdetect.find_np(
                self.sample_freq,
                self.waves[j, :],
                nzc_algorithm_kw={"dev": 1},
                guess_algorithm_kw={"min_latency": min_lat},
            )
            if len(r[j]) > 0:
                n[j] = peakdetect.find_np(
                    self.sample_freq,
                    -self.waves[j, :],
                    nzc_algorithm_kw={"dev": dev},
                    guess_algorithm_kw={"min_latency": self.timebase[r[j][0]]},
                )  # find negative peaks after positive peaks
        self.p1n1 = (r, n)

    def measure_rms(self, tr):
        tx = self.gettimeindices(tr)
        rms = np.zeros(self.waves.shape[0])
        for i in range(self.waves.shape[0]):
            rms[i] = np.std(self.waves[i][tx])
        return rms

    def gettimeindices(self, tr):
        (x,) = np.where((tr[0] <= self.timebase) & (self.timebase < tr[1]))
        return x

    def specpower(self, fr=[500.0, 1500.0], win=[0, -1]):
        fs = 1.0 / self.sample_rate
        # fig, ax = mpl.subplots(1, 1)
        psd = [None] * self.waves.shape[0]
        psdwindow = np.zeros(self.waves.shape[0])
        for i in range(self.waves.shape[0]):
            f, psd[i] = scipy.signal.welch(
                1e6 * self.waves[i][win[0] : win[1]],
                fs,
                nperseg=256,
                nfft=8192,
                scaling="density",
            )
            (frx,) = np.where((f >= fr[0]) & (f <= fr[1]))
            psdwindow[i] = np.nanmean(psd[i][frx[0] : frx[-1]])
        #            ax.semilogy(f, psd[i])
        # ax.set_ylim([0.1e-4, 0.1])
        # ax.set_xlim([10., 2000.])
        # ax.set_xlabel('F (Hz)')
        # ax.set_ylabel('PSD (uV^2/Hz)')
        # mpl.show()
        self.fr = f
        self.psd = psd
        self.psdwindow = psdwindow
        return psdwindow

    def thresholds(self, waves, spls, tr=[0.0, 8.0], SD=4.0):
        """
        Auto threshold detection:
        BMC Neuroscience200910:104  DOI: 10.1186/1471-2202-10-104
        Use last 10 msec of 15 msec window for SD estimates
        Computes SNR (max(abs(signal))/reference SD) for a group of traces
        The reference SD is the MEDIAN SD across the intensity run.
        Th

        """
        refwin = self.gettimeindices([15.0, 25.0])
        sds = np.std(waves[:, refwin[0] : refwin[-1]], axis=1)
        self.median_sd = np.nanmedian(sds)
        tx = self.gettimeindices(tr)
        self.max_wave = np.max(np.fabs(waves[:, tx[0] : tx[-1]]), axis=1)
        (thr,) = np.where(
            self.max_wave >= self.median_sd * SD
        )  # find criteria threshold
        (thrx,) = np.where(
            np.diff(thr) == 1
        )  # find first contiguous point (remove low threshold non-contiguous)
        if len(thrx) > 0:
            return spls[thr[thrx[0]]]
        else:
            return np.nan

    def threshold_spec(self, waves, spls, tr=[0.0, 8.0], SD=4.0):
        """
        Auto threshold detection:
        BMC Neuroscience200910:104  DOI: 10.1186/1471-2202-10-104
        Use last 10 msec of 15 msec window for SD estimates
        Computes SNR (max(abs(signal))/reference SD) for a group of traces
        The reference SD is the MEDIAN SD across the intensity run.

        MODIFIED version: criteria based on power spec

        """
        refwin = self.gettimeindices([15.0, 25.0])
        sds = self.specpower(fr=[800.0, 1250.0], win=[refwin[0], refwin[-1]])
        self.median_sd = np.nanmedian(sds)
        tx = self.gettimeindices(tr)
        self.max_wave = self.specpower(fr=[800.0, 1250.0], win=[tx[0], tx[-1]])
        (thr,) = np.where(
            self.max_wave >= self.median_sd * SD
        )  # find criteria threshold
        if len(thr) > 0:
            return spls[thr[0]]
        else:
            return np.nan
