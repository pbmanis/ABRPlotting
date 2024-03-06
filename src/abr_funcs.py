"""
Library of functions for ABR data handling and analysis.
Pulled out of plotABRs to collect these together.

"""
from pathlib import Path
from typing import Union

# import mat4py

import scipy
import seaborn as sns


class ABRFuncs:
    def __init__(self):
        pass

    def getSPLs(self, datapath: Union[Path, str]):
        """
        Return all the spl files in the directory. There is one spl file
        per intensity run.
        """
        print("ABRFuncs: getSPLs: ", datapath)
        spl_files = list(Path(datapath).glob("*-SPL.txt"))
        # print("splfiles: ", spl_files)
        rundict = {}
        for spl_run in spl_files:
            with open(spl_run, "r") as fh:
                spldata = fh.read()
            timestamp = str(spl_run.name)[:-8]
            rundict[timestamp] = [
                float(spl) for spl in spldata.split("\n") if spl not in ["", "\n"]
            ]
        return rundict

    def getFreqs(self, datapath: Union[Path, str]):
        """
        Return all the tonepip files in the directory. There is one tonepip file
        per frequency for its intensity run. We key off of the kHz.txt file to
        get the timestamp and get the frequencies from that file

        """
        # return all the tone response files in the directory.
        kHz_files = list(Path(datapath).glob("*-kHz.txt"))  # the frequency lists
        frequency_runs = [str(f)[:-8] for f in kHz_files]
        rundict = {}
        for frequency_run in kHz_files:
            with open(frequency_run, "r") as fh:
                freq_data = fh.read()
            freq_data = freq_data.strip()
            freq_data = freq_data.replace("\t", " ")
            freq_data = freq_data.replace(r"[s]*", " ")
            freq_data = freq_data.replace("\n", " ")
            freq_data = freq_data.split(" ")

            timestamp = str(frequency_run.name)[:-8]
            # for khz in freq_data:
            #     print(f"khz: <{khz:s}>")
            rundict[timestamp] = [
                float(khz) for khz in freq_data if len(khz) > 0
            ]  # handle old data with blank line at end
        # print("rundict: ", rundict)
        return rundict

    def get_matlab(self, datapath):
        matfiles = list(Path(datapath).glob("*.mat"))
        # import matlab.engine
        # eng = matlab.engine.start_matlab()
        # for mf in matfiles:
        #     # mdata = scipy.io.loadmat(mf)
        #     # print(mdata['bigdata'].abr4_data_struct)
        #     print(mf)
        #     data = eng.load(str(mf), nargout=1)
        #     print(data['bigdata']) # ['abr4_calibration_struct'])

        # exit()

    def getMarkerStyle(self, directory: str = "", markers: Union[dict, None] = None):
        """assign a marker style based on annotations
        in ABR_Datasets.py, using the "markers" key.

        Parameters
        ----------
        directory : str, optional
            name of the directory holding the recordings for this
            subject.
        markers : a dict mapping datasets go markers
            usually pulled from info.markers
        Returns
        -------
        markerstyle : string (matlab marker)
        group: string (group name from the dictionary)
        Note that if there is no match to the categories (groups)
        in the dictionary, then the marker is "x" and the group is
        "Unidentified".
        """
        markerstyle = "x"  # default
        group = "ND"
        dname = Path(directory).name
        if markers is not None:
            # markers is a dict with code: (markerstyle, location)
            # where code is a string to find in the directory name
            # markersytle is a matlab marker style
            # location is "any" (anywhere in the name) or "end" (find at the end of the string)
            for mi in list(markers.keys()):
                if markers[mi][1] == "any":
                    found = dname.find(mi)
                    if found >= 0:
                        markerstyle = markers[mi][0]
                        group = mi
                elif markers[mi][1] == "end":
                    if dname.endswith(mi):
                        markerstyle = markers[mi][0]
                        group = mi
        else:
            pass
        return markerstyle, group

    def makeColorMap(self, N, labels):
        """
        create a color map of N levels using the specified labels

        Parameters
        ----------
        N : int
            Number of color levels
        labels : tuples
            list of tuples of rgb values corresponding to color levels

        Returns
        -------
        dict for colormap
        """
        rgb_values = sns.color_palette("Set2", N)
        return dict(list(zip(labels, rgb_values)))

    def filter(self, data, order, lowpass, highpass, samplefreq, ftype="butter"):
        """
        Returns waveform filtered using filter paramters specified. Since
        forward and reverse filtering is used to avoid introducing phase delay,
        the filter order is essentially doubled.

        Parameters
        ----------
        data : m numpy array of floats
            the data waveforms, as a 1-d array

        order : filter order.

        lowpass : float
            Low pass filter setting, in Hz

        highpass : float
            High pass filter setting, in Hz

        samplefreq : float
            sample frequency, in Hz (inverse of sample rate)

        ftype : str (default: 'butter')
            Type of filter to implement. Default is a Butterworth filter.

        Returns
        -------
        signal : numpy array of floats
            the filtered signal waveform.
        """

        Wn = highpass / (samplefreq / 2.0), lowpass / (samplefreq / 2.0)
        kwargs = dict(N=order, Wn=Wn, btype="band", ftype=ftype)
        b, a = scipy.signal.iirfilter(output="ba", **kwargs)
        zpk = scipy.signal.iirfilter(output="zpk", **kwargs)
        try:
            self._zpk.append(zpk)
        except:
            self._zpk = [zpk]
        self.signal = scipy.signal.filtfilt(b, a, data, padlen=int(len(data) / 10))
        return self.signal

    def SignalFilter(self, signal, LPF, HPF, samplefreq, debugFlag=True):
        """Filter signal within a bandpass with elliptical filter.

        Digitally filter a signal with an elliptical filter; handles
        bandpass filtering between two frequencies.

        Parameters
        ----------
        signal : array
            The signal to be filtered.
        LPF : float
            The low-pass frequency of the filter (Hz)
        HPF : float
            The high-pass frequency of the filter (Hz)
        samplefreq : float
            The uniform sampling rate for the signal (in seconds)

        Returns
        -------
        w : array
            filtered version of the input signal
        """
        if debugFlag:
            print(f"sfreq: {samplefreq:.1f}, LPF: {LPF:.1f} HPF: {HPF:.1f}")
        flpf = float(LPF)
        fhpf = float(HPF)
        sf = float(samplefreq)
        sf2 = sf / 2.0
        wp = [fhpf / sf2, flpf / sf2]
        ws = [0.5 * fhpf / sf2, 2 * flpf / sf2]
        if debugFlag:
            print(
                "signalfilter: samplef: %f  wp: %f, %f  ws: %f, %f lpf: %f  hpf: %f"
                % (sf, wp[0], wp[1], ws[0], ws[1], flpf, fhpf)
            )
        filter_b, filter_a = scipy.signal.iirdesign(
            wp, ws, gpass=1.0, gstop=60.0, ftype="ellip"
        )
        msig = np.nanmean(signal)
        signal = signal - msig
        w = scipy.signal.lfilter(
            filter_b, filter_a, signal
        )  # filter the incoming signal
        signal = signal + msig
        if debugFlag:
            print(
                "sig: %f-%f w: %f-%f"
                % (np.amin(signal), np.amax(signal), np.amin(w), np.amax(w))
            )
        return w
