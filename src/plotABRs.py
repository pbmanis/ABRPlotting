import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import mat4py
import pylibrary.plotting.plothelpers as PH
import scipy.signal
import scipy.io
import seaborn as sns  # makes plot background light grey with grid, no splines. Remove for publication plots
from matplotlib import pyplot as mpl
from matplotlib.backends.backend_pdf import PdfPages

from . import getcomputer  # stub to return the computer and base directory
from . import abr_analyzer
from .ABR_Datasets import ABR_Datasets  # just the dict describing the datasets
from . import MatFileMethods

basedir, computer_name = getcomputer.getcomputer()


class ABR:
    """
    Read an ABR data set from the matlab program
    Plot the traces.

    Parameters
    ----------
    datapath : str
        Path to the datasets. The data sets are expected to be collected under this
        path into individual directories, each with the results of the ABR runs for
        one subject.
    mode : string ('clicks' or 'tones')
        Specify type of data in the data set
    info : dict
        Dictionary with the following keys:
            invert : boolean (default : False)
                Changes sign of waveform if True
            minlat : float (default 0.75)
                minimum latency for event detection
            term : float (default '\r')
                line terminator (may change if the data has been read by an editor or is
                on a windows vs. mac vs linux system)
    """

    def __init__(
        self,
        datapath,
        mode="clicks",
        info={"invert": False, "minlat": 0.75, "term": "\r"},
    ):

        # Set some default parameters for the data and analyses

        if 'sample_freq' in list(info.keys()):
            self.sample_freq = 50000.0  # Hz
        else:
            self.sample_freq = 100000.0
        self.sample_rate = (
            1.0 / self.sample_freq
        )  # standard interpolated sample rate for matlab abr program: 10 microsecnds
        if "spec_bandpass" in list(info.keys()):
            self.spec_bandpass = info["spec_bandpass"]
        else:
            self.spec_bandpass = [800., 1200.]
        self.hpf = 500.0
        self.lpf = 2500.0  # filter frequencies, Hz
        self.mode = mode
        self.info = info
        self.clickdata = {}
        self.tonemapdata = {}
        # build color map where each SPL is a color (cycles over 12 levels)
        bounds = np.linspace(0, 120, 25)  # 0 to 120 db inclusive, 5 db steps
        color_labels = np.unique(bounds)
        rgb_values = sns.color_palette("Set2", 25)
        # Map label to RGB
        self.color_map = dict(list(zip(color_labels, rgb_values)))

        self.datapath = Path(datapath)
        self.term = info["term"]
        self.minlat = info["minlat"]
        self.invert = info[
            "invert"
        ]  # data flip... depends on "active" lead connection to vertex (false) or ear (true).

        self.characterizeDataset()

        # build color map where each SPL is a color (cycles over 12 levels)
        self.max_colors = 25
        bounds = np.linspace(0, 120, self.max_colors)  # 0 to 120 db inclusive, 5 db steps
        color_labels = np.unique(bounds)
        self.color_map = self.makeColorMap(self.max_colors, color_labels)
        color_labels2 = list(range(self.max_colors))
        self.summaryClick_color_map = self.makeColorMap(
            self.max_colors, list(range(self.max_colors))
        )  # make a default map, but overwrite for true number of datasets
        self.psdIOPlot = False

    def characterizeDataset(self):
        """
        Look at the directory in datapath, and determine what datasets are present.
        A dataset consists of at least 3 files:
        yyyymmdd-HHMM-SPL.txt : the SPL levels in a given run, with year, month day hour and minute
        yyyymmdd-HHMM-{n,p}-[freq].txt : hold waveform data for a series of SPL levels
            For clicks, just click polarity - holds waveforms from the run
            typically there will be both an n and a p file because we use alternating polarities.
            For tones, there is also a number indicating the tone pip frequency used.

        The runs are stored in separate dictionaries for the click and tone map runs, including
        their SPL levels (and for tone maps, frequencies).

        Parameters
        ----------
        None

        Returns
        -------
        Nothing
        """

        self.get_matlab()
        self.spls = self.getSPLs()
        self.freqs = self.getFreqs()
        # A click run will consist of an SPL, n and p file, but NO additional files.
        self.clicks = {}
        for s in self.spls.keys():
            if s not in list(self.freqs.keys()):  # skip SPL files associated with a tone map
                self.clicks[s] = self.spls[s]

        # inspect the directory and get a listing of all the tone and click maps
        # that can be found.
        self.tonemaps = {}
        for i, f in enumerate(self.freqs.keys()):
            self.tonemaps[f] = {
                "stimtype": "tonepip",
                "Freqs": self.freqs[f],
                "SPLs": self.spls[f[:13]],
            }
            if self.mode == "tones":
                if i == 0:
                    print(f"  Frequency maps: ")
                print(f"    {f:s} ", self.tonemaps[f])

        self.clickmaps = {}
        for i, s in enumerate(self.clicks.keys()):
            self.clickmaps[s] = {"stimtype": "click", "SPLs": self.spls[s]}
            if self.mode == "clicks":
                if i == 0:
                    print("\n  Click Intensity Runs: ")
                print(f"    Run: {s:s}")
                print(self.clickmaps[s])

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

    def adjustSelection(self, select, tone=False):
        freqs = []
        if select is None:
            return select, freqs
        for i, s in enumerate(select):
            print("adjust selection: s: ", s)
            if s is None:
                continue
            if isinstance(s, int):
                select[i] = "%04d" % s  # convert to string
            if isinstance(s, str) and len(s) < 4:  # make sure size is ok
                select[i] = "%04d" % int(s)
            if tone:
                base = list(self.tonemaps.keys())[0][:9]
                for f in self.tonemaps[base + select[i]]["Freqs"]:
                    if f not in freqs:
                        freqs.append(f)
        return select, freqs

    def getSPLs(self):
        """
        Return all the spl files in the directory. There is one spl file
        per intensity run.
        """
        spl_files = list(self.datapath.glob("*-SPL.txt"))
        rundict = {}
        for spl_run in spl_files:
            with open(spl_run, "r") as fh:
                spldata = fh.read()
            timestamp = str(spl_run.name)[:-8]
            rundict[timestamp] = [float(spl) for spl in spldata.split("\n") if spl not in ["", "\n"]]
        return rundict

    def getFreqs(self):
        """
        Return all the tonepip files in the directory. There is one tonepip file
        per frequency for its intensity run.
        We key off of the kHz.txt file to get the timestamp
        and get the frequencies from that file

        """
        # return all the tone response files in the directory.
        kHz_files = list(self.datapath.glob("*-kHz.txt"))
        # print("\nkhz files: ", kHz_files)
        frequency_runs = [str(f)[:-8] for f in kHz_files]
        # print("\ngetFreqs frequency_runs: ", frequency_runs)
        rundict = {}
        for frequency_run in kHz_files:
            with open(frequency_run, "r") as fh:
                freq_data = fh.read()
            timestamp = str(frequency_run.name)[:-8]
            rundict[timestamp] = [
                float(khz) for khz in freq_data.split("\t") if khz[0] != "\n"
            ]  # handle old data with blank line at end
        # print("\nrundict: ", rundict)
        return rundict

    def get_matlab(self):
        matfiles = list(self.datapath.glob("*.mat"))
        print(list(matfiles))
        # import matlab.engine
        # eng = matlab.engine.start_matlab()
        # for mf in matfiles:
        #     # mdata = scipy.io.loadmat(mf)
        #     # print(mdata['bigdata'].abr4_data_struct)
        #     print(mf)
        #     data = eng.load(str(mf), nargout=1)
        #     print(data['bigdata']) # ['abr4_calibration_struct'])


        # exit()


    def getClickData(self, select):
        """
        Gets the click data for the current selection
        The resulting data is held in a dictionary structured as
        {mapidentity: dict of {waves, time, spls and optional marker}

        Parameters
        ----------
        select : which data to select

        """
        select, freqs = self.adjustSelection(select)
        # get data for clicks and plot all on one plot
        self.clickdata = {}
        for i, s in enumerate(self.clickmaps.keys()):
            if select is not None:
                if s[9:] not in select:
                    continue
            # if (
            #     s[0:8] == "20170419"
            # ):  # these should not be here... should be in an excel table
            #     smarker = "go-"
            # elif s[0:8] in ["20170608", "20170609"]:
            #     smarker = "bs-"
            # else:
            #     smarker = "kx-"
            smarker = "kx-"
            waves = self.get_combineddata(s, self.clickmaps[s], lineterm=self.term)
            if waves is None:
                print(f"Malformed data set for run {s:s}. Continuing")
                continue
            waves = waves[::-1]  # reverse order to match spls
            t = np.linspace(
                0, waves.shape[1] * self.sample_rate * 1000.0, waves.shape[1]
            )
            spls = self.clickmaps[s]["SPLs"]  # get spls
            self.clickdata[s] = {
                "waves": waves,
                "timebase": t,
                "spls": spls,
                "marker": smarker,
            }

    def getToneData(self, select):
        """
        Gets the tone map data for the current selection
        The resulting data is held in a dictionary structured as
        {mapidentity: dict of frequencies}
        Where each dictoffrequencies holds a dict of {waves, time, spls and optional marker}

        """
        self.tonemapdata = {}
        select, freqs = self.adjustSelection(select, tone=True)
        print("self.tonemaps: ", self.tonemaps)


        # convert select to make life easier
        # select list should have lists of strings ['0124', '0244'] or Nones...
        print("tonemaps: ", self.tonemaps)
        print("select: ", select)
        # iterate through the directories, looking for tone maps
        for i, s in enumerate(self.tonemaps.keys()):
            freqs = []
            if select is not None:
                if s[9:] not in select:
                    continue
            print(self.tonemaps[s])

            for f in self.tonemaps[s]["Freqs"]:
                if f not in freqs:
                    freqs.append(f)
            freqs.sort()
            print("freqs: ", freqs)
            if len(freqs) == 0:  # check of no tone pip ABR data in this directory
                continue
            # now we can build the tonemapdata
            self.tonemapdata[s] = OrderedDict()
            for fr in self.tonemaps[s]["Freqs"]:
                waves = self.get_combineddata(s, self.tonemaps[s], freq=fr)
                if waves is None:
                    print(f"Malformed data set for run {s:s}. Continuing")
                    continue
                t = np.linspace(
                    0, waves.shape[1] * self.sample_rate * 1000.0, waves.shape[1]
                )
                spls = self.tonemaps[s]["SPLs"]
                self.tonemapdata[s][fr] = {
                    "waves": waves,
                    "timebase": t,
                    "spls": spls,
                    "marker": "ko-",
                }

    def plotClicks(
        self,
        select=None,
        plottarget=None,
        IOplot=None,
        PSDplot=None,
        superIOPlot=None,
        colorindex=0,
    ):
        """
        Plot the click ABR intensity series, one column per subject, for one subject

        Parameters
        ----------
        select : list of str (default : None)
            A list of the times for the datasets to plot for each subject.
            If None, then all detected click ABR runs for that subject are superimposed.
            The list looks like [['0115'], None, None]  - for each directory in order.

        plottarget : matplotlib axis object
            The axis to plot the data into.

        IOPlot : Matplotlib Axes object
            Input output plot target. If not None, then use the specified Mabplotlib Axes for the IO plot

        PSDPlot : Matplotlib Axes object
            Power spectral density plot target. If not None, then use the specified Mabplotlib Axes for the plot

        superIOPlot : Matplotlib Axes object
            Input output plot target. If not None, then use the specified Mabplotlib Axes for the IO plot

        """
        # get data for clicks and plot all on one plot
        A = abr_analyzer.Analyzer(sample_frequency=self.sample_freq)
        thrs = {}
        icol = colorindex
        for index, s in enumerate(list(self.clickdata.keys())):
            datatitle = f"{str(self.datapath.parts[-1]):s}\n{s:s}"
            # datatitle = datatitle.replace('_', '\_')  # if TeX is enabled, will need to escape the underscores
            waves = self.clickdata[s]["waves"]
            t = self.clickdata[s]["timebase"]
            spls = self.clickdata[s]["spls"]

            A.analyze(t, waves)
            p1, n1, p2 = A.p1n1p2
            # print("PNP: ", p1, n1, p2)
            thr_spl = A.threshold_spec(waves, spls, tr=[2.0, 6.0], reftimes=[20, 25], spec_bandpass=self.spec_bandpass, SD=4.0)
            thrs[s] = thr_spl

            halfspl = np.max(spls) / 2.0
            
            # generate a line demacating the P1 (first wave)
            latmap = []
            spllat = []
            for j in range(len(spls)):
                if spls[j] > thrs[s]:
                    latmap.append(t[p1[j][0]])  # get latency for first value
                    spllat.append(spls[j])
            if len(latmap) > 2:
                latp = np.polyfit(spllat, latmap, 1)
                fitline = np.polyval(latp, spls)

            linewidth = 1.0
            IO = np.zeros(len(spls))
            sf = 8
            sf_cvt = 1e6
            if index == 0:
                x = [0.5, 0.5]
                y = np.array([0, 1e-6])*sf*sf_cvt + 100.0  # put at 100 dB... 
                plottarget.plot(x, y, linewidth=1.5)  # put 1 uV cal bar at highest sound level
                plottarget.text(x[0]+0.1, np.mean(y), s=r"1 $\mu V$",  ha="left", va="center", fontsize=7)
            for j in range(len(spls)):
                if spls[j] == thr_spl:  # highlight the threshold spl
                    plottarget.plot(
                        t,
                        0 * waves[j] * sf_cvt + spls[j],
                        color=[0.5, 0.5, 0.5, 0.4],
                        linewidth=5,
                    )
                try:
                    c = self.color_map[spls[j]]
                    plottarget.plot(
                        t, sf * waves[j] * sf_cvt + spls[j], color=c, linewidth=linewidth
                    )
                except:
                    plottarget.plot(
                        t, sf * waves[j] * sf_cvt + spls[j], color="k", linewidth=linewidth
                    )
                
                for p in p1[j]:
                    plottarget.plot(
                        t[p], sf * waves[j][p] * sf_cvt + spls[j], "ro", markersize=2
                    )
                for p in n1[j]:
                    plottarget.plot(
                        t[p], sf * waves[j][p] * sf_cvt + spls[j], "bo", markersize=2
                    )
                if len(latmap) > 2:
                    plottarget.plot(fitline, spls, "g-", linewidth=0.7)
                if spls[j] >= thr_spl or len(latmap) <= 2:
                    IO[j] = sf_cvt * (waves[j][p1[j][0]] - waves[j][n1[j][0]])
                else:
                    ti = int(fitline[j] / (self.sample_rate * 1000.0))
                    IO[j] = sf_cvt * waves[j][ti]
                    

            if superIOPlot is not None:  # superimposed IO plots
                datatitle_short = f"{str(self.datapath.parts[-1]):s}/{s:s}"
                superIOPlot.plot(
                    spls,
                    sf_cvt * A.ppio,
                    self.clickdata[s]["marker"],
                    color=self.summaryClick_color_map[icol % self.max_colors],
                    label=datatitle_short,
                )

                # print out the data for import into another plotting program, such as Prism or Igor
                print("*" * 20)
                print(s)
                print(f"dataset: {s:s}")
                print("t\tV")
                for i in range(len(spls)):
                    print(f"{spls[i]:.1f}\t{IO[i]:.3f}")
                print("*" * 20)

            if IOplot is not None:  # generic io plot for cell
                IOplot.set_title(datatitle,  y=0.95,fontdict={"fontsize": 7, "ha": "center", 'va': "top"})  # directory plus file
                IOplot.plot(
                    spls,
                    sf_cvt * A.ppio,
                    marker=A.ppioMarker,
                    markersize=3,
                    color=self.summaryClick_color_map[icol % self.max_colors],
                    label="P-P",
                )
                IOplot.plot(
                    spls,
                    sf_cvt * A.rms_response,
                    marker=A.rmsMarker,
                    markersize=3,
                    color=self.summaryClick_color_map[icol % self.max_colors],
                    label="RMS signal",
                )
                IOplot.plot(
                    spls,
                    sf_cvt * A.rms_baseline,
                    marker=A.baselineMarker,
                    markersize=3,
                    color="k",
                    label="RMS baseline",
                )
                IOplot.set_ylim(0, 6.0)  # microvolts
                IOplot.set_ylabel(f"ABR (uV)")

                if self.psdIOPlot:
                    ax2 = IOplot.twinx()
                    ax2.plot(
                        spls,
                        A.psdwindow,
                        marker=A.psdMarker,
                        color="r",
                        label="PSD signal",
                        markersize=3,
                    )
                    ax2.tick_params("y", colors="r")
                if index == 0 and icol == 0:
                    handles, labels = IOplot.get_legend_handles_labels()
                    legend = IOplot.legend(loc="upper left")
                    for label in legend.get_texts():
                        label.set_fontsize(6)

            if PSDplot is not None:  # power spectral density
                for j in range(len(spls)):
                    PSDplot.semilogy(
                        np.array(A.fr), np.array(A.psd[j])
                    )  # color=self.color_map[spls])
                PSDplot.set_ylim(1e-6, 0.01)
                PSDplot.set_xlim(100.0, 2000.0)
        plottarget.set_xlim(0, 8.0)
        plottarget.set_ylim(10.0, 115.0)
        plottarget.set_title(datatitle, y=1.0,fontdict={"fontsize": 7, "ha": "center", 'va': "top"})

        if superIOPlot is not None:
            legend = superIOPlot.legend(loc="upper left")
            for label in legend.get_texts():
                label.set_fontsize(5)

        print("")
        for s in list(thrs.keys()):
            print(f"dataset: {s:s}  thr={thrs[s]:.0f}")
        print("")

    def plotTones(self, select=None, pdf=None):
        """
        Plot the tone ABR intensity series, one column per frequency, for one subject

        Parameters
        ----------
        select : list of str (default : None)
            A list of the times for the datasets to plot for each subject.
            If None, then all detected tone ABR runs for that subject/frequency
            comtination are superimposed.
            The list looks like [['0115'], None, None]  - for each directory in order.

        pdf : pdfPages object
            The pdfPages object that the plot will be appended to. Results in a multipage
            pdf, one page per subject.

        """
        self.thrs = {}  # holds thresholds for this dataset
        freqs = []
        for run in list(self.tonemapdata.keys()):  # just passing one dict...
            freqs.extend(list(self.tonemapdata[run].keys()))
        freqs.sort()
        if len(freqs) == 0:  # check of no tone pip ABR data in this directory
            return

        f, axarr = mpl.subplots(1, len(freqs), figsize=(12, 6), num="Tones")
        datatitle = str(Path(self.datapath.parent, self.datapath.name))
        datatitle = datatitle.replace("_", "\_")
        f.suptitle(datatitle)
        A = abr_analyzer.Analyzer()
        for i, s in enumerate(self.tonemapdata.keys()):
            thr_spls = np.zeros(len(self.tonemaps[s]["Freqs"]))
            for k, fr in enumerate(self.tonemaps[s]["Freqs"]):  # next key is frequency
                waves = self.tonemapdata[s][fr]["waves"]
                t = self.tonemapdata[s][fr]["timebase"]
                spls = self.tonemapdata[s][fr]["spls"]
                A.analyze(t, waves)
                thr_spl = A.thresholds(waves, spls, SD=3.5)
                thr_spls[k] = thr_spl
                plottarget = axarr[freqs.index(fr)]
                for j in range(len(spls))[::-1]:
                    if spls[j] == thr_spl:
                        plottarget.plot(
                            t,
                            0 * waves[j] * 1e6 + spls[j],
                            color=[0.5, 0.5, 0.5, 0.4],
                            linewidth=5,
                        )
                    plottarget.plot(
                        t, 4 * waves[j] * 1e6 + spls[j], color=self.color_map[spls[j]]
                    )
                plottarget.set_xlim(0, 8.0)
                plottarget.set_ylim(10.0, 110.0)
                frtitle = "%.1f kHz" % (float(fr) / 1000.0)
                plottarget.title.set_text(frtitle)
                plottarget.title.set_size(9)
            self.thrs[s] = [self.tonemaps[s]["Freqs"], thr_spls]
        PH.cleanAxes(axarr)
        if pdf is not None:
            pdf.savefig()
            mpl.close()

    def plotToneThresholds(self, allthrs, num):
        """
        Make a plot of the tone thresholds for all of the datasets
        Data are plotted against a log frequency scale (2-64kHz)
        Data is plotted into the current figure.

        Parameters
        ----------
        allthrs : dict
        A dictionary holding all the threshold information. The following
        structure is required:
            Keys: filenames for each dataset
            Values a dict of thresholds. The keys are the names of the tone maps
            (because more than one tone map may be combined)
            The values are tuples of (frequency, threshold)

        Returns
        -------
        Nothing
        """
        f, ax = mpl.subplots(nrows=1, ncols=1, num=num)
        ax.set_xscale("log", nonpositive="clip", base=2)
        n_datasets = len(list(allthrs.keys()))
        print("Datasets found: ", n_datasets)
        c_map = self.makeColorMap(n_datasets, list(allthrs.keys()))

        thrfrs = {}
        for i, d in enumerate(allthrs):  # for all the datasets
            print("d: ", d)
            print(allthrs[d])
            for m in allthrs[d]:  # for all the maps in the dataset combined
                ax.scatter(
                    np.array(allthrs[d][m][0]) / 1000.0,
                    allthrs[d][m][1],
                    c=c_map[d],
                    s=12,
                )
                for j, f in enumerate(allthrs[d][m][0]):
                    if f not in list(thrfrs.keys()):
                        thrfrs[f] = [allthrs[d][m][1][j]]
                    else:
                        thrfrs[f].append(allthrs[d][m][1][j])
        # sort the threshold list
        thrs_sorted = OrderedDict(sorted(list(thrfrs.items()), key=lambda t: t[0]))
        frmean = np.zeros(len(list(thrs_sorted.keys())))
        frstd = np.zeros(len(list(thrs_sorted.keys())))

        for i, f in enumerate(thrs_sorted):
            #            print ('i, f: ', i, f)
            #            print (thrs_sorted[f])
            frmean[i] = np.nanmean(thrs_sorted[f])
            frstd[i] = np.nanstd(thrs_sorted[f])
        ax.errorbar(
            np.array(list(thrs_sorted.keys())) / 1000.0, frmean, yerr=frstd, fmt="o"
        )
        ax.set_xlim(1.8, 65.0)
        xt = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
        mpl.xticks(xt, [str(x) for x in xt])
        return (thrs_sorted, frmean, frstd)



    def get_combineddata(self, datasetname, dataset, freq=None, lineterm="\r"):
        """
        Read the data sets and combine the p (condensation) and n
        (rarefaction) data sets for alternating polarity stimuli.

        Parameters
        ----------
        datasetname : str
            yyyymmdddd-time format for start of dataset name
        dataset : dict
            dictionary for the dataset that identifies the stimulus
            type, the SPLs in the dataset, and the frequency if the dataset
            is a tone pip run
        freq : float (default: None)
            for tone maps, the specific frequency intensity series to return

        lineterm: str (default: '\r')
            line terminator for this dataset

        Returns
        -------
        waves : numpy array
            Waveforms, as a nxm array, where n is the number of intensities,
            and m is the length of each waveform

        """
        #        print dataset
        if dataset["stimtype"] == "click":
            fnamepos = datasetname + "-p.txt"
            fnameneg = datasetname + "-n.txt"
            waves = self.read_dataset(fnamepos, fnameneg, lineterm)
            return waves
        if dataset["stimtype"] == "tonepip":
            fnamepos = datasetname + "-p-%.3f.txt" % freq
            fnameneg = datasetname + "-n-%.3f.txt" % freq
            waves = self.read_dataset(fnamepos, fnameneg, lineterm)
            return waves

    def read_dataset(self, fnamepos, fnameneg, lineterm="\r"):
        """
        Read a dataset, combining the positive and negative recordings,
        which are stored in separate files on disk. The waveforms are averaged
        which helps to minimize the CAP contribution.
        The waveforms are then bandpass filtered to remove the low frequency
        "rumble" and excess high-frequency noise.

        Parameters
        ----------
        fnamepos : str
             name of the positive (condensation) file
        fnameneg : str
            name of the negative (rarefaction) file
        lineterm: str
            line terminator used for this file set

        Returns
        -------
        waveform
            Waveform, as a nxm array, where n is the number of intensities,
            and m is the length of each waveform

        """
        # handle missing files.
        if not Path(self.datapath, fnamepos).is_file():
            return None
        if not Path(self.datapath, fnameneg).is_file():
            return None
        
        posf = pd.io.parsers.read_csv(
            Path(self.datapath, fnamepos),
            delim_whitespace=True,
            lineterminator=lineterm,
            skip_blank_lines=True,
            header=0,
        )
        negf = pd.io.parsers.read_csv(
            Path(self.datapath, fnameneg),
            delim_whitespace=True,
            lineterminator=lineterm,
            skip_blank_lines=True,
            header=0,
        )

        negseries = []
        posseries = []
        for col in negf.columns:
            for i in range(len(negf)):
                negseries.append(negf[col][i])

        for col in posf.columns:
            for i in range(len(posf)):
                posseries.append(posf[col][i])

        wvfmdata = [(x + y) / 2 for x, y in zip(negseries, posseries)]
        #        wvfmdata = negseries  # just get one polarity
        d1 = int(len(wvfmdata) / len(posf[col]))
        waves = np.reshape(wvfmdata, (d1, len(posf[col])))
        for i, w in enumerate(waves):
            waves[i, -1] = waves[i, -2]  # remove nan from end of waveform...
            waves[i, :] = self.filter(
                waves[i, :], 4, self.lpf, self.hpf, samplefreq=self.sample_freq
            )
            if self.invert:
                waves[i, :] = -waves[i, :]
        return waves

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
        print(f"nans: ", np.argwhere(np.isnan(signal)))
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


def do_clicks(dsname, mode, top_directory, dirs):
    if "clickselect" in list(ABR_Datasets[dsname].keys()):
        clicksel = ABR_Datasets[dsname]["clickselect"]
    else:
        clicksel = [None] * len(dirs)
    m, n = PH.getLayoutDimensions(len(clicksel))
    # rowlen = 8.0
    # m = int(np.ceil(len(clicksel) / rowlen))
    # if m == 1:
    #     n = len(clicksel)
    # else:
    #     n = int(rowlen)
    if m > 1:
        h = 2.5 * m
    else:
        h = 3
    f, axarr = mpl.subplots(m, n, figsize=(12, h), num="Click Traces")
    for ax in axarr:
        PH.nice_plot(ax)
    f2, axarr2 = mpl.subplots(m, n, figsize=(12, h), num="Click IO Summary")
    for ax in axarr2:
        PH.nice_plot(ax)
    #        f3, axarr3 = mpl.subplots(m, n, figsize=(12, h))
    f4, IOax = mpl.subplots(1, 1, figsize=(6, 6), num="Click IO Overlay")
    PH.nice_plot(IOax)

    if axarr.ndim > 1:
        axarr = axarr.ravel()
    if axarr2.ndim > 1:
        axarr2 = axarr2.ravel()
    fofilename = Path(top_directory, "ClickSummary.pdf")
    nsel = len(clicksel)
    print("Nsel: ", nsel)
    for icol, k in enumerate(range(nsel)):
        P = ABR(Path(top_directory, dirs[k]), mode, info=ABR_Datasets[dsname])
        if icol == 0:
            P.summaryClick_color_map = P.makeColorMap(nsel, list(range(nsel)))
        P.getClickData(select=clicksel[k])
        P.plotClicks(
            select=clicksel[k],
            plottarget=axarr[k],
            superIOPlot=IOax,
            IOplot=axarr2[k],
            colorindex=icol,
        )
    mpl.figure("Click Traces")
    mpl.savefig(fofilename)
    mpl.figure("Click IO Summary")
    fo2filename = Path(top_directory, "ClickIOSummary.pdf")
    mpl.savefig(fo2filename)
    mpl.figure("Click IO Overlay")
    fo4filename = Path(top_directory, "ClickIOOverlay.pdf")
    mpl.savefig(fo4filename)
    mpl.show()


def do_tones(dsname, mode, top_directory, dirs):
    if "toneselect" in list(ABR_Datasets[dsname].keys()):
        tonesel = ABR_Datasets[dsname]["toneselect"]
    else:
        tonesel = [None] * len(dirs)

    fofilename = Path(top_directory, "ToneSummary.pdf")
    allthrs = {}
    with PdfPages(fofilename) as pdf:
        for k in range(len(tonesel)):
            P = ABR(Path(top_directory, dirs[k]), mode)
            P.getToneData(select=tonesel[k])
            P.plotTones(select=tonesel[k], pdf=pdf)
            allthrs[dirs[k]] = P.thrs
    population_thrdata = P.plotToneThresholds(allthrs, num="Tone Thresholds")
    print(population_thrdata)
    print("Hz\tmean\tstd\t individual")
    for i, f in enumerate(population_thrdata[0].keys()):
        print(
            f"{f:.1f}\t{population_thrdata[1][i]:.1f}\t{population_thrdata[2][i]:.1f}",
            end="",
        )
        for j in range(len(population_thrdata[0][f])):
            print(f"\t{population_thrdata[0][f][j]:.1f}", end="")
        print("")

    tthr_filename = Path(top_directory, "ToneThresholds.pdf")
    mpl.savefig(tthr_filename)
    mpl.show()


def main():
    if len(sys.argv) > 1:
        dsname = sys.argv[1]
        mode = sys.argv[2]
    else:
        print("Missing command arguments; call: plotABRs.py datasetname [click, tone]")
        exit(1)

    if dsname not in list(ABR_Datasets.keys()):
        print(list(ABR_Datasets.keys()))
        raise ValueError("Data set %s not found in our list of known datasets")
    if mode not in ["tones", "clicks"]:
        raise ValueError("Second argument must be tones or clicks")

    top_directory = Path(basedir, ABR_Datasets[dsname]["dir"])

    dirs = [
        tdir
        for tdir in Path(top_directory).glob("*")
        if Path(top_directory, tdir).is_dir()
    ]
    print("found dirs: ", dirs)
    if mode == "clicks":
        do_clicks(dsname, mode, top_directory, dirs)

    elif mode == "tones":
        do_tones(dsname, mode, top_directory, dirs)
    else:
        raise ValueError(f"Mode is not known: {mode:s}")


if __name__ == "__main__":
    main()
