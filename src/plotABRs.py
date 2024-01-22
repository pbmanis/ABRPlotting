"""PlotABRs

Plot ABR data from our matlab program. This relies on an excel file
to specify experiments (ABRs.xlsx). 
The directory for each subject is expected to be named in a
way that identifies the subject and experimental group(s). This allows us to
make plots that correctly assign each subject with markers and labels. 
However, not everyone did this in a consistent manner.

Organization of ABRs.xlsx file includes:
Dataset: This column refers to the dataset (may be the person who did the experiments, 
or a subset of their experiments, or a particular set of experiments).
Subject: This column refers to the subject (mouse) that was tested. Not all mice have a Subject ID.
Treatment: This column refers to the treatment group (e.g., WT, KO, etc.)
Genotype: This column refers to the genotype of the mouse (e.g., WT, KO, Het, strain, etc.)
Cross: This column identifies the type of cross that led to the genotyp (sometimes).
Strain: This column identifies the strain of the mouse/mouse line (e.g., CBA, C57, DBA, etc.)

Note that not all of these fields are filled in at present, depending on the detailed
data that is in the notebook/file notes and whether I am using the data or not.


The "codefile" may have further information that links the subjects to their treatment if
the study is blinded. 
In this case, we need to cross-reference the codefile Animal_ID (renamed here to Subject) with the
Animal_identifier, and the DataDirectory in ABRS.xlsx.
Also, this assigns the "Group" field.

"""
import ast
import importlib
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pylibrary.plotting.plothelpers as PH
import pylibrary.tools.cprint as CP
import scikit_posthocs
import seaborn as sns  # makes plot background light grey with grid, no splines. Remove for publication plots
import statsmodels
import statsmodels.api as sm
from matplotlib import pyplot as mpl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from mpl_toolkits.axes_grid1 import Divider, Size
from statsmodels.formula.api import ols
import pprint
import pyqtgraph
import src.abr_analyzer
import src.ABR_dataclasses as ABRDC
import src.ABRFuncs as ABRFuncs
import src.getcomputer  # stub to return the computer and base directory
from ABR_Datasets import ABR_Datasets  # just the dict describing the datasets
from src.ABR_dataclasses import ABR_Data
from src.get_configuration import get_configuration
from pyqtgraph.Qt.QtCore import QObject

# from ephys.tools import fitting  # use to fit hill function to IO curve.

pp = pprint.PrettyPrinter(indent=4)

experiments = get_configuration("/Users/pbmanis/Desktop/Python/mrk-nf107/config/experiments.cfg")
experimentnames = experiments[0]
e = experimentnames.index("NF107Ai32_NIHL")
experimentname = experiments[0][e]
Experiment = experiments[1][experimentname]


# basedir, computer_name = getcomputer.getcomputer()

ABRF = ABRFuncs.ABRFuncs()


@dataclass
class plotinfo:
    """A data class to hold information about the plots (to be passed around)"""

    P: object = None
    Plot_f: object = None
    Plot_f2: object = None
    Plot_f4: object = None
    IOax: object = None
    m: int = 0
    n: int = 0
    icol: int = 0
    nrows: int = 1
    ncols: int = 1
    axarr: object = None
    axarr2: object = None


class PData(object):
    def __init__(self, li_obj):
        self.obj = li_obj


class ABR:
    """
    Read an ABR data set from the matlab program
    Provides functions to plot the traces.
    """

    def __init__(
        self,
        datapath: Union[Path, str],
        mode: str = "clicks",
        info: object = ABRDC.ABR_Data(),
        datasetname: str = "",
        subject: str = "",
    ):
        """
        Parameters
        ----------
        datapath : str
            Path to the datasets. The data sets are expected to be collected
            under this path into individual directories, each with the results
            of the ABR runs for one subject.
        mode : string ('clicks' or 'tones')
            Specify type of data in the data set
        info : dict
            Dictionary with the following keys:
                invert : boolean (default : False)
                    Changes sign of waveform if True
                minlat : float (default 0.75)
                    minimum latency for event detection
                term : float (default '\r')
                    line terminator (may change if the data has been read by an
                    editor or is on a windows vs. mac vs linux system)
        """

        # Set some default parameters for the data and analyses

        if info.sample_freq is not None:
            self.sample_freq = info.sample_freq  # Hz
        else:
            self.sample_freq = 100000.0

        self.sample_rate = (
            1.0 / self.sample_freq
        )  # standard interpolated sample rate for matlab abr program: 10 microsecnds
        if info.spec_bandpass is not None:
            self.spec_bandpass = info.spec_bandpass
        else:
            self.spec_bandpass = [800.0, 1200.0]
        if info.showdots:  # turn off dot plotting.
            self.show_dots = info.showdots
        else:
            self.show_dots = True

        self.df_excel = pd.read_excel("ABRS.xlsx", sheet_name="Sheet1")
        self.df_excel = self.df_excel[self.df_excel.Group == datasetname]

        self.dev = 3.0  # should put this in the table
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
        self.datasetname = datasetname

        self.subject = (
            subject  # this is the name of the directory that the data is in, not the full path
        )
        self.term = info.term
        self.minlat = info.minlat
        self.invert = (
            info.invert
        )  # data flip... depends on "active" lead connection to vertex (false) or ear (true).

        # find the matching mouseinfo in the coding file if it exists
        print("\nInfo: ", info)
        if info.codefile is not None:
            code_filename_full = Path("src", Path(info.codefile).name)
            if code_filename_full.is_file():
                code_filename = str(Path("src", Path(info.codefile).stem))
                code_filename = code_filename.replace("/", ".")
                CodeFile = importlib.import_module(code_filename)
                print("Imported Codefile: ", CodeFile)
                raise ValueError("Code files read this way are deprectated")
        else:
            CodeFile = None
        # print("Codefile: ", CodeFile)
        self.characterizeDataset(CodeFile)

        # build color map where each SPL is a color (cycles over 12 levels)
        self.max_colors = 25
        bounds = np.linspace(0, 120, self.max_colors)  # 0 to 120 db inclusive, 5 db steps
        color_labels = np.unique(bounds)
        self.color_map = ABRF.makeColorMap(self.max_colors, color_labels)
        color_labels2 = list(range(self.max_colors))
        self.summaryClick_color_map = ABRF.makeColorMap(
            self.max_colors, list(range(self.max_colors))
        )  # make a default map, but overwrite for true number of datasets
        self.psdIOPlot = False
        self.superIOLabels = []

    def characterizeDataset(self, codefile=None):
        """
        Look at the directory in datapath, and determine what datasets are
        present. A dataset consists of at least 3 files: yyyymmdd-HHMM-SPL.txt :
        the SPL levels in a given run, with year, month day hour and minute
        yyyymmdd-HHMM-{n,p}-[freq].txt : hold waveform data for a series of SPL
        levels
            For clicks, just click polarity - holds waveforms from the run
            typically there will be both an n and a p file because we use
            alternating polarities. For tones, there is also a number indicating
            the tone pip frequency used.

        The runs are stored in separate dictionaries for the click and tone map
        runs, including their SPL levels (and for tone maps, frequencies).

        Parameters
        ----------
        codefile

        Returns
        -------
        Nothing
        """

        # x = ABRF.get_matlab()
        self.spls = ABRF.getSPLs(self.datapath)

        self.freqs = ABRF.getFreqs(self.datapath)
        # print("self.spls: ", self.spls, self.datapath)
        # A click run will consist of an SPL, n and p file, but NO additional files.
        self.clicks = {}
        for s in self.spls.keys():
            if s not in list(self.freqs.keys()):  # skip SPL files associated with a tone map
                self.clicks[s] = self.spls[s]

        # inspect the directory and get a listing of all the tone and click maps
        # that can be found.
        self.tonemaps = {}

        for i, f in enumerate(list(self.freqs.keys())):
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

        # print("click keys: ", self.clicks.keys())
        for i, s in enumerate(list(self.clicks.keys())):
            self.clickmaps[s] = {
                "stimtype": "click",
                "SPLs": self.spls[s],
                "mouseinfo": None,
            }
            if self.mode == "clicks":
                # if i == 0:
                #     print("\n  Click Intensity Runs: ")
                # print(f"    Run: {s:s}")
                if codefile is not None:
                    mouseinfo = codefile.find_clickrun(s)
                    if mouseinfo is not None:
                        self.clickmaps[s]["mouseinfo"] = mouseinfo
                        # print(
                        #     "        ",
                        #     self.clickmaps[s],
                        #     "Mouse ID: ",
                        #     self.clickmaps[s]["mouseinfo"].ID,
                        # )
                else:
                    print("        ", self.clickmaps[s], "NO Mouse ID: ")

    def adjustSelection(self, select, tone=False):
        """Clean up the list of files that will be processed.

        Parameters
        ----------
        select : list
            list of recordings
        tone : bool, optional
            Determine whether also dealing with tone files.
            By default False

        Returns
        -------
        select (list),
        Freqs (list)

        """
        freqs = []
        if select is None:
            return select, freqs
        for i, s in enumerate(select):
            if s is None:
                continue
            # if isinstance(s, int):
            #     select[i] = "%04d" % s  # convert to string
            # if isinstance(s, str) and len(s) < 4:  # make sure size is ok
            #     select[i] = "%04d" % int(s)
            if tone:
                base = list(self.tonemaps.keys())[0][:9]
                for f in self.tonemaps[base + select[i]]["Freqs"]:
                    if f not in freqs:
                        freqs.append(f)
        return select, freqs

    def getClickData(self, select: str = "", directory: Union[Path, str] = None):
        """
        Gets the click data for the current selection The resulting data is held
        in a dictionary structured as {mapidentity: dict of {waves, time, spls
        and optional marker}

        Parameters
        ----------
        select : which data to select
        directory : str
            The directory of the data (used to set marker style)

        """
        select, freqs = self.adjustSelection(select)
        # get data for clicks and plot all on one plot
        self.clickdata = {}
        markerstyle, group = ABRF.getMarkerStyle(directory=directory, markers=self.info.markers)
        print("click maps: ", self.clickmaps.keys())
        for i, s in enumerate(self.clickmaps.keys()):
            if select is not None:
                if s[9:] not in select:
                    continue
            print("working it")
            waves, tb = self.get_combineddata(s, self.clickmaps[s], lineterm=self.term)
            if waves is None:
                print(f"Malformed data set for run {s:s}. Continuing")
                continue
            waves = waves[::-1]  # reverse order to match spls

            spls = self.clickmaps[s]["SPLs"]  # get spls
            self.clickdata[s] = {
                "waves": waves,
                "timebase": tb,
                "spls": spls,
                "marker": markerstyle,
                "group": group,
                "mouseinfo": self.clickmaps[s]["mouseinfo"],
            }
            # print("Populated clickdata: ", self.clickdata[s])

    def getToneData(self, select, directory: str = ""):
        """
        Gets the tone map data for the current selection The resulting data is
        held in a dictionary structured as {mapidentity: dict of frequencies}
        Where each dictoffrequencies holds a dict of {waves, time, spls and
        optional marker}

        """
        self.tonemapdata = {}
        select, freqs = self.adjustSelection(select, tone=True)

        # convert select to make life easier
        # select list should have lists of strings ['0124', '0244'] or Nones...
        markerstyle, group = ABRF.getMarkerStyle(directory=directory, markers=self.info.markers)

        # iterate through the files in the directory, looking for tone maps
        for i, s in enumerate(self.tonemaps.keys()):
            freqs = []
            if select is not None:
                if s[9:] not in select:
                    continue

            for f in self.tonemaps[s]["Freqs"]:
                if f not in freqs:
                    freqs.append(f)
            freqs.sort()
            if len(freqs) == 0:  # check of no tone pip ABR data in this directory
                continue
            # now we can build the tonemapdata
            self.tonemapdata[s] = OrderedDict()
            for fr in self.tonemaps[s]["Freqs"]:
                waves, tb = self.get_combineddata(s, self.tonemaps[s], freq=fr)
                if waves is None:
                    print(f"Malformed data set for run {s:s}. Continuing")
                    continue

                spls = self.tonemaps[s]["SPLs"]
                self.tonemapdata[s][fr] = {
                    "waves": waves,
                    "timebase": tb,
                    "spls": spls,
                    "marker": markerstyle,
                    "group": group,
                    "ID": self.clickmaps[s]["mouseinfo"].ID,
                }

    def plotClicks(
        self,
        select: str = None,
        datadir: Union[Path, str] = None,
        plottarget=None,
        IOplot=None,
        PSDplot=None,
        superIOPlot=None,
        colorindex: int = 0,
        show_y_label: bool = True,  # for the many-paneled plots
        show_x_label: bool = True,
    ) -> List:
        """
        Plot the click ABR intensity series for for one subject
        one column per subject,

        Parameters
        ----------
        select : list of str (default : None)
            A list of the times for the datasets to plot for each subject. If
            None, then all detected click ABR runs for that subject are
            superimposed. The list looks like [['0115'], None, None]  - for each
            directory in order.

        plottarget : matplotlib axis object
            The axis to plot the data into.

        IOPlot : Matplotlib Axes object
            Input output plot target. If not None, then use the specified
            Mabplotlib Axes for the IO plot

        PSDPlot : Matplotlib Axes object
            Power spectral density plot target. If not None, then use the
            specified Mabplotlib Axes for the plot

        superIOPlot : Matplotlib Axes object
            Input output plot target. If not None, then use the specified
            Mabplotlib Axes for the IO plot

        """
        # grab the excel data row.

        drow = self.df_excel[self.df_excel.Subject == self.subject]
        if drow.Sex.values[0] == "M":
            sex_marker = "x"
        elif drow.Sex.values[0] == "F":
            sex_marker = "o"
        else:
            sex_marker = "D"
        A = abr_analyzer.Analyzer(sample_frequency=self.sample_freq)
        thrs = {}
        icol = colorindex
        IO_DF = []  # build a dataframe of the IO funcitons from a list.
        print("PLOT CLICKS: ", self.clickdata.keys())
        for index, s in enumerate(list(self.clickdata.keys())):
            # datatitle = datatitle.replace('_', '\_')  # if TeX is enabled, will need to escape the underscores
            if self.clickdata[s]["mouseinfo"] is not None:
                datatitle = f"{self.clickdata[s]['mouseinfo'].ID:s}\nNE: {self.clickdata[s]['mouseinfo'].SPL:5.1f} dBSPL"
            else:
                datatitle = f"{str(self.datapath.parts[-1]):s}\n{s:s}"
            waves = self.clickdata[s]["waves"]
            t = np.array(self.clickdata[s]["timebase"])
            spls = self.clickdata[s]["spls"]
            ppio = np.zeros(len(spls))
            # Build a pandas dataframe to write a CSV file out.
            columns = ["time"]
            splnames = [f"{int(spl):d}" for spl in spls]
            columns.extend(splnames)
            waves_df = pd.DataFrame(columns=columns)
            # print(waves_df.head())
            waves_df["time"] = tuple(t)
            for i, spln in enumerate(splnames):
                waves_df[spln] = waves[i] * 1e6  # convert to microvolts for the CSV file
            waves_df.to_csv(Path("ABR_CSVs", f"{str(self.datapath.parts[-1]):s}" + ".csv"))
            A.analyze(t, waves, dev=self.dev)
            print("A.thresholds: ", A.thresholds)
            pnp = A.p1n1p2
            p1 = pnp["p1"]
            n1 = pnp["n1"]
            p2 = pnp["p2"]
            tb = A.set_baseline(timebase=t)
            thr_spl = A.thresholds(  # A.threshold_spec(
                waves,
                spls,
                response_window=[2.0, 6.0],
                baseline_window=tb,
                SD=3.0,
                # spec_bandpass=self.spec_bandpass,
            )
            thrs[s] = thr_spl

            halfspl = np.max(spls) / 2.0

            # generate a line demarcating the P1 (first wave)
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
                # plot a calibration bar for the voltage traces
                x = [7.5, 7.5]
                y = np.array([0, 1e-6]) * sf * sf_cvt + 105.0  # put at 105 dB...
                plottarget.plot(x, y, linewidth=1.5)  # put 1 uV cal bar at highest sound level
                plottarget.text(
                    x[0] + 0.1,
                    np.mean(y),
                    s=r"1 $\mu V$",
                    ha="left",
                    va="center",
                    fontsize=7,
                )
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
                        t,
                        sf * waves[j] * sf_cvt + spls[j],
                        color=c,
                        linewidth=linewidth,
                    )
                except:
                    plottarget.plot(
                        t,
                        sf * waves[j] * sf_cvt + spls[j],
                        color="k",
                        linewidth=linewidth,
                    )

                if self.show_dots:
                    for p in p1[j]:
                        plottarget.plot(
                            t[p],
                            sf * waves[j][p] * sf_cvt + spls[j],
                            "ro",
                            markersize=2,
                        )
                    for p in n1[j]:
                        plottarget.plot(
                            t[p],
                            sf * waves[j][p] * sf_cvt + spls[j],
                            "bo",
                            markersize=2,
                        )
                # if len(latmap) > 2 and self.show_dots:
                #     plottarget.plot(fitline, spls, "g-", linewidth=0.7)
                if len(latmap) > 2 and self.show_dots:
                    plottarget.plot(A.p1_latencies[0], A.p1_latencies[1], "g-", linewidth=0.7)

                if spls[j] >= thr_spl or len(latmap) <= 2:
                    IO[j] = sf_cvt * (waves[j][p1[j][0]] - waves[j][n1[j][0]])
                else:
                    ti = int(fitline[j] / (self.sample_rate * 1000.0))
                    if ti < len(waves[j]):
                        IO[j] = sf_cvt * waves[j][ti]

            if show_y_label:
                plottarget.set_ylabel("dBSPL")
            if show_x_label:
                plottarget.set_xlabel("T (ms)")
            plottarget.set_xlim(0, 10.0)
            plottarget.set_ylim(10.0, 115.0)
            PH.set_axes_ticks(
                plottarget,
                xticks=[0, 2, 4, 6, 8, 10],
                xticks_str=["0", "2", "4", "6", "8", "10"],
                x_minor=np.arange(0, 10, 0.5),
                yticks=[0, 40, 80, 120],
                yticks_str=["0", "40", "80", "120"],
                y_minor=[10, 20, 30, 50, 60, 70, 90, 100, 110],
            )
            plottarget.set_title(
                datatitle,
                x=0.5,
                y=1.00,
                fontdict={"fontsize": 6, "ha": "center", "va": "bottom"},
                transform=plottarget.transAxes,
            )

            if superIOPlot is not None:  # Plot superimposed IO curves
                # datatitle_short = f"{str(self.datapath.parts[-1]):s}/{s:s}"
                # if self.clickdata[s]["group"] not in self.superIOLabels:
                #     self.superIOLabels.append(self.clickdata[s]["group"])
                #     label = self.superIOLabels[-1]
                # else:
                # print("s: ", s)
                # print(self.clickdata[s])
                if self.clickdata[s]["mouseinfo"] is not None:
                    label = f"{self.clickdata[s]['mouseinfo'].ID:s}({self.clickdata[s]['mouseinfo'].SPL:5.1f})"
                else:
                    label = f"{str(self.datapath.parts[-1]):s}\n{s:s}"
                # if "ID" in self.clickdata[s]:
                #     label = self.clickdata[s]["ID"]
                # else:
                #     label = s
                self.superIOLabels.append(label)

                superIOPlot.plot(
                    spls,
                    sf_cvt * A.ppio,
                    marker=sex_marker,  # self.clickdata[s]["marker"],
                    linestyle="-",
                    color=self.summaryClick_color_map[icol % self.max_colors],
                    label=label,
                )
                sr = datatitle.split("\n")
                if len(sr) > 1:
                    subjectID = sr[0]
                    run = sr[1]
                else:
                    run = 0
                    subjectID = datatitle
                for i_level, spl in enumerate(spls):
                    IO_DF.append(
                        [
                            subjectID,
                            run,
                            spl,
                            sf_cvt * A.ppio[i_level],
                            thrs[s],
                            self.clickdata[s]["group"],
                        ]
                    )
                    # IO_DF = [spls, (sf_cvt*A.ppio).tolist(), str(self.clickdata[s]["group"])]

                # print out the data for import into another plotting program, such as Prism or Igor
                print("*" * 20)
                print(s)
                print(f"dataset: {s:s}")
                print("t\tV")
                for i in range(len(spls)):
                    print(f"{spls[i]:.1f}\t{IO[i]:.3f}")
                print("*" * 20)

            if IOplot is not None:  # generic io plot for cell
                IOplot.set_title(
                    datatitle,
                    x=0.5,
                    y=1.0,
                    fontdict={"fontsize": 7, "ha": "center", "va": "bottom"},
                    transform=IOplot.transAxes,
                )  # directory plus file
                PH.nice_plot(IOplot, position=-0.03, direction="outward", ticklength=3)
                PH.set_axes_ticks(
                    IOplot,
                    xticks=[0, 25, 50, 75, 100],
                    xticks_str=["0", "25", "50", "75", "100"],
                    yticks=range(0, 7),
                    yticks_str=[f"{y:d}" for y in range(0, 7)],
                    y_minor=0.5 + np.arange(0, 6),
                )

                IOplot.plot(
                    spls,
                    sf_cvt * A.rms_baseline,
                    marker=A.baselineMarker,
                    markersize=3,
                    color="grey",
                    label="RMS baseline",
                    alpha=0.35,
                    clip_on=False,
                )
                IOplot.plot(
                    spls,
                    sf_cvt * A.ppio,
                    marker=sex_marker,  # A.ppioMarker,
                    markersize=3,
                    color=self.summaryClick_color_map[icol % self.max_colors],
                    label="P-P",
                    clip_on=False,
                )
                IOplot.plot(
                    spls,
                    sf_cvt * np.sqrt(A.rms_response**2 - A.rms_baseline**2),
                    marker=sex_marker,  # A.rmsMarker,
                    markersize=3,
                    color=self.summaryClick_color_map[icol % self.max_colors],
                    label="RMS signal",
                    clip_on=False,
                )

                IOplot.set_ylim(0, 6.0)  # microvolts
                if show_y_label:
                    IOplot.set_ylabel(f"ABR ($\mu V$)")
                if show_x_label:
                    IOplot.set_xlabel("Click level (dBSPL)")

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
                    legend = IOplot.legend(loc="center left")
                    for label in legend.get_texts():
                        label.set_fontsize(6)

            if PSDplot is not None:  # power spectral density
                for j in range(len(spls)):
                    PSDplot.semilogy(
                        np.array(A.fr), np.array(A.psd[j])
                    )  # color=self.color_map[spls])
                PSDplot.set_ylim(1e-6, 0.01)
                PSDplot.set_xlim(100.0, 2000.0)

        if superIOPlot is not None:
            PH.set_axes_ticks(
                superIOPlot,
                xticks=[0, 25, 50, 75, 100],
                xticks_str=["0", "25", "50", "75", "100"],
                yticks=range(0, 7),
                yticks_str=[f"{y:d}" for y in range(0, 7)],
                y_minor=0.5 + np.arange(0, 6),
            )
            legend = superIOPlot.legend(loc="upper left")
            for label in legend.get_texts():
                label.set_fontsize(5)
            if show_x_label:
                superIOPlot.set_xlabel("Click level (dBSPL)")
            if show_y_label:
                superIOPlot.set_ylabel(f"ABR ($\mu V$)")

        print("-" * 40)
        for s in list(thrs.keys()):
            print(f"dataset: {s:s}  thr={thrs[s]:.0f}")
        print("-" * 40)
        self.thrs = thrs
        return IO_DF

    def plotTones(self, select=None, pdf=None):
        """
        Plot the tone ABR intensity series, one column per frequency, for one
        subject

        Parameters
        ----------
        select : list of str (default : None)
            A list of the times for the datasets to plot for each subject. If
            None, then all detected tone ABR runs for that subject/frequency
            comtination are superimposed. The list looks like [['0115'], None,
            None]  - for each directory in order.

        pdf : pdfPages object
            The pdfPages object that the plot will be appended to. Results in a
            multipage pdf, one page per subject.

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
        if len(freqs) == 1:
            axarr = [axarr]
        for i, s in enumerate(self.tonemapdata.keys()):
            if self.tonemapdata[s]["mouseinfo"] is not None:
                datatitle = self.tonemapdata[s]["mouseinfo"].ID
            else:
                datatitle = f"{str(self.datapath.parts[-1]):s}\n{s:s}"
            thr_spls = np.zeros(len(self.tonemaps[s]["Freqs"]))
            for k, fr in enumerate(self.tonemaps[s]["Freqs"]):  # next key is frequency
                if fr not in list(self.tonemapdata[s].keys()):
                    continue
                waves = self.tonemapdata[s][fr]["waves"]
                t = self.tonemapdata[s][fr]["timebase"]
                spls = self.tonemapdata[s][fr]["spls"]
                A.analyze(t, waves, dev=self.dev)
                print("spls: ", spls)
                print("wave shape: ", waves.shape)
                if waves.shape[0] <= 1:
                    continue

                thr_spl = A.thresholds(waves, spls, SD=3.5)
                thr_spls[k] = thr_spl
                # print(dir(axarr))
                plottarget = axarr[freqs.index(fr)]
                for j in range(len(spls))[::-1]:
                    if spls[j] == thr_spl:
                        plottarget.plot(
                            t,
                            0 * waves[j] * 1e6 + spls[j],
                            color=[0.5, 0.5, 0.5, 0.4],
                            linewidth=5,
                        )
                    plottarget.plot(t, 4 * waves[j] * 1e6 + spls[j], color=self.color_map[spls[j]])
                plottarget.set_xlim(0, 8.0)
                plottarget.set_ylim(10.0, 110.0)
                frtitle = "%.1f kHz" % (float(fr) / 1000.0)
                plottarget.title.set_text(frtitle)
                plottarget.title.set_size(9)
            self.thrs[s] = [self.tonemaps[s]["Freqs"], thr_spls]
        PH.cleanAxes(axarr)
        legend = f.legend(loc="upper left")
        if pdf is not None:
            pdf.savefig()
            mpl.close()

    def get_dataframe_clicks(self, allthrs):
        # use regex to parse information from the directory name
        re_control = re.compile(r"(?P<control>control)")
        re_noise_exposed = re.compile(r"(?P<noiseexp>NoiseExposed)")
        re_sham_exposed = re.compile(r"(?P<shamexp>ShamExposed)")
        re_un_exposed = re.compile(r"(?P<shamexp>Unexposed)")
        re_sex = re.compile(
            r"(?P<sex>\_[MF]+[\d]{0,3}\_)"
        )  # typically, "F" or "M1" , could be "F123"
        re_age = re.compile(r"(?P<age>_P[\d]{1,3})")
        re_date = re.compile("(?P<date>[\d]{2}-[\d]{2}-[\d]{4})")
        re_genotype_WT = re.compile(r"(?P<GT>_WT)")
        re_genotype_KO = re.compile(r"(?P<GT>_KO)")
        # put data in pd dataframe
        T = []
        # parse information about mouse, experiment, etc.
        for i, d in enumerate(allthrs):
            for m in allthrs[d]:
                name = str(Path(d).name)
                thr = allthrs[d][m]
                exp = re_control.search(name)
                if exp is not None:
                    exposure = "control"
                else:
                    exposure = "exposed"
                sham = re_sham_exposed.search(name)
                if sham is not None:
                    exposure = "sham"
                unexposed = re_un_exposed.search(name)
                if unexposed is not None:
                    exposure = "unexposed"
                exp = re_noise_exposed.search(name)
                if exp is not None:
                    exposure = "exposed"
                sex = re_sex.search(name)
                if sex is not None:
                    sex = sex.groups()[0][1]
                else:
                    sex = "U"
                age = re_age.search(name)
                if age is not None:
                    P_age = age.groups()[0][1:]
                    day_age = int(P_age[1:])
                else:
                    P_age = "U"
                    day_age = np.nan
                Genotype = "ND"  # not defined
                gtype = re_genotype_WT.search(name)
                if gtype is not None:
                    Genotype = gtype.groups()[0][1:]  # remove the underscore
                gtype = re_genotype_KO.search(name)
                if gtype is not None:
                    Genotype = gtype.groups()[0][1:]

                meas = [name, thr, exposure, sex, P_age, day_age, Genotype]
                T.append(meas)

        df = pd.DataFrame(
            T,
            columns=[
                "Subject",
                "threshold",
                "noise_exposure",
                "Sex",
                "P_age",
                "day_age",
                "genotype",
            ],
        )
        df = df.drop_duplicates(subset="Subject", keep="first")
        df.to_pickle("clicks_test.pkl")
        # df = pd.read_pickle('clicks_test.pkl')
        df.to_csv("clicks_test.csv")  # also update the csv for R statistics
        return df

    def get_dataframe_tones(self, allthrs):
        # use regex to parse information from the directory name
        re_control = re.compile(r"(?P<control>control)")
        re_noise_exposed = re.compile(r"(?P<noiseexp>NoiseExposed)")
        re_sham_exposed = re.compile(r"(?P<shamexp>ShamExposed)")
        re_un_exposed = re.compile(r"(?P<shamexp>Unexposed)")
        re_sex = re.compile(
            r"(?P<sex>\_[MF]+[\d]{0,3}\_)"
        )  # typically, "F" or "M1" , could be "F123"
        re_age = re.compile(r"(?P<age>_P[\d]{1,3})")
        re_date = re.compile("(?P<date>[\d]{2}-[\d]{2}-[\d]{4})")
        use_fr = [2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0]
        # put data in pd dataframe
        T = []
        # parse information about mouse, experiment, etc.
        for i, d in enumerate(allthrs):
            for m in allthrs[d]:
                for j in range(len(allthrs[d][m][0])):
                    name = str(Path(d).parts[-1])
                    fr = np.array(allthrs[d][m][0][j]) / 1000.0
                    if fr not in use_fr:
                        continue
                    fr_jit = fr + np.random.uniform(-fr / 8, fr / 8)
                    thr = allthrs[d][m][1][j]
                    exp = re_control.search(name)
                    if exp is not None:
                        exposure = "control"
                    else:
                        exposure = "exposed"
                    sham = re_sham_exposed.search(name)
                    if sham is not None:
                        exposure = "sham"
                    unexposed = re_un_exposed.search(name)
                    if unexposed is not None:
                        exposure = "unexposed"
                    exp = re_noise_exposed.search(name)
                    if exp is not None:
                        exposure = "exposed"
                    sex = re_sex.search(name)
                    if sex is not None:
                        sex = sex.groups()[0][1]
                    else:
                        sex = "U"
                    age = re_age.search(name)
                    if age is not None:
                        P_age = age.groups()[0][1:]
                        day_age = int(P_age[1:])
                    else:
                        P_age = "U"
                        day_age = np.nan

                    meas = [name, fr, fr_jit, thr, exposure, sex, P_age, day_age]
                    T.append(meas)

        df = pd.DataFrame(
            T,
            columns=[
                "Subject",
                "Freq",
                "Freq_jittered",
                "threshold",
                "noise_exposure",
                "Sex",
                "P_age",
                "day_age",
            ],
        )
        df.to_pickle("tones_test.pkl")
        return df

    def plotClickThresholds(self, allthrs, name, show_lines: bool = True, ax: object = None):
        """
        Make a plot of the click thresholds for all of the datasets.
        Data are plotted as scatter/box plots for each category

        Parameters
        ----------
        allthrs : dict
            A dictionary holding all the threshold information. The following
            structure is required:
                Keys: filenames for each dataset Values a dict of thresholds.
                The keys are the names of the click IO functions.

        Returns
        -------
        Nothing
        """

        PH.nice_plot(ax, position=-0.05, direction="outward", ticklength=4)
        n_datasets = len(list(allthrs.keys()))
        # print("# of Datasets found to measure tone thresholds: ", n_datasets)
        c_map = ABRF.makeColorMap(n_datasets, list(allthrs.keys()))

        df = self.get_dataframe_clicks(allthrs)

        # sns.stripplot(x="genotype", y="threshold", data=df, hue="Sex", kind="violin", ax=ax)

        sns.boxplot(
            x="genotype",
            y="threshold",
            hue=S,
            data=df,
            whis=2.0,
            width=0.5,
            dodge=True,
            fliersize=0.15,
            order=["WT", "KO"],
            color=[0.85, 0.85, 0.85, 0.25],
            ax=ax,
        )
        sns.swarmplot(
            x="genotype",
            y="threshold",
            hue="Sex",
            data=df,
            order=["WT", "KO"],
            ax=ax,
            dodge=True,
        )
        ax.set_ylim(20, 120)
        ax.set_ylabel("Threshold (dBSPL)", fontsize=11)
        ax.set_xlabel("Genotype", fontsize=11)
        yvals = [20, 30, 40, 50, 60, 70, 80, 90, 100, "NR"]
        yt = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
        mpl.yticks(yt, [str(x) for x in yvals])

        thrs_sorted = None
        frmean = None
        frstd = None
        return (thrs_sorted, frmean, frstd)

    def plotToneThresholds(self, allthrs, name, show_lines: bool = True):
        """
        Make a plot of the tone thresholds for all of the datasets Data are
        plotted against a log frequency scale (2-64kHz) Data is plotted into the
        current figure.

        Parameters
        ----------
        allthrs : dict
            A dictionary holding all the threshold information. The following
            structure is required:
                Keys: filenames for each dataset Values a dict of thresholds.
                The keys are the names of the tone maps (because more than one
                tone map may be combined) The values are tuples of (frequency,
                threshold)

        Returns
        -------
        Nothing
        """

        fig = mpl.figure(figsize=(7, 5))
        # The first items are for padding and the second items are for the axes.
        # sizes are in inch.
        h = [Size.Fixed(0.8), Size.Fixed(4.0)]
        v = [Size.Fixed(0.8), Size.Fixed(4.0)]
        divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
        # The width and height of the rectangle are ignored.
        ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1))

        fig.suptitle(self.datasetname)
        PH.nice_plot(ax, position=-0.05, direction="outward", ticklength=4)
        n_datasets = len(list(allthrs.keys()))
        print("# of Datasets found to measure tone thresholds: ", n_datasets)
        c_map = ABRF.makeColorMap(n_datasets, list(allthrs.keys()))

        df = self.get_dataframe_tones(allthrs)

        sns.lineplot(
            x="Freq",
            y="threshold",
            hue="noise_exposure",
            data=df,
            ax=ax,
            err_style="band",
            ci=68,
        )
        axs = sns.scatterplot(
            x="Freq_jittered",
            y="threshold",
            hue="Subject",
            data=df,
            alpha=0.65,
            ax=ax,
            s=10,
        )
        axs.set_xscale("log", nonpositive="clip", base=2)
        axs.set_xlim(1, 65)
        axs.set_ylim(20, 100)

        xt = [2.0, 4.0, 8.0, 16.0, 32.0, 48.0, 64.0]
        mpl.xticks(xt, [str(x) for x in xt])
        legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize=7)
        thrs_sorted = None
        frmean = None
        frstd = None
        return (thrs_sorted, frmean, frstd)

    def get_combineddata(
        self, datasetname, dataset, freq=None, lineterm="\r"
    ) -> Tuple[np.array, np.array]:
        """
        Read the data sets and combine the p (condensation) and n
        (rarefaction) data sets for alternating polarity stimuli.

        Parameters
        ----------
        datasetname : str
            yyyymmdddd-time format for start of dataset name
        dataset : dict
            dictionary for the dataset that identifies the stimulus type, the
            SPLs in the dataset, and the frequency if the dataset is a tone pip
            run
        freq : float (default: None)
            for tone maps, the specific frequency intensity series to return

        lineterm: str (default: '\r')
            line terminator for this dataset

        Returns
        -------
        waves : numpy array
            Waveforms, as a nxm array, where n is the number of intensities, and
            m is the length of each waveform
        tb: numpy array for the time base

        """
        if dataset["stimtype"] == "click":
            fnamepos = datasetname + "-p.txt"
            fnameneg = datasetname + "-n.txt"
            try:
                waves, tb = self.read_dataset(self.datapath, "click", fnamepos, fnameneg, lineterm)
            except:
                print("Failed on datasetname: ", datasetname)
                print(" with dataset: ", dataset)
                raise ValueError()
            return waves, tb
        if dataset["stimtype"] == "tonepip":
            fnamepos = datasetname + "-p-%.3f.txt" % freq
            fnameneg = datasetname + "-n-%.3f.txt" % freq
            waves, tb = self.read_dataset("tonepip", fnamepos, fnameneg, lineterm)
            return waves, tb

    def read_dataset(
        self,
        datapath: Union[Path, str],
        datatype: str = "click",
        fnamepos: str = "",
        fnameneg: str = "",
        lineterm="\r",
    ):
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
        timebase

        """
        # handle missing files.
        if not Path(datapath, fnamepos).is_file():
            return None, None
        if not Path(datapath, fnameneg).is_file():
            return None, None
        print("Reading from: ", str(Path(datapath, fnamepos)))
        if datatype == "click":
            spllist = self.clickmaps[fnamepos[:13]]["SPLs"]
        else:
            spllist = self.tonemaps[fnamepos[:13]]["SPLs"]
        cnames = [f"{spl:.1f}" for i, spl in enumerate(spllist)]
        posf = pd.io.parsers.read_csv(
            Path(datapath, fnamepos),
            sep=r"[\t ]+",
            lineterminator=r"[\r\n]+",  # lineterm,
            skip_blank_lines=True,
            header=None,
            names=cnames,
            engine="python",
        )
        negf = pd.io.parsers.read_csv(
            Path(datapath, fnameneg),
            sep=r"[\t ]+",
            lineterminator=r"[\r\n]+",
            skip_blank_lines=True,
            header=None,
            names=cnames,
            engine="python",
        )
        npoints = len(posf[cnames[0]])
        tb = np.linspace(0, npoints * self.sample_rate * 1000.0, npoints)
        if np.max(tb) > 25.0:
            u = np.where(tb < 25.0)
            tb = tb[u]
        npoints = tb.shape[0]

        waves = np.zeros((len(posf.columns), npoints))

        for i, cn in enumerate(posf.columns):
            waves[i, :] = (posf[cn][:npoints] + negf[cn][:npoints]) / 2.0

        for i in range(waves.shape[0]):
            # waves[i, -1] = waves[i, -2]  # remove nan from end of waveform...
            waves[i, :] = ABRF.filter(
                waves[i, :], 4, self.lpf, self.hpf, samplefreq=self.sample_freq
            )
            if self.invert:
                waves[i, :] = -waves[i, :]

        return waves, tb


###
###======================================================================================
###


def build_click_plot(nplots=1):
    m, n = PH.getLayoutDimensions(nplots)
    # print("Grid: ", m, n)
    # print("dirs: ", dirs)

    if m > 1:
        h = 2.5 * m
        if h > 10.5:
            h = 10.5
    else:
        h = 3
    horizontalspacing = 0.08
    if n > 5:
        horizontalspacing = 0.08 / (n / 5.0)
    print("Plot array and count: ", m, n, nplots)

    # generate plot grid for waveforms
    Plot_f = PH.regular_grid(
        m,
        n,
        order="rowsfirst",
        figsize=(11, h),
        verticalspacing=0.04,
        horizontalspacing=0.04,
    )
    if Plot_f.axarr.ndim > 1:
        axarr = Plot_f.axarr.ravel()
    else:
        axarr = Plot_f.axarr

    for ax in Plot_f.axarr:
        PH.nice_plot(ax, position=-0.03, direction="outward", ticklength=3)

    # generate plot grid for individual IO functions
    Plot_f2 = PH.regular_grid(
        m,
        n,
        order="rowsfirst",
        figsize=(11, h),
        verticalspacing=0.04,
        horizontalspacing=0.04,
    )
    for ax in Plot_f2.axarr:
        PH.nice_plot(ax, position=-0.03, direction="outward", ticklength=3)
    if Plot_f2.axarr.ndim > 1:
        axarr2 = Plot_f2.axarr.ravel()
    else:
        axarr2 = Plot_f2.axarr

    # generate plot space for click IO overlay and threshold summary on right
    Plot_f4 = PH.regular_grid(
        1,
        2,
        order="rowsfirst",
        figsize=(8, 5),
        verticalspacing=0.07,
        horizontalspacing=0.125,
        margins={
            "bottommargin": 0.15,
            "leftmargin": 0.1,
            "rightmargin": 0.05,
            "topmargin": 0.08,
        },
        num="Click IO Overlay",
    )

    IOax = Plot_f4.axarr.ravel()
    for ax in Plot_f4.axarr:
        PH.nice_plot(ax, position=-0.03, direction="outward", ticklength=3)
    PlotInfo = plotinfo(
        m=m,
        n=n,
        icol=0,
        Plot_f=Plot_f,
        Plot_f2=Plot_f2,
        Plot_f4=Plot_f4,
        IOax=IOax,
        axarr=axarr,
        axarr2=axarr2,
    )
    return PlotInfo


def populate_plot(P, select, datadir, plot_info: object, plot_index: int, icol: int):
    xlab = False
    ylab = False
    if plot_info.nrows == plot_info.m - 1:
        xlab = True
    if plot_info.ncols == 0:
        ylab = True
    IOdata = P.plotClicks(
        select=select,
        datadir=datadir,
        plottarget=plot_info.axarr[plot_index],
        superIOPlot=plot_info.IOax[0],
        IOplot=plot_info.axarr2[plot_index],
        colorindex=icol,
        show_x_label=xlab,
        show_y_label=ylab,
    )

    return IOdata


def do_clicks(
    dsname: str,
    top_directory: Union[str, Path],
    dirs: list,
    ABR_Datasets: object = None,
    plot_info: object = None,
    nplots: int = 1,
    plot_index: int = 0,
):
    """analyze the click data

    Parameters
    ----------
    dsname : str
        The dataset name, from the info dict ('dir') in ABR_Datasets
        This is the name of the directory that holds the session ABR data
    top_directory : str
        The full path to the dsname directory.
    dirs : list
        A list of the subdirectories under the top directory.
        This routine will look in each subdirectory for
        click data.
    """

    clicksel = ABR_Datasets[dsname].clickselect
    if len(dirs) == 0:
        CP.cprint("r", f"No Directories found: {str(dsname):s}, {str(top_directory):s}")
        return plot_info, None
    if plot_info is None:
        plot_info = build_click_plot(nplots)
    # do analysis and plot, and accumulate thresholds while doing so.
    nsel = len(clicksel)
    allthrs = {}
    IO_DF = []

    print("do_clicks: ", dirs)
    for icol, k in enumerate(range(len(dirs))):  # list_order):
        P = ABR(
            Path(top_directory),
            "clicks",
            info=ABR_Datasets[dsname],
            datasetname=dsname,
            subject=ABR_Datasets[dsname].subject,
        )
        CP.cprint(
            "r",
            P.df_excel[P.df_excel.Subject == ABR_Datasets[dsname].subject].Sex.values,
        )

        if icol == 0:
            P.summaryClick_color_map = ABRF.makeColorMap(nsel, list(range(nsel)))
        print("doClicks: Getting Click data with : ", clicksel[k], dirs[k], dsname)
        P.getClickData(select=clicksel[k], directory=dirs[k])
        IOdata = populate_plot(
            P,
            clicksel[k],
            datadir=dsname,
            plot_info=plot_info,
            plot_index=plot_index,
            icol=icol,
        )
        print("Threshold: ", P.thrs)
        dirname = str(Path(dirs[k]).name)
        allthrs[dirname] = P.thrs
        IO_DF.extend(IOdata)
        plot_info.icol = icol

    clickIODF = pd.DataFrame(IO_DF, columns=["Subject", "run", "spl", "ppio", "thrs", "Group"])
    clickIODF["group_cat"] = clickIODF["Group"].astype("category")
    fill_circ = MarkerStyle("o", fillstyle="full")
    fill_square = MarkerStyle("s", fillstyle="full")
    if plot_info.IOax[0] is not None:
        sns.lineplot(
            x="spl",
            y="ppio",
            hue="group_cat",  # style="group_cat",
            data=clickIODF,
            ax=plot_info.IOax[0],
            hue_order=["WT", "KO"],
            markers=False,  # [fill_circ, fill_square],
            err_style="band",
            errorbar="sd",
            err_kws={"alpha": 0.8, "linewidth": 0.75},
            mew=1.0,
            linewidth=1.5,
            markersize=1,
        )
    clickIODF.to_csv("ClickIO.csv")

    spls = set(clickIODF["spl"])
    clickIOWTlist = []
    clickIOKOlist = []
    # subjs = set(clickIODF['subject'])
    # print(clickIODF.head())
    # for icol in subjs:
    #     if clickIODF.at['group'] == 'WT':
    #         clickIOWT['subject']

    plot_info.IOax[0].legend(loc="upper left", fontsize=7)
    population_thrdata = P.plotClickThresholds(
        allthrs, name="Click Thresholds", ax=plot_info.IOax[1]
    )
    return plot_info, clickIODF


def do_tones(dsname: str, top_directory: Union[str, Path], dirs: list):
    """analyze the tone data

    Parameters
    ----------
    dsname : str
        The dataset name, from the info dict ('dir') in ABR_Datasets
        This is the name of the directory that holds the session ABR data
    top_directory : str
        The full path to the dsname directory.
    dirs : list
        A list of the subdirectories under the top directory.
        This routine will look in each subdirectory for
        click data.
    """
    if "toneselect" in list(ABR_Datasets[dsname].keys()):
        tonesel = ABR_Datasets[dsname]["toneselect"]
    else:
        tonesel = [None] * len(dirs)

    fofilename = Path(top_directory, "ToneSummary.pdf")
    allthrs = {}
    with PdfPages(fofilename) as pdf:
        for k, v in enumerate(tonesel):
            P = ABR(Path(top_directory, dirs[k]), "tones", datasetname=dsname)
            P.getToneData(select=tonesel[k], directory=dirs[k])
            P.plotTones(select=tonesel[k], pdf=pdf)
            allthrs[dirs[k]] = P.thrs
    population_thrdata = P.plotToneThresholds(allthrs, name="Tone Thresholds")
    print("pop thr data: ", population_thrdata)
    print("Hz\tmean\tstd\t individual")
    if population_thrdata[0] is not None:
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


# read the Coding file, which contains information about the ABR dates
def get_coding_data(experiment):
    print("experiment keys: ", list(experiment.keys()))
    assert "databasepath" in experiment.keys()
    print(experiment["databasepath"])
    coding_file = Path(
        experiment["databasepath"], experiment["directory"], experiment["coding_file"]
    )
    assert coding_file.is_file()
    coding = pd.read_excel(coding_file, sheet_name=experiment["coding_sheet"])
    coding.rename(columns={"Animal_ID": "Subject"}, inplace=True)
    print(coding.columns)
    return coding


def analyze_from_ABR_Datasets_main():
    if len(sys.argv) > 1:
        dsname = sys.argv[1]
        mode = sys.argv[2]
    else:
        print("Missing command arguments; call: plotABRs.py datasetname [click, tone]")
        exit(1)

    if dsname not in list(ABR_Datasets.keys()):
        print("These are the known datasets: ")
        print("   ", list(ABR_Datasets.keys()))
        raise ValueError("The selected dataset %s not found in our list of known datasets")
    if mode not in ["tones", "clicks"]:
        raise ValueError("Second argument must be tones or clicks")

    print("Selected ABR Dataset: ", ABR_Datasets[dsname])
    top_directory = Path(basedir, ABR_Datasets[dsname].directory)
    print("\nTop_directory", top_directory)
    # print(list(Path(top_directory).glob("*")))

    dirs = [
        tdir
        for tdir in Path(top_directory).glob("*")
        if Path(top_directory, tdir).is_dir()
        and not str(tdir.name).startswith("Recordings_cant_be_used")  # dirs to exclude
        and not str(tdir.name).startswith("Unsucessful recordings")
        and not str(tdir.name).find("Unsure") >= 0
    ]

    if mode == "clicks":
        plot_info, IO_DF = do_clicks(dsname, top_directory, dirs, ABR_Datasets)

    elif mode == "tones":
        do_tones(dsname, top_directory, dirs, ABR_Datasets)
    else:
        raise ValueError(f"Mode is not known: {mode:s}")


def get_dirs(row, datatype="click", plot_info=None, nplots: int = 1, plot_index: int = 0):
    """get the data set directories for this row that have the data type

    Args:
        row (_type_): _description_
        datatype (str, optional): _description_. Defaults to "click".

    Returns:
        _type_: _description_
    """
    assert datatype in ["click", "tones"]
    if pd.isnull(row.Runs):
        return [], plot_info, None
    d = row.Runs.strip().split(",")
    dirs = []
    clickdirs = []
    tonedirs = []
    for r in d:
        rs = r.split(":")
        if rs[0].strip() == datatype:
            dirs.append(rs[1])
    if datatype == "click":
        clickdirs = dirs
    if datatype == "tones":
        tonedirs = dirs

    ABR_Datasets = {
        row.Group: ABR_Data(
            directory=Path(row.BasePath, row.Group),
            subject=row.Subject,
            invert=True,
            clickselect=clickdirs,
            toneselect=tonedirs,
            term="\r",
            minlat=2.2,
        ),
    }
    plot_info, IO_DF = do_clicks(
        row.Group,
        Path(row.BasePath, row.Group, row.Subject),
        dirs,
        ABR_Datasets,
        plot_info=plot_info,
        nplots=nplots,
        plot_index=plot_index,
    )
    return dirs, plot_info, IO_DF


class Picker(QObject):
    def __init__(self, space=None, data=None, axis=None):
        assert space in [None, 2, 3]
        self.space = space  # dimensions of plot (2 or 3)
        self.setData(data, axis)
        self.annotateLabel = None

    def setData(self, data, axis=None):
        self.data = data
        self.axis = axis

    def setAction(self, action):
        # action is a subroutine that should be called when the
        # action will be called as self.action(closestIndex)
        self.action = action

    def pickEvent(self, event, ax):
        """Event that is triggered when mouse is clicked."""
        # print("event index: ", event.ind)
        # print(dir(event.mouseevent))
        # print(event.mouseevent.inaxes == ax)
        # print(ax == self.axis)
        # print("psi.Picker pickEvent: ", self.data.iloc[event.ind])  # find the matching data.
        return


def pick_handler(event, picker_func):
    print("\nDataframe indices: ", event.ind)
    # print(len(event.ind))

    print("   # points in plot: ", len(picker_func.data))
    print(picker_func.data.index)  # index into the dataframe
    for i in event.ind:
        day = picker_func.data.iloc[i]
        print("index: ", i)
        print(day)

        return None


def analyze_from_excel(datasets=["Tessa_NF107"], agerange=(20, 150)):
    df = pd.read_excel("ABRS.xlsx", sheet_name="Sheet1")
    df.rename(columns={"Group": "dataset"}, inplace=True)
    df["Group"] = ""  # nothing here...
    for dataset in datasets:  # for each of the datasets, analyze
        dfion = []
        plot_info = None
        dfn = df[df.dataset == dataset]  # subset into the full dataset
        dfn = dfn.sort_values(["Age", "Sex"])
        dfn = dfn[(df.Age >= agerange[0]) & (df.Age <= agerange[1])]
        dfn = dfn.reset_index()  # do this LAST

        nplots = int(np.max(dfn.index))
        CP.cprint("c", f"Nplots: {nplots:d}")
        for plot_index in range(nplots):
            dirs, plot_info, IO_DF = get_dirs(
                row=dfn.iloc[plot_index],
                datatype="click",
                plot_info=plot_info,
                nplots=nplots,
                plot_index=plot_index,
            )
            # add data to df_io
            if IO_DF is None:
                continue
            spls = IO_DF.spl.values.tolist()
            ppio = IO_DF.ppio.values.tolist()
            thr = float(list(set(IO_DF.thrs))[0])
            d = dfn.iloc[plot_index]

            dfion.append(
                {
                    "dataset": dataset,
                    "Group": "",  # this has to be translated later...
                    "Date": d.Date,
                    "DataDirectory": d.DataDirectory,
                    "Genotype": d.Genotype,
                    "Strain": d.Strain,
                    "Cross": d.Cross,
                    "Treatment": d.Treatment,
                    "Subject": d.Subject,
                    "Age": d.Age,
                    "Sex": d.Sex,
                    "spls": spls,
                    "ppio": ppio,
                    "threshold": thr,
                }
            )

        # x = dfn.apply(get_dirs,  args=("click", plot_info), axis=1)  # all of the data in a given dataset
        # dfn.apply(get_dirs,  args=("tones",), axis=1)
        top_directory = Path("/Volumes/Pegasus_002/ManisLab_Data3/abr_data/", group)
        df_io = pd.DataFrame(dfion)

        df_io.to_excel(Path(top_directory, f"ClickIO_{group:s}.xlsx"))
        if plot_info is not None:
            top_directory = Path("/Volumes/Pegasus_002/ManisLab_Data3/abr_data/", group)
            mpl.figure(plot_info.Plot_f.figure_handle)
            fofilename = Path(top_directory, "ClickWaveSummary.pdf")
            mpl.savefig(fofilename)

            mpl.figure(plot_info.Plot_f2.figure_handle)
            fo2filename = Path(top_directory, "ClickIOSummary.pdf")
            mpl.savefig(fo2filename)

            mpl.figure("Click IO Overlay")
            fo4filename = Path(top_directory, "ClickIOOverlay.pdf")
            mpl.savefig(fo4filename)
    mpl.show()


def _average(row):
    """Average repeated measures in an IO function:
    If there are multiple recordings at one SPL (for clicks), those are
    averaged together.

    This would be called using pandas dataframe .apply

    Args:
        row (Pandas series): a data row

    Returns:
        Pandas series: data row
    """
    spls = ast.literal_eval(row.spls)
    ppio = ast.literal_eval(row.ppio)
    spl_new = np.unique(spls)
    # Use bincount to get the accumulated summation for each unique x, and
    # divide each summation by the respective count of each unique value in x
    ppio_mean = np.bincount(spls, weights=ppio) / np.bincount(spls)
    ppio_mean = [p for p in ppio_mean if not pd.isnull(p)]  # bincount works on bin width of 1...
    row.spls = spl_new.tolist()
    row.ppio = ppio_mean
    return row


def _row_convert(row):
    # print(row)
    row.spls = ast.literal_eval(row.spls)
    row.ppio = ast.literal_eval(row.ppio)
    return row


def reorganize_abr_data(
    datasets: list,
    groups: list,
    abr_data: pd.DataFrame,
    coding_data: Union[pd.DataFrame, None] = None,
):
    """Reorganize datasets in abrdata into long form
    This also matches animals and tags each animal by the group and dataset it belongs to
    using the animal_id, group and datasets columns.

    """
    # print(abrdata.columns)
    abr_df = pd.DataFrame(["group", "Subject", "Sex", "spls", "ppio", "dataset"])
    abr_dict: dict = {
        "group": [],
        "Subject": [],
        "ExpSPL": [],
        "Sex": [],
        "spls": [],
        "ppio": [],
        "Dataset": [],
    }
    # print("abrdata head: ", abrdata.head())
    # print("abrdata unique subjects: ", abr_data.Subject.unique())
    # print("Unique datasets in abrdata: ", abr_data["Dataset"].unique())
    coding_data = coding_data[coding_data["Group"].isin(groups)]
    for d in datasets:  # top level - data may be derived from multiple datasets
        print("Working with dataset: ", d)
        df_ds = abr_data[abr_data["Dataset"] == d].reset_index()
        unique_subjects = df_ds.Subject.unique()
        # print("    Unique subjects in abr_data: ", unique_subjects)
        if coding_data is not None:
            print("    Unique subjects in coding data: ", coding_data.Subject.unique())
        for subject in unique_subjects:
            if pd.isnull(subject) and coding_data is not None:
                continue  # no subject id, skip if we don't have a coding file
            subject_coding = coding_data[coding_data["Subject"] == subject]
            if subject_coding.empty:
                CP.cprint("r", f"Subject {subject:s} not found in coding data file")
                continue
            print("Adding Subject: ", subject, "Group: ", subject_coding["Group"].values[0])
            group = subject_coding["Group"].values[0]  # get the group this subject belongs to.
            ExpSPL = subject_coding["SPL"].values[0]
            df_g = abr_data[abr_data["Subject"] == subject].reset_index()
            for iloc in df_g.index:
                # print("iloc: ", iloc)
                spls = df_g.iloc[iloc].spls
                if not isinstance(spls, list):
                    spls = ast.literal_eval(spls)
                ppio = df_g.iloc[iloc].ppio
                if not isinstance(ppio, list):
                    ppio = ast.literal_eval(ppio)
                for i, spl in enumerate(spls):
                    abr_dict["spls"].append(spls[i])
                    abr_dict["ppio"].append(ppio[i])
                    abr_dict["group"].append(group)
                    abr_dict["ExpSPL"].append(ExpSPL)
                    abr_dict["Subject"].append(df_g.iloc[iloc].Subject)
                    abr_dict["Sex"].append(df_g.iloc[iloc].Sex)
                    abr_dict["Dataset"].append(df_g.iloc[iloc].Dataset)
    # print("Subject column: ", abr_dict["Subject"])
    abr_df = pd.DataFrame(abr_dict)
    print("# of entries in the abr dataframe: ", len(abr_df))
    # print("abr_df columns: ", abr_df.columns)
    # print(abr_df.head())
    return abr_df


def _compute_threshold(row, baseline):
    """_compute_threshold Compute a threshold by smoothing the curve by fitting
    to a Hill function, then get the threshold crossing to the nearest dB.

    Parameters
    ----------
    row : _type_
        _description_
    baseline : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    y = np.array(row["ppio"])
    x = np.array(row["spls"])
    # Find where the IO function exceeds the baseline threshold
    # print(grand_df.iloc[i]['ppio'])
    # yd[xn] = p[1] / (1.0 + (p[2]/x[xn])**p[3])
    from lmfit import Model

    def hill(x, vmax, v50, n):
        return vmax / (1.0 + (v50 / x) ** n)

    Hillmodel = Model(hill)
    Hillmodel.set_param_hint("vmax", min=0.0, max=20.0)
    Hillmodel.set_param_hint("v50", min=0.0, max=100.0)
    Hillmodel.set_param_hint("n", min=1.0, max=5.0)
    params = Hillmodel.make_params(vmax=5, v50=50.0, n=2)

    result = Hillmodel.fit(y, params, x=x)
    yfit = result.best_fit
    xs = np.linspace(0, 90, 91)
    # print("yfit: ", yfit)
    ys = Hillmodel.eval(x=xs, params=result.params)

    # ithr = np.argwhere(np.array(y) > baseline)
    ithr = np.argwhere(np.array(ys) > baseline)
    # print(len(ithr), np.max(y), baseline)
    if len(ithr) == 0:
        ithr = len(y) - 1
        # print("No threshold found: ", row.Subject)
        interp_thr = 100.0
    else:
        # ithr = ithr[0][0]
        # m = (y[ithr] - y[ithr - 1]) / (x[ithr] - x[ithr - 1])
        # b = y[ithr] - m * x[ithr]
        # bin = 1.0
        # int_thr = (baseline - b) / m
        # interp_thr = np.round(int_thr / bin, 0) * bin
        # print("interp thr: ", row.Subject, int_thr, interp_thr)
        interp_thr = xs[ithr][0][0]
    if interp_thr > 90.0:
        interp_thr = 100.0
    # print("interpthr: ", interp_thr)
    row["interpolated_threshold"] = interp_thr
    row["maxabr"] = np.max(y)
    return row


def compute_io_stats(ppfd: object, Groups: list):
    from statsmodels.stats.anova import AnovaRM

    return
    model = ols(f"interpolated_threshold ~ C(Group)+Subject", ppfd).fit()
    table = sm.stats.anova_lm(model, typ=2)  # Type 2 ANOVA DataFrame
    p = "=" * 80 + "\n"
    p += str(table) + "\n"
    p += str(model.nobs) + "\n"
    p += str(model.summary())
    print(p)


def compute_other_stats(grand_df, groups=["Tessa_FVB", "Tessa_CBA", "Tessa_BNE"]):
    from scipy import stats

    print("groups: ", groups)
    print("Groups in the grand dataframe: ", grand_df.Group.unique())
    data = {}
    for group in groups:
        data[group] = grand_df[grand_df.Group == group]

        U1 = stats.kruskal(*[data[dg].interpolated_threshold for dg in groups])

        # post_U1 = scikit_posthocs.posthoc_dunn([fvb.interpolated_threshold,
        #                                     cba.interpolated_threshold,
        #                                     bne.interpolated_threshold],
        #                                 p_adjust='bonferroni')
        post_U1 = sckikit_posthocs.posthoc_dunn([data[dg].interpolated_threshold for dg in groups])
    print("Kruskall-Wallis on Thresholds")
    print(U1)
    print("Post test Dunns with Bonferroni correction: ")
    print(post_U1)
    U2 = stats.kruskal(*[data[dg].maxabr for dg in groups])
    print("\nKruskall-Wallis on maximum ABR amplitude")
    print(U2)
    post_U2 = scikit_posthocs.posthoc_dunn(
        [data[dg].maxabr for dg in groups], p_adjust="bonferroni"
    )
    print("Post test Dunns with Bonferroni correction: ")
    print(post_U2)


def set_groups(grand_df, coding_data: Union[pd.DataFrame, None] = None):
    if coding_data is None:
        grand_df["group"] = "Control"  # set ALL to control
        return grand_df
    grand_df["group"] = ""
    for i, row in grand_df.iterrows():
        subject = row["Subject"]
        # print("subject: ", subject)
        if pd.isnull(subject) or subject == "nan" or subject == "":
            grand_df.at[i, "group"] = "What?"
            continue
        group = coding_data[coding_data["Subject"] == subject]["Group"]
        ExpSPL = coding_data[coding_data["Subject"] == subject]["SPL"]
        if not pd.isnull(group).all():
            grand_df.at[i, "group"] = group.values[0]
            grand_df.at[i, "ExpSPL"] = ExpSPL.values[0]
    return grand_df

def relabel_xaxes(axp, experiment, angle=None):
    if "new_xlabels" in experiment.keys():
        xlabels = axp.get_xticklabels()
        for i, label in enumerate(xlabels):
            labeltext = label.get_text()
            if labeltext in experiment["new_xlabels"]:
                xlabels[i] = experiment["new_xlabels"][labeltext]   
        axp.set_xticklabels(xlabels)  # we just replace them... 
    else:
        pass
        #  print("no new x labels available")
    if angle is not None:
        axp.set_xticks(axp.get_xticks(), axp.get_xticklabels(), rotation=45, ha="right")

def relabel_yaxes_threshold(axp):
    ticksy = axp.get_yticklabels()
    ticksy[-1].set_text(f"$>$90")
    axp.set_yticklabels(ticksy)

def place_legend(P, x, y, spacing=0.05, experiment=None):
    legend_text = {}
    lmap = experiment["group_legend_map"]
    cols = experiment['plot_colors']
    for l in lmap.keys():
        legend_text[lmap[l]] = cols[l]

    for i, txt in enumerate(legend_text.keys()):
        mpl.text(
            x=x,
            y=y - i * spacing,
            s=txt,
            color=legend_text[txt],  # ns.color_palette()[i],
            fontsize="medium",
            transform=P.figure_handle.transFigure,
        )

        

def plot_io_from_excel(
    datasets: list,
    groups: Union[list, None] = None,  # list of groups to analyze/plot
    agerange: Union[list, tuple] = (30, 105),
    coding: object = None,  # coding dataframe
    picking: bool = False,
    thrplot=False,
):
    df0 = pd.read_excel("ABRS.xlsx", sheet_name="Sheet1")  # get main database

    print("datasets: ", datasets)
    print("Groups on call to plot_io_from_excel: ", groups)
    ncols = 3
    if thrplot:
        ncols = 4
    top_path = df0.iloc[0].BasePath
    PL = PH.regular_grid(
        rows=1,
        cols=ncols,
        figsize=(10, 4),
        horizontalspacing=0.1,
        panel_labels=["A", "B", "C", "D"],
        margins={"topmargin": 0.12, "bottommargin": 0.18, "leftmargin": 0.08, "rightmargin": 0.10},
        labelposition=(-0.08, 1.05),
    )
    # f, ax = mpl.subplots(1, 3, figsize=(9, 5))
    ax = PL.axarr.ravel()

    colors = ["k", "r", "b", "c"]
    markers = ["s", "o", "^", "D"]
    mfill = {"M": "full", "F": None}

    grand_df = pd.DataFrame()  # make a new data frame with all datasets compiled into one
    for k, dataset in enumerate(datasets):
        excelfile = Path(top_path, dataset, f"ClickIO_{dataset:s}.xlsx")
        df = pd.read_excel(Path(top_path, dataset, f"ClickIO_{dataset:s}.xlsx"))
        df.rename(columns={"Animal_identifier": "Subject"}, inplace=True)
        # some data selection - by age, some data from a group, treatment
        df = df[(df.Age >= agerange[0]) & (df.Age <= agerange[1])]
        df = df.loc[df.Treatment.isin([None, "UnExposed", np.nan])].reset_index()
        df = df.apply(_average, axis=1)
        grand_df = pd.concat((grand_df, df))
    p_df = reorganize_abr_data(
        datasets=datasets, groups=groups, abr_data=grand_df, coding_data=coding
    )
    grand_df = set_groups(grand_df, coding_data=coding)
    order = Experiment["plot_order"]["Group"]  # sort by group type
    p_df.to_csv("reorganized_data.csv")  # just so we can look at it separately
    p_df = p_df[p_df.group.isin(groups)]
    print("p_df: ", p_df.group.unique())
    print("order: ", order)
    sns.lineplot(
        data=p_df,
        x="spls",
        y="ppio",
        hue="group",
        hue_order=order,
        estimator=np.mean,
        errorbar="sd",
        ax=ax[0],
        linewidth=2.0,
        # palette=Experiment['plot_colors'],
    )
    sns.lineplot(
        data=p_df,
        x="spls",
        y="ppio",
        hue="group",
        hue_order=order,
        units="Subject",
        estimator=None,
        ax=ax[0],
        linewidth=0.3,
        alpha=0.5,
        # palette=Experiment['plot_colors'],
    )
    # ax.plot(spls, ppio, color=colors[k], marker=markers[k], fillstyle=mfill[sex], linestyle = '-')
    df = grand_df
    PH.nice_plot(ax[0], position=-0.03, direction="outward", ticklength=3)
    PH.referenceline(ax[0], 0.0)
    PH.set_axes_ticks(ax=ax[0], xticks=[20, 30, 40, 50, 60, 70, 80, 90])
    ax[0].set_ylabel(f"N1-P1 ($\mu V$)")
    ax[0].set_xlabel("Click (dB SPL)")
    ax[0].set_xlim(20.0, 100.0)
    ax[0].set_ylim(-0.5, 8)

    labels = {}
    d_gs = {}
    n_gs = {}

    for group in groups:
        d_gs[group] = p_df[p_df.group == group]
        n_gs[group] = len(set(d_gs[group].Subject.values))
        labels[group] = f"{group:s} (N={n_gs[group]:d})"

    # print(colors)
    # custom_legend = []
    # for i, l in enumerate(groups):
    #     custom_legend.append(Line2D([0], [0], marker=None, color=colors[i], lw=2, label=labels[l]))

    # ax[0].legend(
    #     handles=custom_legend,
    #     handlelength=1,
    #     loc="upper left",
    #     fontsize=8,
    #     labelspacing=0.33,
    #     markerscale=0.5,
    # )

    # this plots thresholds from the initial analysis. Those thresholds can be incorrect.
    # sns.boxplot(data=grand_df, x="dataset", y="threshold", palette="colorblind", ax=ax[1])

    # here we compute the SD of *all* curves for levels of 20 and 25dB SPL

    p_thr = p_df[p_df.spls <= 25.0]
    # print("   pthr: ", p_thr)
    baseline = 3.5 * np.std(p_thr["ppio"].values)
    # print("    baseline: ", baseline)
    grand_df["interpolated_threshold"] = np.nan
    grand_df["maxabr"] = np.nan
    grand_df = grand_df.apply(_compute_threshold, baseline=baseline, axis=1)
    # print("Grand df columns: ", grand_df.columns)
    # print("Grand df head: ", grand_df.head())
    grand_df = grand_df[grand_df.group.isin(groups)]
    print("hue order threshold: ", order)
    if not picking:
        sns.swarmplot(
            data=grand_df,
            x="group",
            y="interpolated_threshold",
            hue="group",
            hue_order=order,
            order=order,
            ax=ax[1],
            legend="auto",  # palette=Experiment['plot_colors'],
            picker=False,
            zorder=100,
            clip_on=False,
        )
        sns.boxplot(
            data=grand_df,
            x="group",
            y="interpolated_threshold",
            hue="group",
            hue_order=order,
            order=order,
            saturation=0.25,
            orient="v",
            showfliers=False,
            linewidth=0.5,
            zorder=50,
            ax=ax[1],
        )
    else:
        sns.scatterplot(
            data=grand_df,
            x="group",
            y="interpolated_threshold",
            hue="group",
            hue_order=order,
            order=order,
            ax=ax[1],
            legend=False,  # palette=Experiment['plot_colors'],
            picker=True,
            clip_on=False,
        )

    picker_func1 = Picker(space=2, data=grand_df.copy(deep=True), axis=ax[1])

    PL.figure_handle.canvas.mpl_connect(
        "pick_event", lambda event: pick_handler(event, picker_func1)
    )
    ax[1].set_ylim(0, 100)
    ax[1].set_ylabel("Threshold (dB SPL, interpolated)")
    PH.nice_plot(ax[1], position=-0.03, direction="outward", ticklength=3)
    # ax[1].set_xticklabels(groups)
    ax[1].set_xlabel("Treatment")
    print("hue order, maxabr: ", order)
    if not picking:
        sns.swarmplot(
            data=grand_df,
            x="group",
            y="maxabr",
            hue="group",
            hue_order=order,
            order=order,
            ax=ax[2],
            legend="auto",  # palette=Experiment['plot_colors'],
            picker=False,
            zorder=100,
            clip_on=False,
        )
        sns.boxplot(
            data=grand_df,
            x="group",
            y="maxabr",
            hue="group",
            hue_order=order,
            order=order,
            saturation=0.15,
            orient="v",
            showfliers=False,
            linewidth=0.5,
            zorder=50,
            ax=ax[2],
        )
    else:
        sns.scatterplot(
            data=grand_df,
            x="group",
            y="maxabr",
            hue="group",
            hue_order=order,
            order=order,
            ax=ax[2],
            legend=False,  # palette=Experiment['plot_colors'],
            picker=True,
            clip_on=False,
        )
    picker_func2 = Picker(space=2, data=grand_df.copy(deep=True), axis=ax[2])
    PL.figure_handle.canvas.mpl_connect(
        "pick_event", lambda event: pick_handler(event, picker_func2)
    )
    PH.nice_plot(ax[2], position=-0.03, direction="outward", ticklength=3)
    ax[2].set_ylim(0, 8)
    ax[2].set_ylabel("Maximum ABR ($\mu V$)")
    # ax[2].set_xticklabels(groups)
    ax[2].set_xlabel("Treatment")

    for gs in groups:
        print(f"N {labels[gs]}: {n_gs[gs]:d}")

    if thrplot:
        sns.scatterplot(
            data=grand_df,
            x="ExpSPL",
            y="interpolated_threshold",
            hue="group",
            hue_order=order,
            # order=order
            ax=ax[3],
            legend=False,  # palette=Experiment['plot_colors'],
            picker=True,
            clip_on=False,
        )
        picker_func3 = Picker(space=2, data=grand_df.copy(deep=True), axis=ax[2])
        PL.figure_handle.canvas.mpl_connect(
            "pick_event", lambda event: pick_handler(event, picker_func3)
        )

    relabel_xaxes(ax[1], Experiment, angle=45)
    relabel_xaxes(ax[2], Experiment, angle=45)
    relabel_yaxes_threshold(ax[1])
    handles, previous_labels = ax[0].get_legend_handles_labels()
    # ax.legend(handles=handles, labels=new_labels)
    ax[0].legend(handles=handles, labels=[v for k, v in Experiment['new_xlabels'].items()], bbox_to_anchor=(0.59, 1.05))
        #   bbox_transform=PL.figure_handle.transFigure,)
        #   zorder=200)
    # place_legend(PL, 0.85, 0.95, Experiment)
    # rint("\nThresholds:\n", grand_df[['dataset', 'run', 'threshold']])
    # some stats...
    # compute_io_stats(grand_df, Groups=datasets)
    # compute_other_stats(grand_df, groups=groups)
    mpl.savefig("ABR_fig.pdf")

    mpl.show()


if __name__ == "__main__":
    # Say, "the default sans-serif font is COMIC SANS"
    import matplotlib
    from matplotlib import font_manager as FM

    mpl.rcParams.update(
        {"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]}
    )
    myFont = FM.FontProperties(family="Arial", size=11)

    # main()
    # analyze_from_excel(datasets= ['Tessa_BNE', 'Tessa_NF107Ai32', 'Tessa_CBA'])

    plot_io_from_excel(
        datasets=["Tessa_BNE"],
        groups=["B", "A", "AA", "AAA"],
        coding=get_coding_data(Experiment),
        picking=False,
    )
