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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Tuple

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

import src.abr_analyzer as abr_analyzer
import src.abr_funcs as abr_funcs
import src.abr_reader as ABR_Reader
import src.make_colormaps as make_colormaps

import ABR_Datasets  # just the dict describing the datasets
from src.abr_dataclasses import ABR_Data
from src.abr_dataclasses import plotinfo as PlotInfoDataclass
from src.get_configuration import get_configuration

pp = pprint.PrettyPrinter(indent=4)


ABRF = abr_funcs.ABRFuncs()


class PData(object):
    def __init__(self):
        pass

    def analyze_one_subject_run(
        self, subject_data, run, mode: str = "clicks", analyzer=None, freq: float = None
    ):

        assert analyzer is not None

        if mode == "clicks":
            s_data = subject_data.clickdata[run]
        elif mode == "tones":
            s_data = subject_data.tonemapdata[run]
        else:
            raise ValueError("mode must be 'clicks' or 'tones'")
        data = s_data
        if subject_data.subject is not None:
            datatitle = f"{subject_data.subject:s}\nNE: {subject_data.SPL:5.1f} dBSPL"
        else:
            datatitle = f"{str(subject_data.datapath.parts[-1]):s}\n{run:s}"
        thrs = {}
        if mode == "clicks":
            waves = data["waves"]
            t = np.array(data["timebase"])
            spls = data["spls"]
        elif mode == "tones":
            waves = data[freq]["waves"]
            t = np.array(data[freq]["timebase"])
            spls = data[freq]["spls"]
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
        waves_df.to_csv(Path("ABR_CSVs", f"{str(subject_data.datapath.parts[-1]):s}" + ".csv"))
        analyzer.analyze(t, waves, dev=subject_data.dev)
        pnp = analyzer.p1n1p2
        p1 = pnp["p1"]
        n1 = pnp["n1"]
        p2 = pnp["p2"]
        tb = analyzer.set_baseline(timebase=t)
        thr_spl, baseline_median_sd = analyzer.thresholds(  # A.threshold_spec(
            timebase=t,
            waves=waves,
            spls=spls,
            response_window=[2.0, 6.0],
            baseline_window=tb,
            SD=3.0,
            # spec_bandpass=self.spec_bandpass,
        )
        if len(spls) > 2:
            thr_spl, fit_data = fit_thresholds(spls, analyzer.ppio, baseline_median_sd)

        thrs[run] = thr_spl
        # halfspl = np.max(spls) / 2.0

        # generate a line demarcating the P1 (first wave)
        latmap = []
        spllat = []
        fitline = np.nan * len(spls)
        for j, spl in enumerate(spls):
            if spl > thrs[run]:
                latmap.append(t[p1[j][0]])  # get latency for first value
                spllat.append(spl)
        if len(latmap) > 2:
            latp = np.polyfit(spllat, latmap, 1)
            fitline = np.polyval(latp, spls)
        result = {
            "datatitle": datatitle,
            "waves": waves,
            "t": t,
            "spls": spls,
            "ppio": ppio,
            "thrs": thrs,
            "thr_spl": thr_spl,
            "fitline": fitline,
            "latmap": latmap,
            "p1": p1,
            "n1": n1,
            "Analyzer": analyzer,
        }
        return result

    def io_plot(
        self,
        ax: mpl.axes,
        result: dict,
        sf_cvt: float,
        subject_data: ABR_Data,
        subject_number: int,
        sex_marker: str,
        show_y_label: bool = True,
        show_x_label: bool = True,
    ):
        # Plot an IO dataset to the selected axes
        if ax is None:
            return
        spls = result["spls"]
        ax.set_title(
            result["datatitle"],
            x=0.5,
            y=1.0,
            fontdict={"fontsize": 7, "ha": "center", "va": "bottom"},
            transform=IOplot.transAxes,
        )  # directory plus file
        PH.nice_plot(ax, position=-0.03, direction="outward", ticklength=3)
        PH.set_axes_ticks(
            ax,
            xticks=[0, 25, 50, 75, 100],
            xticks_str=["0", "25", "50", "75", "100"],
            yticks=range(0, 7),
            yticks_str=[f"{y:d}" for y in range(0, 7)],
            y_minor=0.5 + np.arange(0, 6),
        )

        ax.plot(
            spls,
            sf_cvt * result["A"].rms_baseline,
            marker=result["A"].baselineMarker,
            markersize=3,
            color="grey",
            label="RMS baseline",
            alpha=0.35,
            clip_on=False,
        )
        ax.plot(
            spls,
            sf_cvt * result["A"].ppio,
            marker=sex_marker,  # A.ppioMarker,
            markersize=3,
            color=subject_data.summary_color_map[subject_number % subject_data.max_colors],
            label="P-P",
            clip_on=False,
        )

        # ensure sqrt gets a >= 0 value. Sometimes the baseline noise is greater than
        # the response window noise (below threshold).
        y = result["A"].rms_response ** 2 - result["A"].rms_baseline ** 2
        y[y < 0] = 0

        ax.plot(
            spls,
            sf_cvt * np.sqrt(y),
            marker=sex_marker,  # A.rmsMarker,
            markersize=3,
            color=subject_data.summary_color_map[subject_number % subject_data.max_colors],
            label="RMS signal",
            clip_on=False,
        )

        ax.set_ylim(0, 6.0)  # microvolts
        if show_y_label:
            label = r"ABR ($\mu V$)"
            ax.set_ylabel(f"{label:s} (rms)")
        if show_x_label:
            ax.set_xlabel("Level (dBSPL)")

        # if subject_data.psdIOPlot:
        #     ax2 = IOplot.twinx()
        #     ax2.plot(
        #         spls,
        #         result["A"].psdwindow,
        #         marker=result["A"].psdMarker,
        #         color="r",
        #         label="PSD signal",
        #         markersize=3,
        #     )
        #     ax2.tick_params("y", colors="r")
        #     # if index == 0 and subject_number == 0:
        #     #     handles, labels = IOplot.get_legend_handles_labels()
        #     legend = IOplot.legend(loc="center left")
        #     for label in legend.get_texts():
        #         label.set_fontsize(6)

    def plot_io_functions(
        self,
        ax,
        subject_data,
        run: str,
        results: dict,
        scale_factor: float = 1.0e-6,
        stimtype: str = "clicks",
        frequency: float = 1.0e3,
        marker: str = "o",
        color: str = "k",
        group: str = None,
        show_y_label: bool = True,
        show_x_label: bool = True,
    ):
        """plot_io_functions Generate a plot of the input/output function for a single subject

        Parameters
        ----------
        ax : matplotlib.axis
            plot axis into which the data will be plotted
        subject_data : dataclass holding subject data
            _description_
        run : str
            run identifier (e.g., 20180312_1422)
        results : dict
            results dictionary from the analysis
        scale_factor : float, optional
            factor to scale the data by, by default 1.0
        stimtype : str, optional
            stimulus type, by default "clicks"; could be "tones"
        marker : str, optional
            matplotlib marker to use for this plot, by default "o"
        """
        assert stimtype in ["clicks", "tones"]

        if stimtype == "clicks":
            if group is not None:
                label = f"{group:s} ({subject_data.clickdata[run]['SPL']:5.1f})"
            elif subject_data.clickdata[run]["Subject"] is not None:
                label = f"{subject_data.clickdata[run]['Subject']:s} ({subject_data.clickdata[run]['SPL']:5.1f})"
            else:
                label = f"{str(subject_data.datapath.parts[-1]):s}\n{run:s}"

        elif stimtype == "tones":
            freqs = list(subject_data.tonemapdata[run].keys())
            if subject_data.tonemapdata[run][freqs[0]]["Subject"] is not None:
                label = f"{subject_data.tonemapdata[run][freqs[0]]['Subject']:s} ({subject_data.tonemapdata[run][freqs[0]]['SPL']:5.1f})"
            else:
                label = f"{str(subject_data.datapath.parts[-1]):s}\n{run:s}"
            label = f"{int(frequency):d} Hz"
        spls = results["spls"]
        ax.plot(
            spls,
            scale_factor * results["Analyzer"].ppio,
            marker=marker,
            linestyle="-",
            linewidth=1.0,
            color=color,
            label=label,
        )

    def plot_waveforms(
        self,
        ax,
        subject_data,
        run: str,
        results: dict,
        scale_factor: float = 1e-6,
        stimtype: str = "clicks",
        frequency: float = 1.0e3,
        marker: str = "o",
        spl_color_map: dict = None,
        freq_color_map: dict = None,
        show_y_label: bool = True,
        show_x_label: bool = True,
        show_calbar: bool = True,
    ):
        """plot_waveforms _summary_

        Parameters
        ----------
        ax : _type_
            _description_
        subject_data : _type_
            _description_
        run : str
            _description_
        results : dict
            _description_
        scale_factor : float, optional
            _description_, by default 1e-6
        stimtype : str, optional
            _description_, by default "clicks"
        marker : str, optional
            _description_, by default "o"
        spl_color_map : dict, optional
            _description_, by default None
        freq_color_map : dict, optional
            _description_, by default None
        show_y_label : bool, optional
            _description_, by default True
        show_x_label : bool, optional
            _description_, by default True
        show_calbar : bool, optional
            _description_, by default True
        """
        sf = 20
        if show_calbar:
            # plot a calibration bar for the voltage traces
            sf = 8
            x = [7.5, 7.5]
            y = np.array([0, 1e-6]) * sf * scale_factor + 105.0  # put at 105 dB...
            ax.plot(x, y, linewidth=1.5)  # put 1 uV cal bar at highest sound level
            ax.text(
                x[0] + 0.1,
                np.mean(y),
                s=r"1 $\mu V$",
                ha="left",
                va="center",
                fontsize=7,
            )

        IO = np.zeros(len(results["spls"]))

        linewidth = 1.0
        for j, spl in enumerate(results["spls"]):
            if spl == results["thr_spl"]:  # highlight the threshold spl
                ax.plot(
                    results["t"],
                    0 * results["waves"][j] * scale_factor + spl,
                    color=[0.5, 0.5, 0.5, 0.4],
                    linewidth=5,
                )
            if stimtype == "clicks":
                c = spl_color_map[spl]
            elif stimtype == "tones":
                c = freq_color_map
            ax.plot(
                results["t"],
                sf * results["waves"][j] * scale_factor + spl,
                color=c,
                linewidth=linewidth,
            )

            if subject_data.show_dots:
                for p in results["p1"][j]:
                    ax.plot(
                        results["t"][p],
                        sf * results["waves"][j][p] * scale_factor + spl,
                        "ro",
                        markersize=2,
                    )
                for p in results["n1"][j]:
                    ax.plot(
                        results["t"][p],
                        sf * results["waves"][j][p] * scale_factor + spl,
                        "bo",
                        markersize=2,
                    )
            if len(results["latmap"]) > 2 and subject_data.show_dots:
                ax.plot(results["fitline"], results["spls"], "g-", linewidth=0.7)
            # if len(result['latmap']) > 2 and subject_data.show_dots:
            #     plottarget.plot(result['A'].p1_latencies[0], result['A'].p1_latencies[1], "g-", linewidth=0.7)

            if spl >= results["thr_spl"] or len(results["latmap"]) <= 2:
                IO[j] = scale_factor * (
                    results["waves"][j][results["p1"][j][0]]
                    - results["waves"][j][results["n1"][j][0]]
                )
            else:
                ti = int(results["fitline"][j] / (subject_data.sample_rate * 1000.0))
                if ti < len(results["waves"][j]):
                    IO[j] = scale_factor * results["waves"][j][ti]

        if show_y_label:
            ax.set_ylabel("dBSPL")
        if show_x_label:
            ax.set_xlabel("T (ms)")
        ax.set_xlim(0, 10.0)
        ax.set_ylim(10.0, 115.0)
        PH.set_axes_ticks(
            ax,
            xticks=[0, 2, 4, 6, 8, 10],
            xticks_str=["0", "2", "4", "6", "8", "10"],
            x_minor=np.arange(0, 10, 0.5),
            yticks=[0, 40, 80, 120],
            yticks_str=["0", "40", "80", "120"],
            y_minor=[10, 20, 30, 50, 60, 70, 90, 100, 110],
        )
        ax.set_title(
            results["datatitle"],
            x=0.5,
            y=0.90,
            fontdict={"fontsize": 6, "ha": "center", "va": "bottom"},
            transform=ax.transAxes,
        )

    def plotClicks(
        self,
        subject_data,
        subject_number: int = 0,
        select: str = None,
        configuration: dict = None,
        datadir: Union[Path, str] = None,
        color_map: dict = None,
        group_by: str = "group",
        superimposed_io_plot=None,
        show_y_label: bool = True,  # for the many-paneled plots
        show_x_label: bool = True,
        PSDplot: bool = False,
        verbose=False,
        do_individual_plots: bool = True,
    ) -> Tuple[List, object]:
        """plotClicks Generate plots of click ABRS for a SINGLE subject
        Generates one figure with 2 subplots:
        IO function (amplitude vs. SPL)
        Stacked waveforms.

        Parameters
        ----------
        subject_data : _type_
            _description_
        subject_number : int, optional
            _description_, by default 0
        select : str, optional
            _description_, by default None
        datadir : Union[Path, str], optional
            _description_, by default None
        color_map : dict, optional
            _description_, by default None
        show_y_label : bool, optional
            _description_, by default True
        verbose : bool, optional
            _description_, by default False

        Returns
        -------
        List
            _description_
        """

        self.click_io_plots: list = []  # save a list of the plots that are generated
        self.abr_df = subject_data.abr_dataframe
        drow = self.abr_df[self.abr_df.Subject == subject_data.subject]
        sex_marker = "D"
        if not pd.isnull(drow.Sex.values):
            if drow.Sex.values[0] == "M":
                marker = "x"
                sex_marker = "x"
            elif drow.Sex.values[0] == "F":
                marker = "o"
                sex_marker = "o"

        spl_color_map, freq_colormap = make_colormaps.make_spl_freq_colormap()
        if do_individual_plots:
            self.click_io_plot = PH.regular_grid(1, 2, order="rowsfirst", figsize=(6, 3))
        IO_DF = []
        sf = 8
        sf_cvt = 1e6
        for index, run in enumerate(list(subject_data.clickdata.keys())):  # for all the click data
            # datatitle = datatitle.replace('_', '\_')  # if TeX is enabled, will need to escape the underscores
            result = self.analyze_one_subject_run(
                subject_data,
                run,
                mode="clicks",
                analyzer=abr_analyzer.Analyzer(sample_frequency=subject_data.sample_freq),
            )
            group_colormap = configuration["plot_colors"]
            group_symbolmap = configuration["plot_symbols"]
            if subject_data.group in group_colormap:
                group_color = group_colormap[subject_data.group]
                group_symbol = group_symbolmap[subject_data.group]

            else:
                group_color = "grey"
                group_symbol = "+"
            # plot IO functions on standard plot, panel A
            if do_individual_plots:
                self.plot_io_functions(
                    self.click_io_plot.axdict["A"],
                    subject_data=subject_data,
                    run=run,
                    results=result,
                    scale_factor=sf_cvt,
                    stimtype="clicks",
                    marker=marker,
                    color=color_map[subject_data.subject],
                    group=subject_data.group,
                )

                # plot a panel of voltage traces (standard plot, panel B)
                self.plot_waveforms(
                    self.click_io_plot.axdict["B"],
                    subject_data=subject_data,
                    run=run,
                    results=result,
                    scale_factor=sf_cvt,
                    stimtype="clicks",
                    marker="o",
                    spl_color_map=spl_color_map,
                    show_calbar=index == 0,
                )

            if superimposed_io_plot is None:  # need to build superimposed plot across subjects
                print("Creating plot to superimpose IO functions")
                sizer = {'A': {'pos': [0.08, 0.55, 0.08, 0.8]}, 'B': {'pos': [0.67, 0.25, 0.08, 0.8]}}
                superimposed_io_plot = PH.arbitrary_grid(sizer=sizer ,order="rowsfirst", figsize=(8, 5.5))
            
            if subject_data.clickdata[run]["Subject"] is not None:
                label = f"{subject_data.clickdata[run]['Subject']:s} {subject_data.clickdata[run]['strain']:s} ({subject_data.clickdata[run]['SPL']:5.1f})"
            # elif group_by == "group":
            #     label = f"{subject_data.clickdata[run]['group']:s}"
            # elif group_by == "strain":
            #     label = f"{subject_data.clickdata[run]['strain']:s} ({subject_data.clickdata[run]['SPL']:5.1f})"
            else:
                label = f"{str(subject_data.datapath.parts[-1]):s}\n{run:s}"
            if label not in subject_data.superIOLabels:
                subject_data.superIOLabels.append(label)
            else:
                label = None

            superimposed_io_plot.axdict["A"].plot(
                result["spls"],
                sf_cvt * result["Analyzer"].ppio,
                marker=group_symbol,  # sex_marker,
                linestyle="-",
                color=group_color,
                label=label,
            )

            IO_DF.append(
                [
                    subject_data.subject,
                    run,
                    result["spls"],
                    sf_cvt * result["Analyzer"].ppio,
                    result["thrs"][run],
                    subject_data.clickdata[run]["group"],
                    subject_data.clickdata[run]["strain"],
                ]
            )

            PH.set_axes_ticks(
                superimposed_io_plot.axdict["A"],
                xticks=[0, 25, 50, 75, 100],
                xticks_str=["0", "25", "50", "75", "100"],
                yticks=range(0, 7),
                yticks_str=[f"{y:d}" for y in range(0, 7)],
                y_minor=0.5 + np.arange(0, 6),
            )
        legend = superimposed_io_plot.axdict["A"].legend(ncol=4, loc="upper left")
        for label in legend.get_texts():
            label.set_fontsize(5)
        if show_x_label:
            superimposed_io_plot.axdict["A"].set_xlabel("Click level (dBSPL)")
        if show_y_label:
            label = r"ABR ($\mu V$)"
            superimposed_io_plot.set_ylabel(f"{label:s} (V)")

            sr = result["datatitle"].split("\n")
            if len(sr) > 1:
                subjectID = sr[0]
                run = sr[1]
            else:
                run = 0
                subjectID = result["datatitle"]
            for i_level, spl in enumerate(result["spls"]):
                IO_DF.append(
                    [
                        subjectID,
                        run,
                        spl,
                        sf_cvt * result["Analyzer"].ppio[i_level],
                        result["thrs"][s],
                        subject_data.clickdata[s]["group"],
                    ]
                )
                # IO_DF = [spls, (sf_cvt*A.ppio).tolist(), str(subject_data[s]["group"])]

            # print out the data for import into another plotting program, such as Prism or Igor
            if verbose:
                print("*" * 20)
                print(f"dataset: {run:s}")
                print("t\tV")
                for i, spl in enumerate(result["spls"]):
                    print(f"{spl:.1f}\t{sf_cvt * result['Analyzer'].ppio[i]:.3f}")
                print("*" * 20)

            # generic io plot for cell
            # self.io_plot(IOplot, result, sf_cvt, subject_data, subject_number,
            #         sex_marker, show_y_label, show_x_label)

            if PSDplot:  # power spectral density
                for j, spl in enumerate(result["spls"]):
                    PSDplot.semilogy(np.array(result["A"].fr), np.array(result["A"].psd[j]))
                PSDplot.set_ylim(1e-6, 0.01)
                PSDplot.set_xlim(100.0, 2000.0)

        print("-" * 40)
        self.thrs = result["thrs"]
        for s in list(self.thrs.keys()):
            print(f"dataset: {s:s}  thr={self.thrs[s]:.0f}")
        print("-" * 40)

        return IO_DF, superimposed_io_plot

    def plotTones(
        self,
        subject_data,
        subject_number: int = 0,
        select: str = None,
        datadir: Union[Path, str] = None,
        color_map: dict = None,
        IOplot=None,
        PSDplot=None,
        superimposed_io_plot=None,
        show_y_label: bool = True,  # for the many-paneled plots
        show_x_label: bool = True,
        verbose=False,
    ):
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
        self.thrs = {}  # holds thresholds for this dataset, by frequency
        IO_DF = []  # build a dataframe of the IO funcitons from a list.
        spl_color_map, freq_colormap = make_colormaps.make_spl_freq_colormap()
        Analyzer = abr_analyzer.Analyzer(sample_frequency=subject_data.sample_freq)
        marker = "o"
        sf_cvt = 1e6
        self.tone_io_plot = PH.regular_grid(1, 2, order="rowsfirst", figsize=(6, 3))
        for index, run in enumerate(list(subject_data.tonemapdata.keys())):
            mapfreqs = subject_data.freqs[run]
            thr_spls = 100.0 + np.zeros(len(mapfreqs))
            for k, fr in enumerate(mapfreqs):  # next key is frequency
                result = self.analyze_one_subject_run(
                    subject_data,
                    run,
                    analyzer=Analyzer,
                    mode="tones",
                    freq=fr,
                )

                # spl_list = [x[0] for x in Analyzer.p1n1p2_amplitudes]
                # p1n1p2_list = [x[1] for x in Analyzer.p1n1p2_amplitudes]
                # plot IO functions on standard plot, panel A
                self.plot_io_functions(
                    self.tone_io_plot.axdict["A"],
                    subject_data=subject_data,
                    run=run,
                    results=result,
                    scale_factor=sf_cvt,
                    stimtype="tones",
                    frequency=int(fr),
                    marker=marker,
                    color=freq_colormap[int(fr)],
                )
                # plot a panel of voltage traces (standard plot, panel B)
                self.plot_waveforms(
                    self.tone_io_plot.axdict["B"],
                    subject_data=subject_data,
                    run=run,
                    results=result,
                    scale_factor=sf_cvt,
                    stimtype="tones",
                    frequency=fr,
                    marker="o",
                    spl_color_map=spl_color_map,
                    freq_color_map=freq_colormap[int(fr)],
                    show_calbar=index == 0,
                )

                for j, spl in enumerate(result["spls"]):
                    if spl == thr_spls[k]:  # mark the threshold with a gray band
                        self.tone_io_plot.axdict["B"].plot(
                            result["t"],
                            0 * result["waves"][j] * 1e6 + spl,
                            color=[0.5, 0.5, 0.5, 0.4],
                            linewidth=5,
                        )
                    # self.tone_io_plot.axdict["B"].plot(
                    #     result["t"], 4 * result["waves"][j] * 1e6 + spl, color=spl_color_map[spl]
                    # )
                    IO_DF.append(
                        [
                            result["datatitle"],
                            run,
                            spl,
                            fr,
                            sf_cvt * result["ppio"][k],
                            result["thrs"][run],
                            subject_data.tonemapdata[run][fr]["group"],
                        ]
                    )
                # plottarget.set_xlim(0, 8.0)
                # plottarget.set_ylim(10.0, 110.0)
                # frtitle = "%.1f kHz" % (float(fr) / 1000.0)
                # plottarget.title.set_text(frtitle)
                # plottarget.title.set_size(9)
                # if fit_data is not None:
                #     if i == 0:
                #         iop, ioax = mpl.subplots(1, 1)
                #     ioax.plot(spl_list, p1n1p2_list, "o", color=freq_colormap[fr])
                #     ioax.plot(fit_data[0], fit_data[1], "--", color=freq_colormap[fr])
                #     ioax.set_xlim(0, 100)
                #     ioax.set_ylim(0, 5e-6)

                # if fr not in self.thrs.keys():
                #     self.thrs[fr] = [subject_data.tonemapdata[s][fr], thr_spls]
                # else:  # repeat frequency, so add the threshold information
                #     self.thrs[fr].append([subject_data.tonemapdata[s][fr], thr_spls])
        PH.cleanAxes(self.tone_io_plot.axdict["A"])
        legend = self.tone_io_plot.axdict["A"].legend(loc="upper left", fontsize=5)
        return IO_DF, superimposed_io_plot
        # mpl.show()
        # if pdf is not None:
        #     pdf.savefig()
        #     mpl.close()

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
        re_date = re.compile(r"(?P<date>[\d]{2}-[\d]{2}-[\d]{4})")
        re_genotype_WT = re.compile(r"(?P<GT>_WT)")
        re_genotype_KO = re.compile(r"(?P<GT>_KO)")
        # put data in pd dataframe
        T = []
        # parse information about mouse, experiment, etc.
        for i, dataset in enumerate(allthrs):
            print("dataset keys: ", allthrs[dataset])
            thrs = []
            for run in allthrs[
                dataset
            ].thrs.keys():  # find lowest threshold for all click runs this dataset
                thrs.append(allthrs[dataset].thrs[run])
            thr = np.min(thrs)
            # for m in allthrs[dataset]:
            #     name = str(Path(dataset).name)
            #     thr = allthrs[dataset][m]
            #     exp = re_control.search(name)
            #     if exp is not None:
            #         exposure = "control"
            #     else:
            #         exposure = "exposed"
            #     sham = re_sham_exposed.search(name)
            #     if sham is not None:
            #         exposure = "sham"
            #     unexposed = re_un_exposed.search(name)
            #     if unexposed is not None:
            #         exposure = "unexposed"
            #     exp = re_noise_exposed.search(name)
            #     if exp is not None:
            #         exposure = "exposed"
            #     sex = re_sex.search(name)
            #     if sex is not None:
            #         sex = sex.groups()[0][1]
            #     else:
            #         sex = "U"
            #     age = re_age.search(name)
            #     if age is not None:
            #         P_age = age.groups()[0][1:]
            #         day_age = int(P_age[1:])
            #     else:
            #         P_age = "U"
            #         day_age = np.nan
            #     Genotype = "ND"  # not defined
            #     gtype = re_genotype_WT.search(name)
            #     if gtype is not None:
            #         Genotype = gtype.groups()[0][1:]  # remove the underscore
            #     gtype = re_genotype_KO.search(name)
            #     if gtype is not None:
            #         Genotype = gtype.groups()[0][1:]

            # meas = [name, thr, exposure, Group, sex, P_age, day_age, Genotype]
            print("thr: ", thr)
            meas = [
                dataset,
                allthrs[dataset].coding.Subject,
                thr,
                allthrs[dataset].group,
                allthrs[dataset].coding.sex,
                allthrs[dataset].coding.age,
                allthrs[dataset].coding.genotype,
            ]

            T.append(meas)

        df = pd.DataFrame(
            T,
            columns=[
                "dataset",
                "Subject",
                "threshold",
                "Group",
                "Sex",
                "age",
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
        re_date = re.compile(r"(?P<date>[\d]{2}-[\d]{2}-[\d]{4})")
        use_fr = [2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0]
        # put data in pd dataframe
        T = []

        # parse information about mouse, experiment, etc.
        for i, dataset in enumerate(allthrs):
            # print(allthrs[d])
            # print(allthrs[d].group)
            # print(allthrs[d].subject)
            # print(dir(allthrs[d]))
            # print("frequs: ", dataset, allthrs[d].freqs)
            # print("spls: ", d, allthrs[d].spls)
            # print("thrs: ", d, allthrs[d].thrs)
            # print("tonemaps: ", d, allthrs[d].tonemaps)

            # print('dataset: ', dataset)
            for i, abr_map in enumerate(allthrs[dataset].tonemaps):
                tone_map = allthrs[dataset].tonemaps[abr_map]  # 2 el list, freq, threshold
                fr = np.array(allthrs[dataset].thrs[abr_map][0])
                fr_jit = fr + np.random.uniform(-fr / 8, fr / 8)
                thr = np.array(allthrs[dataset].thrs[abr_map][1])
                #  uor j in range(len(allthrs[d][m][0])):
                # name = str(Path(d).parts[-1])
                # fr = np.array(allthrs[d][m][0][j]) / 1000.0
                # if fr not in use_fr:
                #     continue
                # fr_jit = fr + np.random.uniform(-fr / 8, fr / 8)
                # thr = allthrs[d][m][1][j]
                # exp = re_control.search(name)
                # if exp is not None:
                #     exposure = "control"
                # else:
                #     exposure = "exposed"
                # sham = re_sham_exposed.search(name)
                # if sham is not None:
                #     exposure = "sham"
                # unexposed = re_un_exposed.search(name)
                # if unexposed is not None:
                #     exposure = "unexposed"
                # exp = re_noise_exposed.search(name)
                # if exp is not None:
                #     exposure = "exposed"
                # sex = re_sex.search(name)
                # if sex is not None:
                #     sex = sex.groups()[0][1]
                # else:
                #     sex = "U"
                # age = re_age.search(name)
                # if age is not None:
                #     P_age = age.groups()[0][1:]
                #     day_age = int(P_age[1:])
                # else:
                #     P_age = "U"
                #     day_age = np.nan

                meas = [
                    dataset,
                    allthrs[dataset].coding.Subject,
                    fr,
                    fr_jit,
                    thr,
                    allthrs[dataset].group,
                    allthrs[dataset].coding.sex,
                    allthrs[dataset].coding.age,
                ]
                T.append(meas)
        df = pd.DataFrame(
            T,
            columns=[
                "dataset",
                "Subject",
                "Freq",
                "Freq_jittered",
                "threshold",
                # "noise_exposure",
                "Group",
                "Sex",
                "Age",
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
        print("df: ", df.head())
        # print(df.genotype.unique())
        # print(df.threshold.unique())
        # print(df.Sex.unique())
        sns.boxplot(
            x="group",  # "genotype",
            y="threshold",
            hue="group",  # "Sex",
            data=df,
            whis=2.0,
            width=0.5,
            dodge=True,
            fliersize=0.15,
            # order=["WT", "KO"],
            palette="dark:#d9d9d9",
            ax=ax,
        )
        sns.swarmplot(
            x="group",  # "genotype",
            y="threshold",
            hue="group",  # "Sex",
            data=df,
            # order=["WT", "KO"],
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

    def plotToneThresholds(self, allthrs, abr_dataset, name, show_lines: bool = True):
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

        def get_group(row):
            group = [row.subject]
            row.Group = group
            return

        gcolors = {"B": "orange", "A": "blue", "AA": "green", "AAA": "brown"}
        for g in df.Group.unique():
            dg = df[df["Group"] == g]
            for ig in dg.index:
                ax.plot(
                    dg.loc[ig, "Freq_jittered"] / 1000.0,
                    dg.loc[ig, "threshold"],
                    "o-",
                    color=gcolors[g],
                    alpha=0.5,
                )
                print(dg.loc[ig, "Freq"] / 1000.0, dg.loc[ig, "threshold"], g)
            # sns.lineplot(
            #     x="Freq",
            #     y="threshold",
            #     hue="Group",
            #     data=dg,
            #     ax=ax,
            #     # err_style="band",
            #     # errorbar=('ci', 68),
            # )
        # axs = sns.scatterplot(
        #     x="Freq_jittered",
        #     y="threshold",
        #     hue="Subject",
        #     data=df,
        #     alpha=0.65,
        #     ax=ax,
        #     s=10,
        # )
        ax.set_xscale("log", nonpositive="clip", base=2)
        ax.set_xlim(1, 65)
        ax.set_ylim(20, 100)

        xt = [2.0, 4.0, 8.0, 16.0, 32.0, 48.0, 64.0]
        mpl.xticks(xt, [str(x) for x in xt])
        legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize=7)
        mpl.show()
        thrs_sorted = None
        frmean = None
        frstd = None
        return (thrs_sorted, frmean, frstd)


###
###======================================================================================
###


def build_click_grid_plot(nplots=1):
    m, n = PH.getLayoutDimensions(nplots)
    # print("Grid: ", m, n)
    # print("dirs: ", dirs)

    if m > 1:
        h = 2.5 * m
        if h > 10.5:
            h = 10.5
    else:
        h = 4
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
        verticalspacing=0.06,
        horizontalspacing=0.06,
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
    PlotInfo = PlotInfoDataclass(
        m=m,
        n=n,
        subject_number=0,
        Plot_waveforms=Plot_f,
        Plot_IO_functions=Plot_f2,
        Plot_IO_overlay=Plot_f4,
        IOax=IOax,
        axarr=axarr,
        axarr2=axarr2,
    )
    return PlotInfo


def build_tone_plot_grid(nplots=1):
    m, n = PH.getLayoutDimensions(nplots)
    # grid is based on the number of subects with data
    print("Toneplot Grid: ", m, n, nplots)
    # print("dirs: ", dirs)

    if m > 1:
        h = 2.5 * m
        if h > 10.5:
            h = 10.5
    else:
        h = 4
    horizontalspacing = 0.08
    if n > 5:
        horizontalspacing = 0.08 / (n / 5.0)
    print("Plot array and count: ", m, n, nplots)

    # generate plot grid for waveforms
    Plot_waveforms = PH.regular_grid(
        m,
        n,
        order="rowsfirst",
        figsize=(11, h),
        verticalspacing=0.06,
        horizontalspacing=0.06,
    )
    if Plot_waveforms.axarr.ndim > 1:
        axarr_waveforms = Plot_waveforms.axarr.ravel()
    else:
        axarr_waveforms = Plot_waveforms.axarr

    for ax in Plot_waveforms.axarr:
        PH.nice_plot(ax, position=-0.03, direction="outward", ticklength=3)

    # generate plot grid for individual IO functions
    Plot_IO_functions = PH.regular_grid(
        m,
        n,
        order="rowsfirst",
        figsize=(11, h),
        verticalspacing=0.04,
        horizontalspacing=0.04,
    )
    for ax in Plot_IO_functions.axarr:
        PH.nice_plot(ax, position=-0.03, direction="outward", ticklength=3)
    if Plot_IO_functions.axarr.ndim > 1:
        axarr_IO_functions = Plot_IO_functions.axarr.ravel()
    else:
        axarr_IO_functions = Plot_IO_functions.axarr

    # generate plot space for tone IO overlay and threshold summary on right
    Plot_IO_overlay = PH.regular_grid(
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

    IOax = Plot_IO_overlay.axarr.ravel()
    for ax in Plot_IO_overlay.axarr:
        PH.nice_plot(ax, position=-0.03, direction="outward", ticklength=3)
    PlotInfo = PlotInfoDataclass(
        m=m,
        n=n,
        subject_number=0,
        Plot_waveforms=Plot_waveforms,
        Plot_IO_functions=Plot_IO_functions,
        Plot_IO_overlay=Plot_IO_overlay,
        IOax=IOax,
        axarr=axarr_waveforms,
        axarr2=Plot_IO_functions,
    )
    return PlotInfo


def populate_io_plot(
    subject_data,
    select,
    datadir,
    plot_index: int,
    subject_number: int,
    configuration: dict = None,
    group_by: str = "group",
    superimposed_io_plot: object = None,
    mode: str = "clicks",
    color_map: list = ["r", "b"],
    do_individual_plots: bool = True,
):
    """populate_io_plot _summary_

    Parameters
    ----------
    subject_data : _type_
        _description_
    select : _type_
        _description_
    datadir : _type_
        _description_
    plot_index : int
        _description_
    subject_number : int
        _description_
    group_by : str, optional
        color the data by groups: "group", "strain", "subject"
    superimposed_io_plot : object, optional
        _description_, by default None
    mode : str, optional
        _description_, by default "clicks"
    color_map : list, optional
        _description_, by default ["r", "b"]

    Returns
    -------
    _type_
        _description_
    """
    xlab = False
    ylab = False

    if mode == "clicks":
        IOdata, superimposed_io_plot = PData().plotClicks(
            subject_data,
            select=select,
            datadir=datadir,
            configuration=configuration,
            subject_number=subject_number,
            group_by=group_by,
            superimposed_io_plot=superimposed_io_plot,
            color_map=color_map,
            show_x_label=xlab,
            show_y_label=ylab,
            do_individual_plots=do_individual_plots,
        )
    elif mode == "tones":
        IOdata, superimposed_io_plot = PData().plotTones(
            subject_data,
            select=select,
            datadir=datadir,
            subject_number=subject_number,
            superimposed_io_plot=superimposed_io_plot,
            color_map=color_map,
            show_x_label=xlab,
            show_y_label=ylab,
        )

    return IOdata, superimposed_io_plot


def do_clicks(
    dsname: str,
    ABR_Datasets: dict = None,
    configuration: dict = None,
    coding: pd.DataFrame = None,
    group_by: str = "group",
    top_directory: Union[str, Path] = None,
    dirs: list = None,
    plot_info: object = None,
    nplots: int = 1,
    plot_index: int = 0,
    do_individual_plots: bool = True,
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

    if len(dirs) == 0:
        CP.cprint("r", f"No Directories found: {str(dsname):s}, {str(top_directory):s}")
        return plot_info, None
    # if plot_info is None:
    #     plot_info = build_click_plot(nplots)
    # do analysis and plot, and accumulate thresholds while doing so.
    allthrs = {}
    IO_DF = []
    abr_dataframe = ABR_Reader.get_ABR_Dataset(datasetname=dsname, configuration=configuration)

    clicksel = {}
    subjects = set(abr_dataframe.Subject)
    # print("Subjects: ", subjects)
    match group_by:
        case "subjects":
            color_map = make_colormaps.make_subject_colormap(subjects)
        case "groups":
            color_map = make_colormaps.make_group_colormap(abr_dataframe.Group)
        case "strain":
            color_map = make_colormaps.make_group_colormap(abr_dataframe.strain)

    superimposed_io_plot = None
    for subject_number, subject in enumerate(subjects):
        Group = coding[coding.Subject == subject].Group.values[0]
        subject_data = ABR_Reader.ABR_Reader()
        subject_data.setup(
            datapath=Path(top_directory),
            configuration=configuration,
            mode="clicks",
            info=ABR_Datasets[dsname],
            datasetname=dsname,
            abr_dataframe=abr_dataframe,
            subject=subject,
            group=Group,
            sex=coding[coding.Subject == subject].sex.values[0],
            age=coding[coding.Subject == subject].age.values[0],
        )
        if subject_data.error:
            continue  # skip this directory - probably none defined
        runs = subject_data.abr_subject.Runs
        if len(runs) == 0:
            continue
        # print("runs: ", runs.values[0])
        clicksel[subject] = [
            x.split(":")[1] for x in runs.values[0].split(",") if x.strip().startswith("click:")
        ]
        # print("clicksel: ", clicksel[subject])
        nsel = 32
        if subject_number == 0:
            subject_data.summary_color_map = ABRF.makeColorMap(nsel, list(range(nsel)))

        # print("doClicks: Getting Click data with : ", clicksel[subject], P.datadir, dsname)
        subject_data.getClickData(
            select=clicksel[subject], directory=subject_data.datadir, configuration=configuration
        )
        IOdata, superimposed_io_plot = populate_io_plot(
            subject_data,
            clicksel[subject],
            mode="clicks",
            configuration=configuration,
            datadir=dsname,
            plot_index=plot_index,
            subject_number=subject_number,
            group_by=group_by,
            color_map=color_map,
            superimposed_io_plot=superimposed_io_plot,
            do_individual_plots=do_individual_plots,
        )
        dirname = str(Path(subject_data.datadir).name)
        allthrs[dirname] = subject_data
        IO_DF.extend(IOdata)

    clickIODF = pd.DataFrame(IO_DF, columns=["Subject", "run", "spl", "ppio", "thrs", "Group", "strain"])
    clickIODF["group_cat"] = clickIODF["Group"].astype("category")
    filled_circle = MarkerStyle("o", fillstyle="full")
    filled_square = MarkerStyle("s", fillstyle="full")
 
    if superimposed_io_plot is not None:
        # superimposed_io_plot.axdict['B'].plot(clickIODF["spl"], clickIODF["thrs"], "o", color="black")
        sns.boxplot(
            x="Group",
            y="thrs",
            hue='group_cat',
            order = configuration['plot_order']["Group"],
            data=clickIODF,
            ax = superimposed_io_plot.axdict['B'],
            whis=[5, 95],
        )
        smarkers = {"VGAT": 'o', 'GlyT2': 's'}
        for s in clickIODF['strain'].unique():
            sns.stripplot(
                x="Group",
                y="thrs",
                hue='group_cat',
                order = configuration['plot_order']["Group"],
                data=clickIODF[clickIODF['strain'] == s],
                marker=smarkers[s],
                ax = superimposed_io_plot.axdict['B'],
                linewidth=0.5,
                edgecolor="black",
            )
        # sns.lineplot(
        #     x="spl",
        #     y="ppio",
        #     # hue="group_cat",
        #     data=clickIODF,
        #     ax=superimposed_io_plot.axdict['B'],
        #     # hue_order=list(configuration['plot_symbols'].keys()),
        #     # markers=False,  # [fill_circ, fill_square],
        #     # err_style="band",
        #     # errorbar="sd",
        #     # err_kws={"alpha": 0.8, "linewidth": 0.75},
        #     # mew=1.0,
        #     # linewidth=1.5,
        #     # markersize=1,
        # )
        superimposed_io_plot.axdict['B'].legend(loc="upper left", fontsize=7)
    mpl.show()
    clickIODF.to_csv("ClickIO.csv")

    # spls = set(clickIODF["spl"])
    # clickIOWTlist = []
    # clickIOKOlist = []
    # subjs = set(clickIODF['subject'])
    # print(clickIODF.head())
    # for icol in subjs:
    #     if clickIODF.at['group'] == 'WT':
    #         clickIOWT['subject']

    # population_thrdata = self.plotClickThresholds(
    #     allthrs, name="Click Thresholds", ax=plot_info.IOax[1]
    # )
    return plot_info, clickIODF


def do_tones(
    dsname: str,
    ABR_Datasets: dict = None,
    configuration: dict = None,
    coding: pd.DataFrame = None,
    top_directory: Union[str, Path] = None,
    dirs: Union[list, None] = None,
    plot_info: object = None,
    nplots: int = 1,
    plot_index: int = 0,
):
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

    tonesel = {}
    abr_dataframe = ABR_Reader.get_ABR_Dataset(datasetname=dsname, configuration=configuration)
    print("ABR Dataframe Group: ", abr_dataframe["Group"])
    fofilename = Path(top_directory, "ToneSummary.pdf")
    allthrs = {}
    IO_DF = []
    print("dsname: ", dsname)

    subjects = set(abr_dataframe.Subject)
    print("Subjects: ", subjects)
    print(f"Processing {len(dirs):d} directories")
    color_map = make_colormaps.make_subject_colormap(subjects)
    superimposed_io_plot = None
    # if pdf is not None:
    #     print("pdf defined")
    #     with PdfPages(fofilename) as pdf:
    #         for k, subject in enumerate(tonesel):
    #             P = ABR_Reader(
    #                 datapath=Path(top_directory),
    #                 mode="tones",
    #                 info=ABR_Datasets[dsname],
    #                 datasetname=dsname,
    #                 abr_dataframe=abr_dataframe,
    #                 subject=subject,
    #             )
    #             # P = ABR(Path(top_directory, dirs[k]), "tones", datasetname=dsname)
    #             if P.error:
    #                 continue
    #             P.getToneData(select=tonesel[subject], directory=dirs[subject])
    #             P.plotTones(select=tonesel[subject], pdf=pdf)
    #             allthrs[dirs[subject]] = P.thrs
    # else:
    for subject_number, subject in enumerate(subjects):
        print("Subject: ", subject)
        Group = coding[coding.Subject == subject].Group.values[0]
        subject_data = ABR_Reader.ABR_Reader()
        subject_data.setup(
            datapath=Path(top_directory),
            configuration=configuration,
            mode="tones",
            info=ABR_Datasets[dsname],
            datasetname=dsname,
            abr_dataframe=abr_dataframe,
            subject=subject,
            group=Group,
            sex=coding[coding.Subject == subject].sex.values[0],
            age=coding[coding.Subject == subject].age.values[0],
        )
        if subject_data.error:
            continue  # skip this directory - probably none defined
        runs = subject_data.abr_subject.Runs
        if len(runs) == 0:
            continue
        # print("runs: ", runs.values[0])
        tonesel[subject] = [
            x.split(":")[1] for x in runs.values[0].split(",") if x.strip().startswith("tone:")
        ]
        nsel = 32
        if subject_number == 0:
            subject_data.summary_color_map = ABRF.makeColorMap(nsel, list(range(nsel)))

        subject_data.getToneData(select=tonesel[subject], directory=subject_data.datadir)
        IOdata, superimposed_io_plot = populate_io_plot(
            subject_data,
            tonesel[subject],
            mode="tones",
            datadir=dsname,
            superimposed_io_plot=superimposed_io_plot,
            plot_index=plot_index,
            subject_number=subject_number,
            color_map=color_map,
        )
        if IOdata is None:
            continue
        dirname = str(Path(subject_data.datadir).name)
        allthrs[dirname] = subject_data
        IO_DF.extend(IOdata)
    toneIODF = pd.DataFrame(
        IO_DF, columns=["Subject", "run", "spl", "freq", "ppio", "thrs", "Group"]
    )
    toneIODF["group_cat"] = toneIODF["Group"].astype("category")
    filled_circle = MarkerStyle("o", fillstyle="full")
    filled_square = MarkerStyle("s", fillstyle="full")

    # population_thrdata = subject_data.plotToneThresholds(
    #     allthrs, abr_dataframe, name="Tone Thresholds"
    # )
    # print("pop thr data: ", population_thrdata)
    # print("Hz\tmean\tstd\t individual")
    # if population_thrdata[0] is not None:
    #     for i, f in enumerate(population_thrdata[0].keys()):
    #         print(
    #             f"{f:.1f}\t{population_thrdata[1][i]:.1f}\t{population_thrdata[2][i]:.1f}",
    #             end="",
    #         )
    #         for j in range(len(population_thrdata[0][f])):
    #             print(f"\t{population_thrdata[0][f][j]:.1f}", end="")
    #         print("")
    mpl.show()
    tthr_filename = Path(top_directory, "ToneThresholds.pdf")
    return plot_info, toneIODF


def find_stimuli(top_directory, mode="click"):
    """find_stimuli Find directories that contain responses to
    either click or tone stimuli.

    Parameters
    ----------
    topdir : _type_
        the directory under which to search for stimuli types
    mode : str, optional
        search for tones or clickes, by default "click"

    Returns
    -------
    _type_
        a list of directories that contain the stimuli type

    Raises
    ------
    ValueError
        bad mode definition
    """
    if mode in ["click", "clicks"]:
        search = "*-n.txt"
    elif mode in ["tone", "tones"]:
        search = "*kHz.txt"
    else:
        raise ValueError(f"Mode is not known: {mode:s}, must be 'click' or 'tone'")

    dirs = [
        tdir
        for tdir in Path(top_directory).glob("*")
        if Path(top_directory, tdir).is_dir()
        and not str(tdir.name).startswith("Recordings_cant_be_used")  # dirs to exclude
        and not str(tdir.name).startswith("Unsucessful recordings")
        and not str(tdir.name).find("Unsure") >= 0
    ]

    dirs_with_data = [d for d in dirs if len(list(Path(d).glob(search))) > 0]
    total_dirs = len(dirs)
    return dirs_with_data, total_dirs


def analyze_from_ABR_Datasets_main(
    datasets: Union[list, None] = None,
    configuration: dict = None,
    mode: str = "clicks",
    basedir: Union[Path, str, None] = None,
    coding=None,
    do_individual_plots: bool = True,
):

    for i, dsname in enumerate(datasets):
        if dsname not in list(ABR_Datasets.keys()):
            print("These are the known datasets: ")
            print("   ", list(ABR_Datasets.keys()))
            raise ValueError("The selected dataset %s not found in our list of known datasets")
        if mode not in ["tones", "clicks"]:
            raise ValueError("Second argument must be tones or clicks")

        print("Selected ABR Dataset: ", ABR_Datasets[dsname])
        top_directory = Path(basedir)  # Path(basedir, ABR_Datasets[dsname].directory)
        print("\nTop_directory", top_directory)

        dirs_with_data, total_dirs = find_stimuli(top_directory, mode=mode)

        if mode == "clicks":
            plot_info, IO_DF = do_clicks(
                dsname=dsname,
                configuration=configuration,
                ABR_Datasets=ABR_Datasets,
                coding=coding,
                group_by="strain",  # "group", "genotype", "subject", "sex"
                nplots=total_dirs,
                top_directory=top_directory,
                dirs=dirs_with_data,
                do_individual_plots=do_individual_plots,
            )

        elif mode == "tones":
            plot_info, IO_DF = do_tones(
                dsname,
                configuration=configuration,
                ABR_Datasets=ABR_Datasets,
                coding=coding,
                top_directory=top_directory,
                nplots=total_dirs,
                dirs=dirs_with_data,
            )
        else:
            raise ValueError(f"Mode is not known: {mode:s}")
    if plot_info is not None:
        mpl.show()


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


class Picker:
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


def analyze_from_excel(
    datasets=["Tessa_NF107"], agerange=(20, 150), datatype: str = "click", group: str = "WT"
):
    df = pd.read_excel("ABRs.xlsx", sheet_name="Sheet1")
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
                datatype=datatype,
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
    interp_thr, _ = fit_thresholds(x, y, baseline)
    row["interpolated_threshold"] = interp_thr
    row["maxabr"] = np.max(y)
    return row


def fit_thresholds(x, y, baseline):
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
    xs = np.linspace(1, 90, 91)
    ys = Hillmodel.eval(x=xs, params=result.params)

    # ithr = np.argwhere(np.array(y) > baseline)
    ithr = np.argwhere(np.array(ys) > baseline)
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
    return interp_thr, (xs, ys)


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
        print("subject: ", subject)
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
    cols = experiment["plot_colors"]
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


def plot_click_io_from_excel(
    datasets: list,
    groups: Union[list, None] = None,  # list of groups to analyze/plot
    agerange: Union[list, tuple] = (30, 105),
    coding: object = None,  # coding dataframe
    picking: bool = False,
    thrplot=False,
):
    df0 = pd.read_excel("ABRs.xlsx", sheet_name="Sheet1")  # get main database

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

    # colors = ["k", "r", "b", "c"]
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
    #
    # IO functions
    #
    print("p_df: ", p_df.group.unique())
    print("order: ", order)
    # Mean IO
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
        palette=Experiment["plot_colors"],
    )
    # Individual IO
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
    label = r"N1-P1 ($\mu V$)"
    ax[0].set_ylabel(f"{label:s}")
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
    # thresholds,
    if not picking:
        sns.swarmplot(
            data=grand_df,
            x="group",
            y="interpolated_threshold",
            hue="group",
            hue_order=order,
            palette=Experiment["plot_colors"],
            order=order,
            linewidth=0.5,
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
            palette=Experiment["plot_colors"],
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
            linewidth=0.5,
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
    #
    # ABR Amplitude
    #
    if not picking:
        sns.swarmplot(
            data=grand_df,
            x="group",
            y="maxabr",
            hue="group",
            hue_order=order,
            palette=Experiment["plot_colors"],
            linewidth=0.5,
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
            palette=Experiment["plot_colors"],
            saturation=0.25,
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
            linewidth=0.5,
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
    label = r"Maximum ABR ($\mu V$)"
    ax[2].set_ylabel(f"{label:s}")
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
            palette=Experiment["plot_colors"],
            linewidth=0.5,
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
    ax[0].legend(
        handles=handles,
        labels=[v for k, v in Experiment["new_xlabels"].items()],
        bbox_to_anchor=(0.59, 1.05),
    )
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

    # plot_click_io_from_excel(
    #     datasets=["Tessa_BNE"],
    #     groups=["B", "A", "AA", "AAA"],
    #     coding=get_coding_data(Experiment),
    #     picking=False,
    # )
    from ABR_Datasets import ABR_Datasets

    dataset = "Reggie_NIHL"
    experimentname = "GlyT2_NIHL"

    ds = ABR_Datasets[dataset]
    if ds.config is not None:
        configurations = get_configuration(ds.config)
        if experimentname not in configurations[0]:
            raise ValueError(f"Experiment {experimentname:s} not found in configuration file")

        configuration = configurations[1][experimentname]
        basedir = configuration["abrpath"]
        # print("basedir: ", basedir)
        coding = ABR_Reader.get_coding_data(configuration)
        # print(coding)
    else:
        basedir = "/Volumes/Pegasus_002/ManisLab_Data3/abr_data/"
        coding = None
        configuration = None

    analyze_from_ABR_Datasets_main(
        datasets=[dataset],  # datasets to analyze
        configuration=configuration,  # this is the configuration file
        mode="clicks",
        basedir=basedir,
        coding=coding,
        do_individual_plots=False,
    )
