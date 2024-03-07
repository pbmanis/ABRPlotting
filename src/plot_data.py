import pprint
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pylibrary.plotting.plothelpers as PH
from matplotlib import pyplot as mpl
from matplotlib.backends.backend_pdf import PdfPages

import src.abr_analyzer as abr_analyzer
import src.make_colormaps as make_colormaps
from src.abr_dataclasses import ABR_Data
from src.fit_thresholds import fit_thresholds
from src.get_configuration import get_configuration

pp = pprint.PrettyPrinter(indent=4)


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
            SD=2.5,
            # spec_bandpass=self.spec_bandpass,
        )
        if len(spls) > 2:  # this needs some protection from "stopped" runs
            i_spls = np.where(~np.isnan(spls))[0]  # find valid measurements
            i_ppio = np.where(~np.isnan(analyzer.ppio))[0]
            spls = [spls[i] for i in i_spls if i in i_ppio]
            analyzer.ppio = np.array([analyzer.ppio[i] for i in i_spls if i in i_ppio])
            if len(i_ppio) > 2:
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
        ylim = list(ax.get_ylim())
        if ylim[1] < 2.0:
            ylim[1] = 2.0
        ax.set_ylim(0, ylim[1])  # microvolts
        ax.set_ylabel(r"ABR ($\mu$V)")

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

        anymark = False
        if results["thr_spl"] > np.max(results["spls"]):
            thr_spl = np.max(results["spls"])
        else:
            thr_spl = results["thr_spl"]
        for j, spl in enumerate(results["spls"]):
            if j < len(results['spls'])-1:
                if results["spls"][j] <= thr_spl < results["spls"][j+1]:
                    anymark = True
                    ax.plot(
                        results["t"],
                        (0 * results["waves"][j] * scale_factor)+ thr_spl,
                        color=[0.5, 0.5, 0.5, 0.4],
                        linewidth=5,
                    )
            if stimtype == "clicks":
                c = spl_color_map[spl]
            elif stimtype == "tones":
                c = freq_color_map
            if spl == results["thr_spl"]:
                c = 'k'
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
        if not anymark:
            print("failed to mark spl threshold")
            print(results["spls"])
            print(results["thr_spl"], "\n")


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
            yticks=[0, 20, 40, 60, 80, 100, 120],
            yticks_str=["0", "20", "40", "60", "80", "100", "120"],
            y_minor=[10, 30, 50, 70, 90, 110],
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
        self.click_io_plot = None
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
            self.click_io_plot = PH.regular_grid(
                1,
                2,
                order="rowsfirst",
                figsize=(6, 3),
                margins={
                    "leftmargin": 0.1,
                    "rightmargin": 0.1,
                    "topmargin": 0.06,
                    "bottommargin": 0.12,
                },
            )
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
            group_color = "grey"
            group_symbol = "+"
            if group_by == "Group":
                if subject_data.group in group_colormap:
                    group_color = group_colormap[subject_data.group]
                    group_symbol = group_symbolmap[subject_data.group]
                grouping = subject_data.group
            elif group_by == "Strain":
                if subject_data.Strain in group_colormap:
                    group_color = group_colormap[subject_data.Strain]
                    group_symbol = group_symbolmap[subject_data.Strain]
                grouping = subject_data.Strain
            else:
                print("group_by is: ", group_by)
                exit()
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
                    color=group_color,
                    group=grouping,
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
                sizer = {
                    "A": {"pos": [0.08, 0.55, 0.08, 0.8]},
                    "B": {"pos": [0.67, 0.25, 0.08, 0.8]},
                }
                superimposed_io_plot = PH.arbitrary_grid(
                    sizer=sizer,
                    order="rowsfirst",
                    figsize=(8, 5.5),
                    margins={
                        "leftmargin": 0.1,
                        "rightmargin": 0.08,
                        "topmargin": 0.1,
                        "bottommargin": 0.12,
                    },
                )

            if subject_data.clickdata[run]["Subject"] is not None:
                label = f"{subject_data.clickdata[run]['Subject']:s} {subject_data.clickdata[run]['Strain']!s} ({subject_data.clickdata[run]['SPL']:5.1f})"
            # elif group_by == "group":
            #     label = f"{subject_data.clickdata[run]['group']:s}"
            # elif group_by == "Strain":
            #     label = f"{subject_data.clickdata[run]["Strain"]:s} ({subject_data.clickdata[run]['SPL']:5.1f})"
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
            ylims = list(superimposed_io_plot.axdict["A"].get_ylim())
            if ylims[1] < 2.0:
                ylims[1] = 2.0
            superimposed_io_plot.axdict["A"].set_ylim(ylims)
            superimposed_io_plot.axdict["A"].set_ylabel(r"ABR ($\mu V$)")

            IO_DF.append(
                [
                    subject_data.subject,
                    run,
                    result["spls"],
                    sf_cvt * result["Analyzer"].ppio,
                    result["thrs"][run],
                    subject_data.clickdata[run]["group"],
                    subject_data.clickdata[run]["Strain"],
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

        return IO_DF, superimposed_io_plot, self.click_io_plot

    def plotTones(
        self,
        subject_data,
        subject_number: int = 0,
        select: str = None,
        datadir: Union[Path, str] = None,
        configuration: dict = None,
        color_map: dict = None,
        group_by: str = "group",
        IOplot=None,
        PSDplot=None,
        superimposed_io_plot=None,
        show_y_label: bool = True,  # for the many-paneled plots
        show_x_label: bool = True,
        do_individual_plots: bool = True,
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
        self.tone_io_plot = None
        self.thrs = {}  # holds thresholds for this dataset, by frequency
        IO_DF = []  # build a dataframe of the IO funcitons from a list.
        spl_color_map, freq_colormap = make_colormaps.make_spl_freq_colormap()
        Analyzer = abr_analyzer.Analyzer(sample_frequency=subject_data.sample_freq)
        marker = "o"
        sf_cvt = 1e6
        if do_individual_plots:
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
                if do_individual_plots:
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

                if k < len(result["ppio"]):  # possible the full frequency run was not completed
                    IO_DF.append(
                        [
                            result["datatitle"],
                            run,
                            result["spls"],
                            fr,
                            sf_cvt * result["ppio"][k],
                            result["thrs"][run],
                            subject_data.group,
                            subject_data.Strain,
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
        if do_individual_plots:
            PH.cleanAxes(self.tone_io_plot.axdict["A"])
            legend = self.tone_io_plot.axdict["A"].legend(loc="upper left", fontsize=5)
        return IO_DF, superimposed_io_plot, self.tone_io_plot
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
