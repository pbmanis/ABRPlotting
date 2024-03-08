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
import pprint
import re
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

import ABR_Datasets  # just the dict describing the datasets
import src.abr_analyzer as abr_analyzer
import src.abr_funcs as abr_funcs
import src.abr_reader as ABR_Reader
import src.make_colormaps as make_colormaps
from src.abr_dataclasses import ABR_Data
from src.abr_dataclasses import plotinfo as PlotInfoDataclass
from src.fit_thresholds import fit_thresholds
from src.get_configuration import get_configuration
from src.plot_data import PData

pp = pprint.PrettyPrinter(indent=4)

ABRF = abr_funcs.ABRFuncs()

def plotClickThresholds(
    allthrs,
    df: pd.DataFrame = None,
    name: str = "",
    show_lines: bool = True,
    ax: object = None,
):
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
    if ax is None:
        raise ValueError("No axis provided for plotting")
    PH.nice_plot(ax, position=-0.05, direction="outward", ticklength=4)
    n_datasets = len(list(allthrs.keys()))
    # print("# of Datasets found to measure tone thresholds: ", n_datasets)
    c_map = ABRF.makeColorMap(n_datasets, list(allthrs.keys()))

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


def plotToneThresholds(
    allthrs: list,
    df: pd.DataFrame,
    # abr_dataset,
    name: str = "",
    show_lines: bool = True,
    configuration: dict = None,
    ax: object = None,
):
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
    if ax is None:
        fig = mpl.figure(figsize=(7, 5))
        # The first items are for padding and the second items are for the axes.
        # sizes are in inches.
        h = [Size.Fixed(0.8), Size.Fixed(5.0)]
        v = [Size.Fixed(0.8), Size.Fixed(3.75)]
        divider = Divider(fig, pos=(0, 0, 1, 1), horizontal=h, vertical=v, aspect=False)
        # The width and height of the rectangle are ignored.
        ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1))
        fig.suptitle(f"{name:s}\n{configuration['abr_plot_title']}", fontsize=12, ha="center")
    PH.nice_plot(ax, position=-0.05, direction="outward", ticklength=4)
    n_datasets = len(list(allthrs.keys()))
    print("# of Datasets found to measure tone thresholds: ", n_datasets)
    c_map = ABRF.makeColorMap(n_datasets, list(allthrs.keys()))

    def get_group(row):
        group = [row.subject]
        row.Group = group
        return

    def hztokhz(row):
        row.freq = row.freq / 1000.0
        return row

    gcolors = configuration["plot_colors"]
    for g in df.Group.unique():
        dg = df[df["Group"] == g]
        dg = dg.apply(hztokhz, axis=1)
        for ig in dg.index:
            ax.scatter(
                dg.loc[ig, "freq"] + np.random.uniform(-0.1, 0.1),
                dg.loc[ig, "thrs"] + np.random.uniform(-2, 2),
                s=10,
                marker="o",
                c=gcolors[g],
                alpha=0.5,
                edgecolors="k",
                linewidths=0.5,
                clip_on=False,
            )

        sns.lineplot(
            x="freq",
            y="thrs",
            #
            data=dg,
            color=gcolors[g],
            ax=ax,
            err_style="band",
            errorbar=("sd", 1),
            clip_on=False,
            label=g,
        )

    ax.set_xscale("log", nonpositive="clip", base=2)
    ax.set_xlim(1.5, 65)
    ax.set_ylim(20, 105)
    ax.set_xlabel("Frequency (kHz)", fontsize=11)
    ax.set_ylabel("Threshold (dBSPL)", fontsize=11)

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
        color the data by groups: "group", "Strain", "subject"
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
        IOdata, superimposed_io_plot, figure = PData().plotClicks(
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
        IOdata, superimposed_io_plot, figure = PData().plotTones(
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

    return IOdata, superimposed_io_plot, figure


def do_clicks(
    dsname: str,
    ABR_Datasets: dict = None,
    configuration: dict = None,
    coding: pd.DataFrame = None,
    group_by: str = "Group",
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
    subjects = sorted(list(set(abr_dataframe.Subject)))
    print("Subjects: ", subjects)
    abr_dataframe.rename(columns={"Strain_x": "Strain"}, inplace=True, errors="ignore")
    abr_dataframe.rename(columns={"Animal_identier": "Subject"}, inplace=True, errors="ignore")
    print("group by: ", group_by)
    match group_by:
        case "subjects":
            color_map = make_colormaps.make_subject_colormap(subjects)
        case "Group":
            color_map = make_colormaps.make_group_colormap(abr_dataframe.Group)
        case "Strain":
            color_map = make_colormaps.make_group_colormap(abr_dataframe.Strain)
        case _:
            raise ValueError(f"Group by {group_by:s} not recognized")

    superimposed_io_plot = None
    figures = []
    for subject_number, subject in enumerate(subjects):
        if coding is not None:
            Group = coding[coding.Subject == subject].Group.values[0]
            sex = (coding[coding.Subject == subject].sex.values[0],)
            age = coding[coding.Subject == subject].age.values[0]
        else:
            Group = "Control"
            sex = "U"
            age = "U"
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
            sex=sex,
            age=age,
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
        subject_data.print()

        IOdata, superimposed_io_plot, figure = populate_io_plot(
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
        figures.append(figure)
    
    
    clickIODF = pd.DataFrame(
        IO_DF, columns=["Subject", "run", "spl", "ppio", "thrs", "Group", "Strain"]
    )
    clickIODF["group_cat"] = clickIODF[group_by].astype("category")
    filled_circle = MarkerStyle("o", fillstyle="full")
    filled_square = MarkerStyle("s", fillstyle="full")

    if superimposed_io_plot is not None:
        # superimposed_io_plot.axdict['B'].plot(clickIODF["spl"], clickIODF["thrs"], "o", color="black")
        # sns.boxplot(
        #     x=group_by,
        #     y="thrs",
        #     hue="group_cat",
        #     order=configuration["plot_order"][group_by],
        #     data=clickIODF,
        #     ax=superimposed_io_plot.axdict["B"],
        #     whis=[5, 95],
        # )
        smarkers = configuration["plot_symbols"]
        # print("smarkers: ", smarkers)
        # print("strains: ", clickIODF["Strain"].unique())
        for s in clickIODF["Strain"].unique():
            print("s: ", s)
            if pd.isnull(s):
                sm = 'o'
            else:
                sm = smarkers[s]
            sns.stripplot(
                x=group_by,
                y="thrs",
                hue="group_cat",
                order=configuration["plot_order"][group_by],
                data=clickIODF[clickIODF["Strain"] == s],
                marker=sm,
                ax=superimposed_io_plot.axdict["B"],
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
        superimposed_io_plot.axdict["B"].legend(loc="upper left", fontsize=7)
    pdf_filename = Path("Individual_Click_Responses.pdf")
    pdf_file = PdfPages(pdf_filename)
    # put superimposed io iplot first in the file.
    superimposed_io_plot.figure_handle.savefig(pdf_file, format="pdf")
    for fig in figures:
        fig.figure_handle.savefig(pdf_file, format="pdf")
    pdf_file.close()

    # mpl.show()
    clickIODF.to_csv("ClickIO.csv")

    # spls = set(clickIODF["spl"])
    # clickIOWTlist = []
    # clickIOKOlist = []
    # subjs = set(clickIODF['subject'])
    # print(clickIODF.head())
    # for icol in subjs:
    #     if clickIODF.at['group'] == 'WT':
    #         clickIOWT['subject']
    # f, ax = mpl.subplots(1,1)
    # population_thrdata = plotClickThresholds(
    #     allthrs=allthrs, df=clickIODF, name="Click Thresholds",
    #     ax = ax,
    # )
    # mpl.show()
    return plot_info, clickIODF


def do_tones(
    dsname: str,
    ABR_Datasets: dict = None,
    configuration: dict = None,
    coding: pd.DataFrame = None,
    group_by: str = "Group",
    top_directory: Union[str, Path] = None,
    dirs: Union[list, None] = None,
    plot_info: object = None,
    do_individual_plots: bool = True,
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
    print("runs: ", abr_dataframe["Runs"])
    fofilename = Path(top_directory, "ToneSummary.pdf")
    allthrs = {}
    IO_DF = []
    print("dsname: ", dsname)

    subjects = set(abr_dataframe.Subject)
    print("Subjects: ", subjects)
    print(f"Processing {len(dirs):d} directories")
    print("dirs: ", dirs)
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
            strain=coding[coding.Subject == subject].Strain.values[0],
        )
        print("setup ok")
        if subject_data.error:
            CP.cprint("r", f"Subject errored: {subject:s}")
            continue  # skip this directory - probably none defined
        print("RUns: ", subject_data.abr_subject.Runs.values[0])
        runs = subject_data.abr_subject.Runs.values[0]
        if pd.isnull(runs) or len(runs) == 0:
            CP.cprint("r", "No Runs found")
            continue
        print("runs: ", runs)
        tonesel[subject] = [
            x.split(":")[1] for x in runs.split(",") if x.strip().startswith("tone:")
        ]
        nsel = 32
        if subject_number == 0:
            subject_data.summary_color_map = ABRF.makeColorMap(nsel, list(range(nsel)))

        subject_data.getToneData(select=tonesel[subject], directory=subject_data.datadir)
        print("Start populate io plot")
        IOdata, superimposed_io_plot, figure = populate_io_plot(
            subject_data,
            tonesel[subject],
            mode="tones",
            datadir=dsname,
            superimposed_io_plot=superimposed_io_plot,
            plot_index=plot_index,
            subject_number=subject_number,
            color_map=color_map,
            do_individual_plots=do_individual_plots,
        )
        if IOdata is None:
            print("IOdata is NONE")
            continue
        dirname = str(Path(subject_data.datadir).name)
        allthrs[dirname] = subject_data
        IO_DF.extend(IOdata)
        print("IODATA: ", IOdata)
    toneIODF = pd.DataFrame(
        IO_DF, columns=["Subject", "run", "spl", "freq", "ppio", "thrs", "Group", "Strain"]
    )
    toneIODF["group_cat"] = toneIODF["Group"].astype("category")
    filled_circle = MarkerStyle("o", fillstyle="full")
    filled_square = MarkerStyle("s", fillstyle="full")

    population_thrdata = plotToneThresholds(
        allthrs=allthrs, df=toneIODF, name="Tone Thresholds", configuration=configuration
    )
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
                group_by="Group",  # "group", "genotype", "subject", "sex"
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
                group_by="Group",
                coding=coding,
                top_directory=top_directory,
                nplots=total_dirs,
                dirs=dirs_with_data,
                do_individual_plots=do_individual_plots,
            )
        else:
            raise ValueError(f"Mode is not known: {mode:s}")
    if plot_info is not None:
        mpl.show()


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

    # dataset = "Reggie_CBA_Age"
    # experimentname = "CBA_Age"
    dataset = "Tessa_BNE"  #
    experimentname = "NF107Ai32_NIHL"
    # dataset = "Reggie_NIHL"
    # experimentname = "NIHL"
    # dataset = "Tessa_VGAT"
    # experimentname = "VGAT"

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
        do_individual_plots=True,
    )
