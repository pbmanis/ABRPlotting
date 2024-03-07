import matplotlib.pyplot as mpl
import pandas as pd
import numpy as np
from pylibrary.tools import cprint as CP
import seaborn as sns

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
