from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import pylibrary.plotting.plothelpers as PH
import pylibrary.tools.cprint as CP
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pprint
import src.abr_dataclasses as ABR_DC
from src.get_configuration import get_configuration
from src.abr_funcs import ABRFuncs
import src.make_colormaps as make_colormaps

ABRF = ABRFuncs()

pp = pprint.PrettyPrinter(indent=4)



def get_ABR_Dataset(datasetname, configuration):
    """get_ABR_Dataset Read the big abr file and merge with a coding sheet,
    from Configuration['coding_file'] if it exists. The coding file is used to
    then clean up. A few columns are renamed and one is dropped to make
    things clean.
    Note we use the coding sheet as the primary in the merge, because it
    is likely to be more complete for a given experiment than the big excel sheet.

    Parameters
    ----------
    datasetname : string
        The name of the dataset (experiment) in the ABRS.xlsx file.
    configuration: dict 
        configuration file for the experiment. Likely found in the project
        directory, under config/experiments.cfg

    Returns
    -------
    pandas data frame
        The merged dataframe
    """
    # (re-)read the big abr database file.
    # combine with the coding directory
    df_excel = pd.read_excel("ABRS.xlsx", sheet_name="Sheet1")
    # print(df_excel['DataDirectory'])
    print("ABR_Reader: get_ABR_dataset: dataset name: ", datasetname)
    # print("Configuration: ", configuration)
    df_excel = df_excel[df_excel.Dataset == datasetname]  # reduce to just this dataset
    df_excel.reindex()
    if "Subject" not in df_excel.columns and "Animal_identifier" in df_excel.columns:
        df_excel.rename({"Animal_identifier": "Subject"}, axis=1, inplace=True)
    df_excel.rename({"Treatment": "Group"}, axis=1, inplace=True)
    group_name = 'Control'
    if configuration is not None and configuration["coding_file"] is not None:
        group_name = configuration["coding_name"]
        print("     Configuration coding file: ", configuration["coding_file"])
        coding = get_coding_data(configuration)  # read the coding file
        if "Animal_ID" in coding.columns:
            coding.rename({"Animal_ID": "Subject"}, axis=1, inplace=True)
        if "animal_identifier" in coding.columns:
            coding.rename({"animal_identifier": "Subject"}, axis=1, inplace=True)
        coding = coding[coding[group_name].isin(configuration['group_map'].keys())]
        print("     get_ABR_dataset:  ", coding.Subject)
        df_excel_merge = df_excel.merge(
            coding, left_on="Subject", right_on="Subject", how="right"
        )  # set the groups based on the coding file
    else:
        df_excel_merge = df_excel
        df_excel_merge[group_name] = "Control"
    if group_name+'_x' in df_excel_merge.columns:
        df_excel_merge.drop(labels=[group_name+'_x'], axis=1, inplace=True)
    df_excel_merge.rename({group_name+'_y': "Group"}, axis=1, inplace=True)
    if 'DataDirectory_x' in df_excel_merge.columns:
        df_excel_merge.rename({'DataDirectory_x': 'DataDirectory'}, axis=1, inplace=True)
    # print(df_excel_merge.columns)
    # print(df_excel_merge.DataDirectory)
    df_excel_merge.to_excel(f"ABR_{datasetname:s}.xlsx", sheet_name="Sheet1")
    return df_excel_merge

# read the Coding file, which contains information about the ABR dates,
# group assignments, etc.
def get_coding_data(configuration):
    CP.cprint("c", "    ABR_Reader: Getting Coding Data")
    assert "databasepath" in configuration.keys()
    # print("     database Path: ", configuration["databasepath"])
    coding_file = Path(
        configuration["databasepath"], configuration["directory"], configuration["coding_file"]
    )
    if not coding_file.is_file():
        raise ValueError(f"Could not find coding file: {coding_file!s}")
    coding = pd.read_excel(coding_file, sheet_name=configuration["coding_sheet"])
    coding.rename(columns={"Animal_ID": "Subject"}, inplace=True)
    if "genotype" not in coding.columns:
        coding["genotype"] = ""
    return coding


class ABR_Reader:
    """
    Read an a single ABR data set from the matlab program
    Provides functions to plot the traces.
    """
    def __init__(self):
        pass

    def setup(
        self,
        datapath: Union[Path, str],
        configuration: dict,
        datasetname: str = "",
        mode: str = "clicks",
        info: object = ABR_DC.ABR_Data,
        abr_dataframe: pd.DataFrame = None,
        subject: str = "",
        strain: str = "",
        age: str = "",
        sex: str = "",
        group: str = "",
        genotype: str = "",
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

        self.error = False
        # Set some default parameters for the data and analyses
        self.datapath = Path(datapath)
        self.datasetname = datasetname
        self.abr_dataframe = abr_dataframe
        self.info = info
        self.subject = subject
        self.group = group
        self.Strain = strain
        if self.subject is None:
            CP.cprint("r", f"Subject field is empty, cannot select data")
            raise ValueError()
            # self.subject = "NoID"
            # datadir = self.abr_subject.DataDirectory.values[0]
            # if pd.isnull(datadir):
            # CP.cprint("r", f"Datadir is empty for subject {self.subject} in dataframe:\n{self.abr_subject!s} ")
            # return None

        # reduce incoming dataframe to the current subject
        else:
            CP.cprint("c", f"    ABR_Reader: Subject is: {self.subject:s}")
            self.abr_subject = self.abr_dataframe[self.abr_dataframe.Subject == self.subject]

            self.datadir = self.abr_subject.DataDirectory.values[0]
            if "Group" in self.abr_subject.columns:
                self.group = self.abr_subject.Group.values[0]
            else:
                self.group = "Unidentified"
            if "SPL" in self.abr_subject.columns:
                self.SPL = self.abr_subject.SPL.values[0]
            else:
                self.SPL = 0.0
            print("original strain: ", self.Strain)
            if "Strain" in self.abr_subject.columns:
                self.Strain = self.abr_subject.Strain.values[0]
            print("strain is: ", self.Strain)

            dset = self.abr_subject.Dataset.values[0]
            def set_dataset(row, datasetname):
                row.Dataset = datasetname
            if pd.isnull(dset):
                self.abr_subject.apply(set_dataset, axis=1, datasetname = self.datasetname) # self.abr_subject.loc["Dataset"] = self.datasetname
            print(f"    br_reader.setup: has Datadir:  {self.datadir!s}")
            if pd.isnull(self.datadir) or len(self.datadir) == 0:
                CP.cprint(
                    "r",
                    f"Datadir is empty for subject {self.subject} in dataframe:\n{self.abr_subject!s} ",
                )
                CP.cprint("r", f"    Likely cause is no ABR was found for this subject in the ABRs database")
                self.error = True
                return

        CP.cprint("c", f"    ABR_Reader: datasetname = {self.datasetname!s}")
        self.coding = None  # coding data frame, if it exists.
        if configuration is not None and configuration["coding_file"] is not None:
            coding = get_coding_data(configuration)
            self.coding = coding[coding.Subject == self.subject]

        if self.info.sample_freq is not None:
            self.sample_freq = self.info.sample_freq  # Hz
        else:
            self.sample_freq = 100000.0

        self.sample_rate = (
            1.0 / self.sample_freq
        )  # standard interpolated sample rate for matlab abr program: 10 microsecnds
        if self.info.spec_bandpass is not None:
            self.spec_bandpass = info.spec_bandpass
        else:
            self.spec_bandpass = [800.0, 1200.0]
        if info.showdots:  # turn off dot plotting.
            self.show_dots = info.showdots
        else:
            self.show_dots = True

        self.dev = 3.0  # should put this in the table
        self.hpf = 500.0
        self.lpf = 2500.0  # filter frequencies, Hz
        self.mode = mode
        self.info = info
        self.clickdata = {}
        self.tonemapdata = {}
        self.color_map, self.freq_colormap = make_colormaps.make_spl_freq_colormap()
        # if self.subject is None:
        #     CP.cprint("r", f"Subject field is empty, cannot select data")
        #     return None
        self.term = self.info.term
        self.minlat = self.info.minlat
        self.invert = (
            self.info.invert
        )  # data flip... depends on "active" lead connection to vertex (false) or ear (true).
        # select the subject from the dataframe

        self.characterize_abr_datafiles(self.datadir)

        # build color map where each SPL is a color (cycles over 12 levels)
        self.max_colors = 25
        bounds = np.linspace(0, 120, self.max_colors)  # 0 to 120 db inclusive, 5 db steps
        color_labels = np.unique(bounds)
        self.color_map = ABRF.makeColorMap(self.max_colors, color_labels)
        color_labels2 = list(range(self.max_colors))
        self.summary_color_map = ABRF.makeColorMap(
            self.max_colors, list(range(self.max_colors))
        )  # make a default map, but overwrite for true number of datasets
        self.psdIOPlot = False
        self.superIOLabels = []
        return

    def print(self):
        """print some subject information
        """
        print(f"Subject: {self.subject:s}")
        print(f"Group: {self.group:s}")
        print(f"Strain: {self.Strain!s}")
        print(f"Sample Frequency: {self.sample_freq:.2f} Hz")
        print("="*80)

    def characterize_abr_datafiles(self, directory):
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
        self.spls = ABRF.getSPLs(Path(self.datapath, directory))
        self.freqs = ABRF.getFreqs(Path(self.datapath, directory))

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
                "datadir": self.datadir,
                "Subject": self.subject,
                "group": self.group,
                "Strain": self.Strain,
                "SPL": self.SPL,
            }

        self.clickmaps = {}
        for i, s in enumerate(list(self.clicks.keys())):
            self.clickmaps[s] = {
                "stimtype": "click",
                "SPLs": self.spls[s],
                "Subject": self.subject,
                "datadir": self.datadir,
                "group": self.group,
                "Strain": self.Strain,
                "SPL": self.SPL,
            }
            # if self.mode == "clicks":
            #     if self.abr_dataframe is not None:
            #         mouseinfo = self.subject
            #         codefile.find_clickrun(s)
            #         if mouseinfo is not None:
            #             self.clickmaps[s]["subject"] = mouseinfo
            #     else:
            #         print("        ", self.clickmaps[s], "NO Mouse ID: ")

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
        CP.cprint("c", f"    ABR_Reader: adjustSelection: {select!s}")
        freqs = []
        if select is None or len(select) == 0:
            return select, freqs
        for i, s in enumerate(select):
            if s is None:
                continue
            if tone:
                for tm in self.tonemaps.keys():
                    for f in self.tonemaps[tm]["Freqs"]:
                        if f not in freqs:
                            freqs.append(f)
        return select, freqs


    def read_dataset(
        self,
        datapath: Union[Path, str],  # path to the data (.txt files are in this directory)
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
        pos_file = Path(datapath, fnamepos)
        neg_file = Path(datapath, fnameneg)
        if not pos_file.is_file():
            CP.cprint("r", f"    ABR_Reader.read_dataset: Did not find pos file: {pos_file!s}")
            return None, None
        if not neg_file.is_file():
            CP.cprint("r", f"    ABR_Reader.read_dataset: Did not find neg file: {pos_file!s}")
            return None, None
        CP.cprint("c", f"    ABR_Reader.read_dataset: Reading from: {pos_file!s}")
        if datatype == "click":
            spllist = self.clickmaps[fnamepos[:13]]["SPLs"]
        else:
            spllist = self.tonemaps[fnamepos[:13]]["SPLs"]
        cnames = [f"{spl:.1f}" for i, spl in enumerate(spllist)]
        posf = pd.io.parsers.read_csv(
            pos_file,
            sep=r"[\t ]+",
            lineterminator=r"[\r\n]+",  # lineterm,
            skip_blank_lines=True,
            header=None,
            names=cnames,
            engine="python",
        )
        negf = pd.io.parsers.read_csv(
            neg_file,
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


    def get_combineddata(
        self, datasetname, datadir, dataset, freq=None, lineterm="\r"
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
        waves = None
        tb = None
        if dataset["stimtype"] == "click":
            fnamepos = datasetname + "-p.txt"
            fnameneg = datasetname + "-n.txt"
            datap = Path(self.datapath, datadir)
            try:
                waves, tb = self.read_dataset(
                    datap,
                    datatype="click",
                    fnamepos=fnamepos,
                    fnameneg=fnameneg,
                    lineterm=lineterm,
                )
            except ValueError:
                CP.cprint("r", f"ABR_Reader.get_combineddata: Failed on datasetname: {datasetname!s}")
                CP.cprint("r", f"ABR_Reader.get_combineddata:      with dataset: {dataset!s}")
                raise ValueError("ABR_Reader.get_combineddata: Failed to read click data.")
        elif dataset["stimtype"] == "tonepip":
            fnamepos = datasetname + "-p-%.3f.txt" % freq
            fnameneg = datasetname + "-n-%.3f.txt" % freq
            datap = Path(self.datapath, datadir)
            waves, tb = self.read_dataset(
                datap, datatype="tonepip", fnamepos=fnamepos, fnameneg=fnameneg, lineterm=lineterm
            )
            return waves, tb
        else:
            CP.cprint("r", f"ABR_Reader.read_dataset: Unknown stimulus type: {dataset['stimtype']!s}")
            raise ValueError("Unknown stimulus type in dataset. Cannot process.")

        return waves, tb

    def getClickData(self, select: list, directory: Union[Path, str] = None, configuration:dict=None):
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
        select, _ = self.adjustSelection(select)
        # get data for clicks and plot all on one plot
        self.clickdata = {}
        # print("select: ", select)
        # print("Clickmap keys: ", self.clickmaps.keys())

        markerstyle, group = ABRF.getMarkerStyle(directory=directory, markers=self.info.markers)
        for i, s in enumerate(select):
            # print("Working on run: ", s)
            # print("clickmaps: ", self.clickmaps[s])
            waves, tb = self.get_combineddata(
                datasetname=s, datadir=self.datadir, dataset=self.clickmaps[s], lineterm=self.term
            )
            if waves is None:
                CP.cprint("r", f"ABR_Reader.getClickData: Malformed data set for run {s:s}. Continuing")
                continue
            waves = waves[::-1]  # reverse order to match spls

            spls = self.clickmaps[s]["SPLs"]  # get spls
            if configuration is not None and 'plot_symbols' in configuration.keys():
                print(self.clickmaps[s]['group'])
                markerstyle = configuration['plot_symbols'][self.clickmaps[s]['group']]
            self.clickdata[s] = {
                "waves": waves,
                "timebase": tb,
                "spls": spls,
                "marker": markerstyle,
                "group": self.clickmaps[s]["group"],
                "Strain": self.clickmaps[s]["Strain"],
                "Subject": self.clickmaps[s]["Subject"],
                "SPL": self.clickmaps[s]["SPL"],
            }
            # print("Populated clickdata: ", self.clickdata[s])

    def getToneData(self, select, directory: str = ""):
        """
        Gets the tone map data for the current selection The resulting data is
        held in a dictionary structured as {mapidentity: dict of frequencies}
        Where each dictoffrequencies holds a dict of {waves, time, spls and
        optional marker}

        select will be the path to the main holding directory

        """
        self.tonemapdata = {}
        # if not isinstance(select, list):
        #     select = [select]

        select, freqs = self.adjustSelection(select, tone=True)
        # convert select to make life easier
        # select list should have lists of strings ['0124', '0244'] or Nones...
        markerstyle, group = ABRF.getMarkerStyle(directory=directory, markers=self.info.markers)
        # iterate through the files in the directory, looking for tone maps
        for i, s in enumerate(select):
            freqs = []

            for f in self.tonemaps[s]["Freqs"]:
                if f not in freqs:
                    freqs.append(f)
            freqs.sort()
            if len(freqs) == 0:  # check of no tone pip ABR data in this directory
                continue
            # now we can build the tonemapdata
            self.tonemapdata[s] = {}
            for fr in self.tonemaps[s]["Freqs"]:
                waves, tb = self.get_combineddata(
                    datasetname=s,
                    datadir=self.datadir,
                    dataset=self.tonemaps[s],
                    freq=fr,
                    lineterm=self.term,
                )
                if waves is None:
                    CP.cprint("r"f"ABR_Reader.getToneData: Malformed data set for run {s:s}. Continuing")
                    continue

                spls = self.tonemaps[s]["SPLs"]
                self.tonemapdata[s][fr] = {
                    "waves": waves,
                    "timebase": tb,
                    "spls": spls,
                    "marker": markerstyle,
                    "group": group,
                    "Subject": self.subject,
                    "SPL": self.SPL,
                }
