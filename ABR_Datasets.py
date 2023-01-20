# This file holds a listing of the ABR datasets, divided by user, experiment(s), mouse type, etc. 
# This is the primary table used by plotABRs in ABRPlotting to generate plots with superimposed
# click IO functions, frequency-intensity maps, and "raw" traces. 
# The dictionaries also specify how to handle different groups (markers), trace inversion, filtering
# latency etc. 
# A special field is "codefile". This is the name of a python file that holds additional information
# about the specific experiments, linking the ABR datasets to specific experimental subjects,
# their treatment (noise exposure, etc) and physiology data collected in acq4.
#
# Original: 10 June 2017
# Most recent update: 12 August 2022
# This table should eventually be converted to an excel spreadsheet organized
# with a worksheet for each data set, and within each worksheet, each subject's information
# placed in a single row. Make the table pandas-readable.
# on the other hand, for now, it is useful as is.

"""
The keys are the names of the data sets (used on the command line)
The order in the nested dictionary is unimportant.
Each entry in the nested dictionary is structured as follows:

'dir': string - the path to the data
'invert' : boolena - if True, the polarity will be flipped
'clickselect': a list indicating the times of the useful protocol runs for the clicks
                if not defined, then all runs that are found are used.
                if runs need to be combined, then include them together (see toneselect in the NrCAMKO dataset)
'toneselect': a list of the tone protocol runs (see clickselect for format)
'term' : line terminator - may depend on how data has been passed through an editor. 
'minlat' : float. Minimum latency for an event (response)
'nameselect': if a dataset directory has more than one type of data, this helps to filter it.
"codefile = the name of a python file that has additional information about the animal codes 
    (IDs, etc.). 
"""
from src.ABR_dataclasses import ABR_Data

ABR_Datasets = {
    "NrCAMKO": ABR_Data(
        directory ="Tessa/Coate-NrCAM-ABRs/KO",
        invert = True,
        clickselect = [["0849"], None, None, None, None, None, None, None],
        toneselect = [["0901"], ["1446", "1505"], None, None, None, None, None, None],
        term = "\r",
        minlat = 2.2,
    ),
    "NrCAMWT": ABR_Data(
        directory ="Tessa/Coate-NrCAM-ABRs/WT",
        invert = True,
        term = "\r",
        minlat = 2.2,
    ),

    "CNTNAP2X": ABR_Data(
        directory ="Tessa/CNTNAP2", 
        term = "\r",
        minlat = 2.2, 
        invert = True,
    ),
    "CNTNAP2_Het": ABR_Data(
        directory ="Tessa/CNTNAP2_Het",
        term = "\r",
        minlat = 2.2,
        invert = True,
    ),
    "CNTNAP2_HG": ABR_Data(
        directory ="Tessa/CNTNAP2_Het_GP4.3",
        term = "\r",
        minlat = 2.2,
        invert = True,
    ),
    "CNTNAP2_KO": ABR_Data(
       directory ="Tessa/CNTNAP2_KO",
       term = "\r",
       minlat = 2.2,
       invert = True,
    ),
    "CNTNAP2_WT": ABR_Data(
        directory ="Tessa/CNTNAP2_WT",
        term = "\r",
        minlat = 2.2,
        invert = True,
    ),
    "GP43Norm ": ABR_Data(
        directory ="Tessa/GP4.3-Thy1-Normal",
        term = "\r",
        minlat = 2.2,
        invert = True,
    ),
    "Tessa_VGAT": ABR_Data(
        directory ="Tessa/Tessa_ABR_data/VGAT",
        term = "\r",
        minlat = 2.4,
        invert = True,
        markers = {"VGATWT":  ("s", "any"), "VGATFF":  ("o", "any"), "Unsure":  ("x", "any")},
    ),
    "Tessa_VGATNIHL": ABR_Data(
        directory ="Tessa/VGAT_NIHL",
        term = "\r",
        minlat = 2.4,
        invert = True,
        markers = {"VGATWT":  ("s", "any"), "VGATFF":  ("o", "any"), "Unsure":  ("x", "any")},
    ),
    "Tessa_FVB": ABR_Data(
        directory ="Tessa/FVB",
        term = "\r",
        minlat = 2.4,
        invert = False,
        showdots = False,
        markers = {"FVB":  ("s", "end"), "Unsure":  ("x", "any")},
    ),
    "Tessa_CBA": ABR_Data(
        directory ="Tessa/Tessa_ABR_data/CBA",
        term = "\r",
        minlat = 2.4,
        invert = True,
        showdots = False,
        markers = {"M":  ("s", "any"), "F": ("o", "any")},
    ),
    "Tessa_CBA2": ABR_Data(
        directory ="Tessa/CBA", 
        term = "\r", 
        minlat = 2.2, 
        invert = True,
    ),
    "Tessa_CBA_NE": ABR_Data(
        directory ="Tessa/CBA_NoiseExposed",
        term = "\r",
        minlat = 2.2,
        invert = True,
    ),
    "Tessa_NF107": ABR_Data(
        directory ="Tessa/Tessa_ABR_data/NF107",
        term = "\r",
        minlat = 2.4,
        invert = True,
        showdots = False,
        markers = {"NF107": ("s", "end"), "NF107_Exposed": ("o", "end"), "Unsure":  ("x", "any")},
    ),
    "TessaNF107Ai32": ABR_Data(
        directory ="Tessa/Tessa_ABR_data/NF107Ai32",
        term = "\r",
        minlat = 2.4,
        invert = True,
        showdots = False,
        markers = {"NF107": ("s", "end"), "NF107_Exposed": ("o", "end"), "Unsure": ("x", "any")},
    ),
    "Tessa_NF107Ai32": ABR_Data(
        directory ="Tessa/NF107Ai32",
        term = "\r",
        minlat = 2.4,
        invert = True,
        showdots = False,
        markers = {"NF107": ("s", "end"), "NF107_Exposed": ("o", "end"), "Unsure": ("x", "any")},
        codefile= ""
    ),
    "Tessa_BNE": ABR_Data(
        directory ="Tessa/Tessa_ABR_data/BNE",
        term = "\r",
        minlat = 2.4,
        invert = True,
        showdots = False,
        markers = {"NF107Ai32":  ("s", "end"), "VGAT": ("o", "end"), "Unsure":  ("x", "any")},
        codefile = "Tessa/Tessa_ABR_data/NF107_BNE_Code.py",
    ),
    "Yong": ABR_Data(
        directory ="Yong's ABRs",
        invert = True,
        term = "\n",
        minlat = 0.6,
    ),
    "Jun": ABR_Data(
        directory ="JUN's ABRs",
        term = "\r",
        minlat = 2.2,
        invert = False,
        ),
    "Ruili": ABR_Data(
        directory ="Ruilis ABRs",
        invert = True,
        nameselect = "CBA",
        term = "\n",
        minlat = 0.6,
    ),
    "RuiliCBAP40": ABR_Data(
        directory ="RuiliABRData_2010-2015/CBA-P21-P40",
        invert = True,
        nameselect = "CBA",
        term = "\n",
        minlat = 0.6,
    ),
    "RuiliCBAP20": ABR_Data(
        directory ="RuiliABRData_2010-2015/CBA-P10-P20",
        invert = True,
        nameselect = "CBA",
        term = "\n",
        minlat = 0.6,
    ),
    "Eveleen": ABR_Data(
        directory ="Eveleen's ABRs", 
        term = "\r", 
        minlat = 2.2, 
        invert = False,
        ),
    "Amber": ABR_Data(
        directory ="Amber_ABR_data", 
        term = "\r", 
        minlat = 2.2, 
        invert = True,
    ),

    "Reggie": ABR_Data(
        directory ="Reggie_E", 
        term = "\r",
        minlat = 2.2, 
        invert = True,
        sample_freq = 50000.0, 
        markers = {"WT": ("s", "end"), "KO": ("o", "end"), "Unsure": ("x", "any")},
    ),
}



# ABR_Datasets = {
#     "NrCAMKO = {
#         "directory ="Tessa/Coate-NrCAM-ABRs/KO",
#         "invert = True,
#         "clickselect = [["0849"], None, None, None, None, None, None, None],
#         "toneselect = [["0901"], ["1446", "1505"], None, None, None, None, None, None],
#         "term = "\r",
#         "minlat = 2.2,
#     },
#     "NrCAMWT = {
#         "directory ="Tessa/Coate-NrCAM-ABRs/WT",
#         "invert = True,
#         "term = "\r",
#         "minlat = 2.2,
#     },
#     "TessaCBA = {"directory ="Tessa/CBA", "term = "\r", "minlat = 2.2, "invert = True},
#     "TessaCBANE = {
#         "directory ="Tessa/CBA_NoiseExposed",
#         "term = "\r",
#         "minlat = 2.2,
#         "invert = True,
#     },
#     "CNTNAP2X = {"directory ="Tessa/CNTNAP2", "term = "\r", "minlat = 2.2, "invert = True},
#     "CNTNAP2Het = {
#         "directory ="Tessa/CNTNAP2_Het",
#         "term = "\r",
#         "minlat = 2.2,
#         "invert = True,
#     },
#     "CNTNAP2HG = {
#         "directory ="Tessa/CNTNAP2_Het_GP4.3",
#         "term = "\r",
#         "minlat = 2.2,
#         "invert = True,
#     },
#     "CNTNAP2KO = {
#         "directory ="Tessa/CNTNAP2_KO",
#         "term = "\r",
#         "minlat = 2.2,
#         "invert = True,
#     },
#     "CNTNAP2WT = {
#         "directory ="Tessa/CNTNAP2_WT",
#         "term = "\r",
#         "minlat = 2.2,
#         "invert = True,
#     },
#     "GP43Norm = {
#         "directory ="Tessa/GP4.3-Thy1-Normal",
#         "term = "\r",
#         "minlat = 2.2,
#         "invert = True,
#     },
#     "Yong = {"directory ="Yong's ABRs", "invert = True, "term = "\n", "minlat = 0.6},
#     "Jun = {"directory ="JUN's ABRs", "term = "\r", "minlat = 2.2, "invert = False},
#     "Ruili = {
#         "directory ="Ruilis ABRs",
#         "invert = True,
#         "nameselect = "CBA",
#         "term = "\n",
#         "minlat = 0.6,
#     },
#     "RuiliCBAP40 = {
#         "directory ="RuiliABRData_2010-2015/CBA-P21-P40",
#         "invert = True,
#         "nameselect = "CBA",
#         "term = "\n",
#         "minlat = 0.6,
#     },
#     "RuiliCBAP20 = {
#         "directory ="RuiliABRData_2010-2015/CBA-P10-P20",
#         "invert = True,
#         "nameselect = "CBA",
#         "term = "\n",
#         "minlat = 0.6,
#     },
#     "Eveleen = {"directory ="Eveleen's ABRs", "term = "\r", "minlat = 2.2, "invert = False},
#     "Amber = {"directory ="Amber_ABR_data", "term = "\r", "minlat = 2.2, "invert = True},
# }
