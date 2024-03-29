# current 10 June 2017
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

"""

ABR_Datasets = {
    "NrCAMKO": {
        "dir": "Tessa/Coate-NrCAM-ABRs/KO",
        "invert": True,
        "clickselect": [["0849"], None, None, None, None, None, None, None],
        "toneselect": [["0901"], ["1446", "1505"], None, None, None, None, None, None],
        "term": "\r",
        "minlat": 2.2,
    },
    "NrCAMWT": {
        "dir": "Tessa/Coate-NrCAM-ABRs/WT",
        "invert": True,
        "term": "\r",
        "minlat": 2.2,
    },

    "CNTNAP2X": {"dir": "Tessa/CNTNAP2", "term": "\r", "minlat": 2.2, "invert": True},
    "CNTNAP2_Het": {
        "dir": "Tessa/CNTNAP2_Het",
        "term": "\r",
        "minlat": 2.2,
        "invert": True,
    },
    "CNTNAP2_HG": {
        "dir": "Tessa/CNTNAP2_Het_GP4.3",
        "term": "\r",
        "minlat": 2.2,
        "invert": True,
    },
    "CNTNAP2_KO": {
        "dir": "Tessa/CNTNAP2_KO",
        "term": "\r",
        "minlat": 2.2,
        "invert": True,
    },
    "CNTNAP2_WT": {
        "dir": "Tessa/CNTNAP2_WT",
        "term": "\r",
        "minlat": 2.2,
        "invert": True,
    },
    "GP43Norm": {
        "dir": "Tessa/GP4.3-Thy1-Normal",
        "term": "\r",
        "minlat": 2.2,
        "invert": True,
    },
    "Tessa_VGAT": {
        "dir": "Tessa/Tessa_ABR_data/VGAT",
        "term": "\r",
        "minlat": 2.4,
        "invert": True,
        "markers": {"VGATWT": ("s", "any"), "VGATFF": ("o", "any"), "Unsure": ("x", "any")},
    },
    "Tessa_VGATNIHL": {
        "dir": "Tessa/VGAT_NIHL",
        "term": "\r",
        "minlat": 2.4,
        "invert": True,
        "markers": {"VGATWT": ("s", "any"), "VGATFF": ("o", "any"), "Unsure": ("x", "any")},
    },
    "Tessa_FVB": {
        "dir": "Tessa/FVB",
        "term": "\r",
        "minlat": 2.4,
        "invert": False,
        "show_dots": False,
        "markers": {"FVB": ("s", "end"), "Unsure": ("x", "any")},
    },
    "Tessa_CBA": {
        "dir": "Tessa/Tessa_ABR_data/CBA",
        "term": "\r",
        "minlat": 2.4,
        "invert": True,
        "show_dots": False,
        "markers": {"M": ("s", "any"), "F": ("o", "any")},
    },
    "Tessa_CBA2": {
        "dir": "Tessa/CBA", 
        "term": "\r", 
        "minlat": 2.2, 
        "invert": True
        },
    "Tessa_CBA_NE": {
        "dir": "Tessa/CBA_NoiseExposed",
        "term": "\r",
        "minlat": 2.2,
        "invert": True,
    },
    "Tessa_NF107": {
        "dir": "Tessa/Tessa_ABR_data/NF107",
        "term": "\r",
        "minlat": 2.4,
        "invert": True,
        "show_dots": False,
        "markers": {"NF107": ("s", "end"), "NF107_Exposed": ("o", "end"), "Unsure": ("x", "any")},
    },
    "TessaNF107Ai32": {
        "dir": "Tessa/Tessa_ABR_data/NF107Ai32",
        "term": "\r",
        "minlat": 2.4,
        "invert": True,
        "show_dots": False,
        "markers": {"NF107": ("s", "end"), "NF107_Exposed": ("o", "end"), "Unsure": ("x", "any")},
    },
    "Tessa_NF107Ai32": {
        "dir": "Tessa/NF107Ai32",
        "term": "\r",
        "minlat": 2.4,
        "invert": True,
        "show_dots": False,
        "markers": {"NF107": ("s", "end"), "NF107_Exposed": ("o", "end"), "Unsure": ("x", "any")},
    },
    "Tessa_BNE": {
        "dir": "Tessa/Tessa_ABR_data/BNE",
        "term": "\r",
        "minlat": 2.4,
        "invert": True,
        "show_dots": False,
        "markers": {"NF107Ai32": ("s", "end"), "VGAT": ("o", "end"), "Unsure": ("x", "any")},
        "codefile": "Tessa/Tessa_ABR_data/NF107_BNE_Code.py",
    },
    "Yong": {"dir": "Yong's ABRs", "invert": True, "term": "\n", "minlat": 0.6},
    "Jun": {"dir": "JUN's ABRs", "term": "\r", "minlat": 2.2, "invert": False},
    "Ruili": {
        "dir": "Ruilis ABRs",
        "invert": True,
        "nameselect": "CBA",
        "term": "\n",
        "minlat": 0.6,
    },
    "RuiliCBAP40": {
        "dir": "RuiliABRData_2010-2015/CBA-P21-P40",
        "invert": True,
        "nameselect": "CBA",
        "term": "\n",
        "minlat": 0.6,
    },
    "RuiliCBAP20": {
        "dir": "RuiliABRData_2010-2015/CBA-P10-P20",
        "invert": True,
        "nameselect": "CBA",
        "term": "\n",
        "minlat": 0.6,
    },
    "Eveleen": {"dir": "Eveleen's ABRs", "term": "\r", "minlat": 2.2, "invert": False},
    "Amber": {"dir": "Amber_ABR_data", "term": "\r", "minlat": 2.2, "invert": True},
    "Reggie": {"dir": "Reggie_E", "term": "\r", "minlat": 2.2, "invert": True,
            "sample_freq": 50000.0, 
            "markers": {"WT": ("s", "end"), "KO": ("o", "end"), "Unsure": ("x", "any")},
            },
}
