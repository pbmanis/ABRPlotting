"""Read the "ABR_Datasets.py" directory and convert it to an excel worksheet.

"""
from pathlib import Path
import pandas as pd
import ABR_Datasets as AD

columns = ["dir",
    "invert",
    "clickselect", #: [["0849"], None, None, None, None, None, None, None],
    "toneselect", # : [["0901"], ["1446", "1505"], None, None, None, None, None, None],
    "term",
    "minlat",
    "showdots", # boolean
    "markers", #  {"VGATWT": ("s", "any"), "VGATFF": ("o", "any"), "Unsure": ("x", "any")},
    "codefile", # "codefile": "Tessa/Tessa_ABR_data/NF107_BNE_Code.py",
    "samplefreq", # 50000. for example
]

def convert_ABR_Datasets():
    """Convert the datasets. Make a single worksheet, with a new row for every dataset

    """

if __name__ == "__main__":
    convert_ABR_Datasets()
