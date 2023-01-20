"""
Define the data classes we use in this program
Mouse_Info: The information about the mouse data class
ABR_data
"""
from dataclasses import dataclass, field
from typing import Union, List
from pathlib import Path

def defemptydict():
    return {}


def defemptylist():
    return []

"""Data class with all of the info for a particular animal.
This will not be populated unless the appropriate Code file has been read.

"""
@dataclass
class Mouse_Info:
    ID: str = ""
    Book: int=0
    Page: int=0
    CageCard: int=0
    DOB: str=""
    Sex: str=""
    Strain: str=""
    ExposeDate: str=""
    ABRDate: str=""
    RecordDate: str=""
    Group: str=""
    SPL: Union[float, None] = None
    ExposeAge: int=0 # or 39...
    RecordAge: int=0
    PostExposure: int=3 # or 14
    ABRToneFiles: list = field(default_factory=defemptylist)
    ABRClickFiles: list = field(default_factory=defemptylist)
    ABRPath: Union[str, Path]=""
    DataPath: Union[str, Path]=""
    Cells: list = field(default_factory=defemptylist)
    Quality: str="ok"


@dataclass
class ABR_Data:
    directory: Union[str, Path] = None # the path to the data
    datadirectory: Union[str, Path] = None
    invert: bool=False # if True, the polarity will be flipped
    clickselect: list = field(default_factory=defemptylist) #a list indicating the times of the useful protocol runs for the clicks
                # if empty, then all runs that are found are used.
                # if runs need to be combined, then include them together (see toneselect in the NrCAMKO dataset)
    markers: dict = field(default_factory=defemptydict)
    toneselect: list = field(default_factory=defemptylist) #a list of the tone protocol runs (see clickselect for format)
    term: str="" # line terminator
    minlat: float=0. #Minimum latency for an event (response)
    nameselect: str="" #if a dataset directory has more than one type of data, this helps to filter it.
    showdots: bool=False
    codefile: Union[str, None] = None  # name of file used to exctract "Code" information for subjects
    sample_freq: Union[float, None] = None # use non-default sample frequency
    spec_bandpass: Union[list, None] = None

