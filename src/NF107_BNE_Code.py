from dataclasses import dataclass, field
from pathlib import Path
from typing import Union
import numpy as np
import re
import datetime
from pylibrary.tools import cprint
# import src.plotABRs as PA
from ABR_dataclasses import Mouse_Info
CP = cprint.cprint
"""

Data pulled from "set_expt_paths.py" in mrk_nf107 library.
A   : NIHL, measured 14 d later, 106 or 109 dB SPL
B   : control (unexposed or sham)
AA  : NIHL, measured 14 d later, 115 dB SPL 
AAA : NIHL, measured ~ 3d later, 115 dB SPL
A_O   : Same as A, but "outlier"
B_O   : Same as B, but "outliter 

Birthdates, IDs and Cage cards are cross referenced to the Archived Colony Record
from DCM. 
"""

re_control = re.compile(r"(?P<control>control)")
re_noise_exposed = re.compile(r"(?P<noiseexp>NoiseExposed)")
re_sham_exposed = re.compile(r"(?P<shamexp>ShamExposed)")
re_un_exposed = re.compile(r"(?P<shamexp>Unexposed)")
re_sex = re.compile(r"(?P<sex>\_[MF]+[\d]{0,3}\_)")   # typically, "F" or "M1" , could be "F123"
re_age = re.compile(r"(?P<age>_P[\d]{1,3})")
re_date = re.compile("(?P<date>[\d]{2}-[\d]{2}-[\d]{4})")

def get_age(name):
    age = re_age.search(name)
    if age is not None:
        P_age = age.groups()[0][1:]
        day_age = int(P_age[1:])
    else:
        P_age = 'U' 
        day_age = np.nan
    return day_age

def mk_datetime(sdate:str):
    """Convert a string in form of "yyyy.mm.dd" to a datetime.

    Parameters
    ----------
    sdate : str
        date 

    Returns
    -------
    _type_
        datetime
    """
    if sdate is None or sdate in ["None", "none", ""]:
        dt = [1970, 1, 1]
    else:
        dt = [int(d) for d in sdate.split('.')]
    dt = datetime.date(dt[0], dt[1], dt[2])
    return dt


"""
The original coding dictionary holds information about the animal ID,
    exposure and recording dates, dob, SPL, cross reference to lab notebooks,
    the abr directory, the tone, and click file names, maybe the cells that are good, etc.
    
"""
basepath = Path('/Volumes/Pegasus_002/ManisLab_Data3/abr_data/Tessa/NF107Ai32')
    
coding_NF107_nihl = {
        "Animal1": Mouse_Info(ID="Animal1", ExposeDate="2018.06.01", ABRDate="2018.06.15", 
        RecordDate="2018.06.19", DOB="2018.03.31", Group="A", Sex="M", Strain="NF107Ai32",
        SPL=109, ExposeAge=52, Book=2, Page=25,
        ABRPath="06-15-2018_ABR_P52_M1_NF107Ai32_Exposed", 
        ABRToneFiles=["20180615-1112", "20180615-1132", "20180615-1209"],
        ABRClickFiles=["20180615-1223"], 
        Cells="", Quality="ok",
        ),

    "Animal2": Mouse_Info(ID="Animal2", ExposeDate="2018.06.01", ABRDate="2018.06.15", 
        RecordDate="2018.06.20", DOB="2018.03.31", Group="A", Sex="M", Strain="NF107Ai32",
        SPL=109, Book=2, Page=25,
        ABRPath="06-15-2018_ABR_P52_M2_NF107Ai32_Exposed",
        ABRToneFiles=["20180615-1233", "20180615-1255", "20180615-1232"], ABRClickFiles=["20180615-1103"],
        Cells="", Quality="ok",
        ), #["Animal2", "A", "ok"],

    "Animal3": Mouse_Info(ID="Animal3", ExposeDate="2018.06.01", ABRDate="2018.06.15", 
        RecordDate = "2018.06.22", DOB="2018.03.31", Group="A", Sex="M", Strain="NF107Ai32",
        SPL=109, Book=2, Page=25,
        ABRPath="06-15-2018_ABR_P52_M3_NF107Ai32_Exposed",
        ABRToneFiles=["20180615-1354", "20180615-1414", "20180615-1451"], ABRClickFiles=["20180615-1345"],
        Cells="", Quality="ok",
        ), #["Animal3", "A", "ok"],

    "MS3": Mouse_Info(ID="MS3", ExposeDate="2018.07.03", RecordDate="2018.07.17", 
        ABRDate="2018.07.16", Group="B_O", DOB="2018.03.31", Sex="M", Strain="NF107Ai32",
        SPL=109, Book=2, Page=33,
        ABRPath="07-16-2018_ABR_P107_M4_NF107Ai32",
        ABRClickFiles=["20180716-1441"], 
        ABRToneFiles = ["20180716-1511", "20170716-1554"],
        Cells="", Quality="ok",
        ), #["MS3", "D", "outlier"],  # B

    "MS8": Mouse_Info(ID="MS8", ExposeDate="2018.07.03", RecordDate="2018.07.20", 
        ABRDate="2018.07.16", Group="A", DOB="2018.03.31",  Sex="M", Strain="NF107Ai32",
        SPL=109, Book=2, Page=32,
        ABRPath="07-16-2018_ABR_P107_M3_NF107Ai32",
        ABRClickFiles=["20180716-1303"], 
        ABRToneFiles = ["20180716-1313", "20170716-1336", "20170716-1419"], 
        Cells="", Quality="ok"), # ["MS8", "C", "outlier"],  # A

    "MS4": Mouse_Info(ID="MS4", ExposeDate="2018.07.03", RecordDate="2018.07.23",
        ABRDate="2018.07.16", Group="A_O", DOB="2018.03.31",  Sex="M", Strain="NF107Ai32",
        SPL=109, Book=2, Page=33,
        ABRPath="07-16-2018_ABR_P107_M4_NF107Ai32",
        ABRClickFiles=["20180716-1441"], 
        ABRToneFiles = ["20180716-1511", "20170716-1554"],
        Cells="", Quality="ok"), # ["MS4", "C", "outlier"],  # A

    "MS1": Mouse_Info(ID="MS1", ExposeDate="2018.07.03", RecordDate="2018.07.25",
        ABRDate="2018.07.16", Group="B_O", DOB="2018.03.31",  Sex="M", Strain="NF107Ai32",
        SPL=109, Book=2, Page=32,
        ABRPath="07-16-2018_ABR_P107_M2_NF107Ai32",
        ABRClickFiles=["20180716-1143"],
        ABRToneFiles=["20180716-1152", "20180716-1211", "20180716-1248"],
         Cells="", Quality="ok"), #["MS1", "D", "outlier"],  # B
         
    "NI4": Mouse_Info(ID="NI4", ExposeDate="2018.07.03", RecordDate="2018.07.27",
        ABRDate="2018.07.25", Group="B", DOB="2018.05.27",  Sex="M", Strain="NF107Ai32",
        SPL=0, Book=2, Page=34,
        ABRPath="07-25-2018_ABR_P59_M2_NF107Ai32",
        ABRClickFiles=["20180725-1513"],
        ABRToneFiles = ["20180725-1522", "20180725-1542", "20180725-1619"], 
        Cells="", Quality="ok"), #["NI4", "B", "ok"],

    "NI3": Mouse_Info(ID="NI3", ExposeDate="2018.07.03", RecordDate="2018.07.30",
        ABRDate="2018.07.17", Group="B", DOB="2018.05.27",  Sex="M", Strain="NF107Ai32",
        SPL=0, Book=2, Page=33,
        ABRPath="07-17-2018_ABR_P51_M1_NF107Ai32",
        ABRClickFiles=["20180717-1023"],
        ABRToneFiles = ["20180717-1035", "20180717-1058", "20180717-1135"],
         Cells="", Quality="ok"), # ["NI3", "B", "ok"],

    "NI1": Mouse_Info(ID="NI1", ExposeDate="2018.07.03", RecordDate="2018.08.01",
        ABRDate="2018.07.17",Group="A", DOB="2018.05.27", Sex="M", Strain="NF107Ai32",
        SPL=109, Book=2, Page=34,
        ABRPath="07-17-2018_ABR_P51_M2_NF107Ai32",
        ABRClickFiles=["20180717-1149"],
        ABRToneFiles = ["20180717-1159", "20180717-1219", "20180717-1304"],
        Cells="", Quality="ok"),  # ["NI1", "A", "ok"],

    "NI2": Mouse_Info(ID="NI2", ExposeDate="2018.07.03", RecordDate="2018.08.03",
        ABRDate="2018.07.25", Group="A", DOB="2018.05.27",  Sex="M", Strain="NF107Ai32",
        SPL=109, Book=2, Page=34,
        ABRPath="07-25-2018_ABR_P59_M1_NF107Ai32",
        ABRClickFiles=["20180725-0906"],
        ABRToneFiles = ["20180725-0917", "20180725-0942", "20180725-1021"], 
        Cells="", Quality="ok"), # ["NI2", "A", "ok"],

#     # The next ones do not appear in the book or in the NF107AI32 folder
#     # "unclipped": Mouse_Info(ID="unclipped", RecordDate="2018.09.12", Group="A", ABRClickFiles=["2018."], Cells="", Quality="ok"), # ["unclipped", "A", "ok"],
#     # "clipped": Mouse_Info(ID="clipped", RecordDate="2018.09.19", Group="A", ABRClickFiles=["2018."], Cells="", Quality="ok"), # ["clipped", "A", "ok"],
#     # ---------------------------added after 11/28/2018
# #    The first for "animal#" do not appear to be part of this dataset either
#     # "animal1": Mouse_Info(ID="animal1", RecordDate="2018.10.30", Group="b", ABRClickFiles=["2018."], Cells="", Quality="ok"), # ["animal1", "B", "ok"],
#     # "animal2": Mouse_Info(ID="animal2", RecordDate="2018.10.31", Group="A", ABRClickFiles=["2018."], Cells="", Quality="ok"), # ["animal2", "A", "ok"],
#     # "animal3": Mouse_Info(ID="animal3", RecordDate="2018.11.06", Group="A", ABRClickFiles=["2018."], Cells="", Quality="ok"), # ["animal3", "A", "ok"],
#     # "animal4": Mouse_Info(ID="animal4", RecordDate="2018.11.08", Group="B", ABRClickFiles=["2018."], Cells="", Quality="ok"), # ["animal4", "B", "ok"],

    "OE1": Mouse_Info(ID="OE1", ExposeDate="2018.12.04", RecordDate="2018.12.27",
        ABRDate="2018.12.21", Group="A", DOB="2018.11.05",  Sex="M", Strain="NF107Ai32", CageCard=2858876,
        SPL=109, Book=2, Page=63,
        ABRPath="12-21-2018_ABR_P49_M3_NF107Ai32",
        ABRClickFiles=["2018."],
        Cells="", Quality="ok"), # ["OE1", "A", "ok"],
    
    "OE4": Mouse_Info(ID="OE4", ExposeDate="2018.12.04", RecordDate="2018.12.27",
        ABRDate="2018.12.21", Group="A", DOB="2018.11.05",  Sex="M", Strain="NF107Ai32", CageCard=2858876,
        SPL=109, Book=2, Page=63,
        ABRPath="12-21-2018_ABR_P49_M3_NF107Ai32",
        ABRClickFiles=["2018."],
        Cells="", Quality="ok"), # ["OE4", "A", "ok"],

    "OE3": Mouse_Info(ID="OE3", ExposeDate="2018.12.04", RecordDate="2019.01.02",
        ABRDate="2018.12.21", Group="B", DOB="2018.11.05", Sex="M", Strain="NF107Ai32", CageCard=2858876,
        SPL=0, Book=2, Page=63,
        ABRClickFiles=["2018."],
        Cells="", Quality="ok"), # ["OE3", "B", "ok"],

    "OE2": Mouse_Info(ID="OE2", ExposeDate="2018.12.04", RecordDate="2019.01.04",
        ABRDate="2018.12.21", Group="B", DOB="2018.11.05", Sex="M", Strain="NF107Ai32", CageCard=2858876,
        SPL=0, Book=2, Page=63,
        ABRPath="12-21-2018_ABR_P49_M1_NF107Ai32",
        ABRClickFiles=["2018."],
        Cells="", Quality="ok"), # ["OE2", "B", "ok"],

"""
----------------------------- HERE ARE the BNE series ------------------------
BNE 1-13 "Accidentally connected DACOUT1 to SA to Crown. Page 71, Book 2. TFR
Actual value is not attenuated via PA5.
Measured exposure is: 103.8 dB SPL for noise (Attenuator was not in line)
Note that some mice in this range were not noise-exposed.
pbm 8/10/2022.
"""

    "BNE1": Mouse_Info(ID="BNE1", ExposeDate="2018.12.13", RecordDate="2019.01.23",
        ABRDate="2018.12.27", Group="AA", DOB="2018.11.13", CageCard=2862336, Sex="M", Strain="NF107Ai32",
        SPL=103.8, Book=2, Page=65,  # Book 2 page 71 Correction to output
        ABRPath="12-27-2018_ABR_P49_BNE1_NF107Ai32",
        ABRClickFiles=["20181212-0816"], ABRToneFiles=["20181227-0825","20181227-0830"],
        Cells="", Quality="ok",
        ), 

    "BNE2": Mouse_Info(ID="BNE2", ExposeDate="2018.12.13", RecordDate="2019.01.16",
        ABRDate="2018.12.27", Group="AA", DOB="2018.11.13", CageCard=2862336, Sex="M", Strain="NF107Ai32",
        SPL=103.8, Book=2, Page=65,  # Book 2 page 71 Correction to output
        ABRPath="12-27-2018_ABR_P49_BNE2_NF107Ai32",
        ABRClickFiles=["12272018-0920"], ABRToneFiles=["20181227-0929","20181227-0936"],
        Cells="", Quality="ok"),    

    "BNE3": Mouse_Info(ID="BNE3", ExposeDate="2018.12.13", RecordDate="2019.01.18",
        ABRDate="2018.12.27", Group="AA", DOB="2018.11.13", CageCard=2862336, Sex="M", Strain="NF107Ai32",
        SPL=103.8, Book=2, Page=65,# Book 2 page 71 Correction to output
        ABRPath="12-27-2018_ABR_P49_BNE3_NF107Ai32",
        ABRClickFiles=["12272018-1014"], ABRToneFiles=["20181227-1023", "20181227-1028"],
        Cells="", Quality="ok"),  

    "BNE4": Mouse_Info(ID="BNE4", ExposeDate="2019.12.20", RecordDate="2019.01.11",
        ABRDate="2018.12.27", Group="AA",  DOB="2018.11.13", CageCard=2862336, Sex="M",  Strain="NF107Ai32",# DOB="2018.11.05", Sex="M", Where did this date come from?
        SPL=103.8, Book=2, Page=65, # Book 2 page 71 Correction to output
        ABRPath="12-27-2018_ABR_P49_BNE4_NF107Ai32",
        ABRClickFiles=["12272018-1109"], ABRToneFiles=["20181227-1118", "20181227-1124"],
        Cells="", Quality="ok"), 
    
    "BNE5": Mouse_Info(ID="BNE5", ExposeDate="2018.12.13", RecordDate="2019.01.14",
        ABRDate="2018.12.27", Group="B", DOB="2018.11.13", CageCard=2862336, Sex="M", Strain="NF107Ai32",
        SPL=0, Book=2, Page=65,  # according to p65, BNE5 was not exposed.
        ABRPath="12-27-2018_ABR_P49_M1_NF107Ai32",
        ABRClickFiles=["12272018-0646"], ABRToneFiles=["20181227-0656", "20181227-0718"],
        Cells="", Quality="ok"), # ["BNE5", "B", "ok"],  # note reads: Animal #BNE5 (1-5?)
                   
# #BNE6 was not exposed ::: P67 book2
    "BNE6": Mouse_Info(ID="BNE6", ExposeDate="2018.12.13", RecordDate="",
        ABRDate="2018.01.02", Group="B", DOB="2018.11.13", CageCard=2862336, Sex="M", Strain="NF107Ai32",
        SPL=0, Book=2, Page=65,  # according to p65, BNE5 was not exposed.
        ABRPath="01-02-2019_ABR_P50_BNE6_NF107Ai32",
        ABRClickFiles=["20190102-1230"], ABRToneFiles=["20190102-1249"],
        Cells="", Quality="ok"), # ["BNE5", "B", "ok"],  # note reads: Animal #BNE5 (1-5?)
     
    "BNE7": Mouse_Info(ID="BNE7", ExposeDate="2019.12.20", RecordDate="2019.01.24",
        ABRDate="2019.01.02", Group="AA", DOB="2018.11.13", CageCard=2862336, Sex="M", Strain="NF107Ai32",
        SPL=103.8, Book=2, Page=67, # Book 2 page 71 Correction to output
        ABRPath="01-02-2019_ABR_P50_BNE7_NF107Ai32",
        ABRClickFiles=["20190102-0914"], ABRToneFiles=["20190102-0923"],
        Cells="", Quality="ok"), 

     "BNE8": Mouse_Info(ID="BNE8", ExposeDate="2019.12.20", RecordDate="2019.01.09",
        ABRDate="2019.01.02", Group="AA", DOB="2018.11.13", CageCard=2862336, Sex="M", Strain="NF107Ai32",
        SPL=103.8, Book=2, Page=67, # Book 2 page 71 Correction to output
        ABRPath="01-02-2019_ABR_P50_BNE8_NF107Ai32",
        ABRClickFiles=["20190102-1006"], ABRToneFiles=["20190102-1016", "20190102-1022"],
        Cells="", Quality="ok"), 

    "BNE9": Mouse_Info(ID="BNE9", ExposeDate="2019.12.20", RecordDate="2019.01.09",
        ABRDate="2019.01.02", Group="AA", DOB="2018.11.13", CageCard=2862336, Sex="M", Strain="NF107Ai32",
        SPL=103.8, Book=2, Page=67, # Book 2 page 71 Correction to output
        ABRPath="01-02-2019_ABR_P50_BNE9_NF107Ai32",
        ABRClickFiles=["20190102-1111"], ABRToneFiles=["20190102-1126", "20190102-1134"],
        Cells="", Quality="ok"), 

    "BNE10": Mouse_Info(ID="BNE10", ExposeDate="2019.12.20", RecordDate="2019.01.09",
        ABRDate="2019.01.02", Group="AA", DOB="2018.11.13", CageCard=2862336, Sex="M", Strain="NF107Ai32",
        SPL=103.8, Book=2, Page=67, # Book 2 page 71 Correction to output
        ABRPath="01-02-2019_ABR_P50_BNE10_NF107Ai32",
        ABRClickFiles=["20190102-1349"], ABRToneFiles=["20190102-1358", "20190102-1403"],
        Cells="", Quality="ok"), 

    "BNE11": Mouse_Info(ID="BNE11", ExposeDate="2019.12.20", RecordDate="2019.01.09",
        ABRDate="", Group="B", DOB="2018.11.13", CageCard=28623367, Sex="M", Strain="NF107Ai32",
        SPL=0, Book=2, Page=67, # Not exposed
        ABRPath="",  # no ABRs noted
        ABRClickFiles=[""],
        Cells="", Quality="ok"), 
    
    "BNE12": Mouse_Info(ID="BNE12", ExposeDate="2019.12.20", RecordDate="2019.01.09",
        ABRDate="", Group="B", DOB="2018.11.13", CageCard=28623367, Sex="M", Strain="NF107Ai32",
        SPL=0, Book=2, Page=67, # Not exposed
        ABRPath="", # no ABRs noted
        ABRClickFiles=[""],
        Cells="", Quality="ok"), 
    
    "BNE13": Mouse_Info(ID="BNE13", ExposeDate="2019.12.20", RecordDate="2019.01.09",
        ABRDate="2019.01.17", Group="B", DOB="2018.11.13", CageCard=28623367, Sex="M", Strain="NF107Ai32",
        SPL=0, Book=2, Page=67, # Not exposed
        ABRPath="01-17-2019_ABR_P65_BNE13_NF107Ai32",
        ABRClickFiles=["20190117-1420"], ABRToneFiles=["20190118-1430", "20190117-1449"],
        Cells="", Quality="ok"), 

    "BNE14": Mouse_Info(ID="BNE14", ExposeDate="2019.01.03", RecordDate="2019.01.09",
        ABRDate="2019.01.17", Group="AA", DOB="2018.12.03", CageCard=28623367, Sex="M", Strain="NF107Ai32",
        SPL=0, Book=2, Page=67, # Not exposed
        ABRPath="01-17-2019_ABR_P65_BNE13_NF107Ai32",
        ABRClickFiles=["20190117-1420"], ABRToneFiles=["20190118-1430", "20190117-1449"],
        Cells="", Quality="ok"), 
    
    "BNE15": Mouse_Info(ID="BNE15", ExposeDate="2019.01.03", RecordDate="2019.01.09",
        ABRDate="2019.01.17", Group="AA", DOB="2018.12.03", CageCard=28623367, Sex="M", Strain="NF107Ai32",
        SPL=0, Book=2, Page=67, # Not exposed
        ABRPath="01-17-2019_ABR_P65_BNE13_NF107Ai32",
        ABRClickFiles=["20190117-1420"], ABRToneFiles=["20190118-1430", "20190117-1449"],
        Cells="", Quality="ok"), 

    "BNE16": Mouse_Info(ID="BNE15", ExposeDate="2019.01.03", RecordDate="2019.01.09",
        ABRDate="2019.01.17", Group="B", DOB="2018.12.03", CageCard=28623367, Sex="M", Strain="NF107Ai32",
        SPL=0, Book=2, Page=67, # Not exposed
        ABRPath="01-17-2019_ABR_P65_BNE13_NF107Ai32",
        ABRClickFiles=["20190117-1420"], ABRToneFiles=["20190118-1430", "20190117-1449"],
        Cells="", Quality="ok",
        ), 

# BOOKMARK for data entry 10 Aug2022

#     "BNE7?": Mouse_Info(ID="BNE7?", RecordDate="2019.01.30",
#         Group="A", ABRClickFiles=["2018."], Cells="", Quality="ok"), #  [ "BNE7", "A", "ok",],  # ???? Animal #BNE17/18/20 (need verification from Tessa) No discernable marks on mouse ears/toes.

    # "BNE18": Mouse_Info(ID="BNE18", ExposeDate="RecordDate="2019.02.01",
    #     ABRDate"2019.01.24",
    #     SPL=96.5, Book=2, Page=72,  # 13.5 dB Attn
    #     Group="A", ABRClickFiles=["2018."], Cells="", Quality="ok"), # ["BNE18", "A", "ok"],  # Clear upper? left notch

#     "BNE24": Mouse_Info(ID="BNE24", RecordDate="2019.02.15", 
#         Group="A", ABRClickFiles=["2018."], Cells="", Quality="ok"), # ["BNE24", "A", "ok"],  # left upper ear notch

#     "BNE21": Mouse_Info(ID="BNE21", RecordDate="2019.02.18", 
#         Group="A", ABRClickFiles=["2018."], Cells="", Quality="ok"), # ["BNE21", "A", "ok"],  # both ears notched

#     "BNE23": Mouse_Info(ID="BNE23", RecordDate="2019.02.19",
#         Group="B", ABRClickFiles=["2018."], Cells="", Quality="ok"), # ["BNE23", "B", "ok"],  # lower right ear notched

#     "BNE22": Mouse_Info(ID="BNE22", RecordDate="2019.02.20",
#         Group="B", ABRClickFiles=["2018."], Cells="", Quality="ok"), # ["BNE22", "B", "ok"],  # lower left ear notched

#     "BNE27": Mouse_Info(ID="BNE27", RecordDate="2019.02.22",
#         Group="AA", ABRClickFiles=["2018."], Cells="", Quality="ok"), #["BNE27", "AA", "ok"],  # P24 At exposure
#     # '2019.02.25': Mouse_Info(ID="NI3", RecordDate="2018.08.01", Group="A", ABRClickFiles=["2018."], Cells="", Quality="ok"), # ['BNE25', 'AA', 'no'], # no data from this animal  P24 at
#     # exposure

#     "BNE32": Mouse_Info(ID="BNE32", RecordDate="2019.03.04",
#         Group="AA", ABRClickFiles=["2018."], Cells="", Quality="ok"), # ["BNE32", "AA", "ok"],

#     "BNE31": Mouse_Info(ID="BNE31", RecordDate="2019.03.05",
#         Group="AA", ABRClickFiles=["2018."], Cells="", Quality="ok"), # ["BNE31", "AA", "ok"],

#     "BNE30": Mouse_Info(ID="BNE30", RecordDate="2019.03.06",
#         Group="AA", ABRClickFiles=["2018."], Cells="", Quality="ok"), # ["BNE30", "AA", "ok"],

#     "BNE2Y": Mouse_Info(ID="BNE2Y", RecordDate="2019.03.01",
#         Group="AAA", ABRClickFiles=["2018."], Cells="", Quality="ok"), # ["BNE2Y", "AAA", "ok"],  # noise exposed high level; 3 day wait

#     "BNE3Y": Mouse_Info(ID="BNE3Y", RecordDate="2019.03.15",
#         Group="AAA", ABRClickFiles=["2018."], Cells="", Quality="ok"), # ["BND3Y", "AAA", "ok"],
#     "?????": Mouse_Info(ID="?????", RecordDate="2019.03.18",
#         Group="AAA", ABRClickFiles=["2018."], Cells="", Quality="ok"), # ["???", "AAA", "ok"],
#     "BXXXX": Mouse_Info(ID="BXXXX", RecordDate="2019.04.15",
#         Group="A", ABRClickFiles=["2018."], Cells="", Quality="ok"), # ["BXXX", ""],

#     "BNE102": Mouse_Info(ID="BNE102", RecordDate="2019.03.18",
#         Group="AAA", ABRClickFiles=["2018."], Cells="", Quality="ok"),

#     "BNE105": Mouse_Info(ID="BNE105", RecordDate="2019.05.10",
#         Group="AAA", ABRClickFiles=["2018."], Cells="", Quality="ok"),

#     "BNE106": Mouse_INFO(ID="BNE106", RecordDate="2019.05.13",
#         Group="AAA", ABRClickFiles=["2018."], Cells="", Quality="ok"),
    
    # "2019.05.13": ["BNE106", "AAA", "ok"],
    # "2019.05.17": ["BNE107", "AA", "ok"], # check - is this the mouse exposed on 5/3 2019?
    # "2019.05.31": ["BNE39", "AA", "ok"],
    # "2019.06.03": ["BNE108", "AAA", "ok"], # exposed on 5/31
    # "2019.06.04": ["BNE40", "AA", "ok"],
    # "2019.06.06": ["BNE41", "AA", "ok"],
    # "2019.06.17": ["BNE42", "AA", "ok"],
    # "2019.06.18": ["BNE43", "AA", "ok"],
    # "2019.07.29": ["BNE109", "AAA", "ok"],
    # "2019.07.30": ["BNE110", "AAA", "ok"],
    # "2019.08.30": ["BNE112", ]


}

"""
From TFR Notebook Noise exposures:
Book 2:
P63: NF107Ai32 @ P31, OE1, OE4
P65: NF107AI32: BNE1-4 noise exposure.  BNE5: unexposed. 
P65: 12/14/18 OA1, OA2, OA8 ABR
P66: 12/14/18 OA4, AO3, OA9 ABR
P67: 12/20/18 BNE 6, 7, 8, 9, 10 : exposed    BNE 11, 12, 13 Not exposed.
P67: 12/20/18: OE2, OE4 ABR
P68: 12/21/18: OE1, OE3 ABR
P68: 12/27/18: BNE5, BNE1 ABR
P69: 12/27/18: BNE2, BNE3, BNE4, BNE7 ABR
P70: 1/2/19: BNE8, BNE9, BNE6, BNE10
P71: 1/3/19 BNE 14-16 noise exposure
P71: 1/4/19: Note BNE 1-13, OA1-4 (VGAT) 103.4 dB SPL instead of 110.
P72: 1/9/19: BNE10 ABR
P72: 1/9/19: BNE 17, 18, 19, 20 Noise exposure. 
P72: 1/14/19: OE4 ABR
P74: 1/17/19: BNE14, 15, 16, 13 ABR
P75: 1/24/19: BNE17, BNE18, BNE19, BNE20 ABR (BNE19 euthanized)
P76: 1/25/19: BNE21-24 : Noise exposure (brown coats exposed, 2 m; black coates unexposed)
P76: 1/25/19: BNE 25-29 3m brown coat exposed; white coat unexposed.
P76: 2/11/19: BNE21, 22, 23, 24 ABR
P77: 2/12/19: BNE25, 26, 27, 28, 29 ABR
P78: 2/15/19: 4 mice exposed; no label info.
P80: 3/1/19: BNE 30, 31, 32, 100, 33, 34, 35 ABR tests. First one says P45, exposed 2/15/19.
P82: 3/11/19: BNE36, 37 ABR
P83: 3/13/19: BNE 37, 38 ABR
P84: 3/15/19: BNE101 ABR
P85: 4/12/19: NF107 exposed ? 2 mice ? 
P86: 4/15/19: ABR on 4/12 mice ? 
P87: 5/3/19: NF107 P40 exposed No label
P87: 5/6/19: NF107 ABR (from 5/3 mouse?)
P88: 5/7/19: NF107 noise exposre P46 male no label
P89: 5/10/19: NF107 ABR (3d exp)
P90: 5/13/19: NF107 ABR "PL"
P90: 5/17/19: NF107 ABR, noise exposed on 5/14/19
P90: 5/17/19: NF107 Noise expsoure 115
P92: 5/21/19: BNE39, BNE40 ABR (2 week
P93: 5/21/19: BNE108 noise exposure (115)
P93: 6/3/19: ABR BNE108
P93: P32 NF107 noise exposure 2 m, 115dB no label
P94: 6/17/19 : BNE ? NF107 ABR males, exposed 6/3/19.
------------ continued: ----------------------


"""

def verify():
    """verify the data set is somewhat complete

    Raises:
        ValueError: if a required field is missing
    """    

    for subject in list(coding_NF107_nihl.keys()):
        topdir = coding_NF107_nihl[subject].ABRPath 
        if topdir == "":
            continue
        # print("basepath: ", basepath)
        # print("topdir: ", topdir)
        # abrp = Path(basepath, topdir)
        # for cf in coding_NF107_nihl[subject].ABRClickFiles:
        #     print(list(Path(abrp).glob(f"{cf:s}*.txt")))
        #     PA.do_clicks()
        # for cf in coding_NF107_nihl[subject].ABRToneFiles:
        #     print(list(Path(abrp).glob(f"{cf:s}*.txt")))
        try:
            dob = mk_datetime(coding_NF107_nihl[subject].DOB)
        except: 
            print(f"DOB not set for subject - {subject:s}")
            raise ValueError()
        try:
            d_exp = mk_datetime(coding_NF107_nihl[subject].ExposeDate)
        except:
            print(f"Exposure date not set for subject - {subject:s}")
            raise ValueError()
        try:
            d_abr = mk_datetime(coding_NF107_nihl[subject].ABRDate)
        except:
            print(f"ABR Recording date not set for subject - {subject:s}")
            raise ValueError()
        try:
            d_rec = mk_datetime(coding_NF107_nihl[subject].RecordDate)
        except:
            print(f"Recording date not set for subject - {subject:s}")
            raise ValueError()

        md = get_age(topdir)
        print(f"Subject: {subject:<12s}")
        print(f"   Exposure age: {(d_exp-dob).days:4d}  ABR age: {(d_abr-dob).days:4d}  Age at recording: {(d_rec-dob).days:4d}  Time since exposure: {(d_rec-d_exp).days:4d}")
        if (d_abr-dob).days == md:
            CP('g', f"   Age in ABR filename:  {md:d}")
        else:
            CP('r', f"   Age in ABR filename does not match:  {md:d} vs {(d_abr-dob).days:4d}")
        
        # verify we can find and read the abr file
        import src.plotABRs as pABR
        from ABR_Datasets import ABR_Datasets 
        dsname = "somename"
        P = pABR.ABR(Path(topdir, [coding_NF107_nihl[subject].ABRPath]), "clicks", info=ABR_Datasets[dsname], datasetname = dsname)
    


if __name__ == "__main__":
    verify()
    