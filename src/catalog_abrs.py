import datetime
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

toppath = Path("/Volumes/Pegasus_002/ManisLab_Data3/abr_data")

@dataclass
class metadata:
    date: str = None
    strain: str = None
    age: str = None
    sex: str = None
    animal_identifier: str = None
    treatment: str = None
    genotype: str = None
    cross: str = None


"""Try to find all the duplicated ABR datasets
"""




def find_dupes():
    abrf = toppath.glob("*/*")
    abrfiles = {}
    abr_traces = []
    abr_f = []
    for f in abrf:
        if str(f).find("Tessa") > 0: 
            subdir = f.stem
            af = f.glob("*/*.txt")
            for localf in af:
                # print(localf)
                abr_traces.append(localf)
    fx = []
    dupes = []
    for i, f in enumerate(abr_traces):
        if f.name in fx:
            dupes.append(f)
        else:
            fx.append(f.name)
    ftop = []
    n_true = 0
    for i, f in enumerate(abr_traces):
        if str(f).find("unfiled")> 0:
            continue
        if str(f).find("Tessa_ABR_data") > 0:
            continue
        fp = f.parts
        if fp[-2] in ftop:
            continue
        for j, d in enumerate(dupes):
            if str(d).find("Tessa_ABR_data") > 0:
                continue
            if f.name == d.name and f != d:
                print("\n")
                print(fp[-2])
                print(f"    {i:5d}:  {str(f):s}")
                print(f"    {j:5d}:  {str(d):s}")
                ftop.append(fp[-2])
                n_true += 1
    print("\nN True Dupes: ", n_true)

def get_runs(f):
    """Get the ABR stimulus runs

    Args:
        f (path): Path to data dirextory

    Returns:
        string: a string listing the runs that were done.
    """
    dsets1 = f.glob("*-SPL.txt")
    dsets2 = f.glob("*-kHz.txt")
    dn1 = [str(d.name)[:13] for d in dsets1]
    dn2 = [str(d.name)[:13] for d in dsets2]
    n = set(dn1).difference(set(dn2))  # clicks don't have kHz
    prots = ''
    if len(n) > 0:
        for p in n:
            prots += f"click:{p:s}, "
    for p in dn2:
        prots += f"tone:{p:s}, "
    return prots

def parse_date(text, f):
    refile = None
    ds = re.split("[-/ ]", text)
    if len(ds[0]) == 1:
        ds[0] = f"{int(ds[0]):02d}" # make sure it has leading 0
        refile = ds # adjust file name so that this is true!
        text = '-'.join(ds)
    for fmt in ('%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y', "%Y%m%d", "%m-%d-%Y", "%m-%d-%y"):
        try:
            return datetime.datetime.strptime(text, fmt), refile
        except ValueError:
                pass
    try:
        # print("Use File Date", f)
        # make a datetime from the file timestamp
        ctime = datetime.datetime.fromtimestamp(Path(f).stat().st_ctime)
        text = datetime.datetime.strftime(ctime, "%Y-%m-%d")
        return(ctime, refile)
    except:
        print("Cannot match date time for directory: ", text, f)
        pass
    raise ValueError(f'no valid date format found for: {text:s}, {str(f):s}')

def get_metadata(dirname, filepath=None):
    """Given the directory name, try to parse it to fill out some
    metadata fields for the table. 
    Different users used different fields for the names, so we have a lot to parse
    Typical:
    10-04-2019_ABR_P63_F2_CNTNAP2KO
    from this we get the date, the age, the sex and animal # for the day, and the genotype.
    Separators might be spaces or underscores (most commonly), or dashes (less common)

    Args:
        f (Path or str): directory name to parse
    """
    md = metadata()
    # before splitting string on underscores and spaces, change some designators
    # used by some personnel to a standard format
    dirname = dirname.lower()
    dirname = dirname[:8] + dirname[8:].replace('-', '_') # replace hyphens but only after date portion of name
    dirname = dirname.replace("vgatexposed", "vgat_exposed")
    dirname = dirname.replace("vgat", "vgat_wt")
    dirname = dirname.replace("cntnap2homo", "cntnap2_homo")
    dirname = dirname.replace("cntnap2ko", "cntnap2_ko")
    dirname = dirname.replace("cntnap2wt", "cntnap2_wt")
    dirname = dirname.replace("cntnap2kohomo", "cntnap2_ko_homo")
    dirname = dirname.replace("ai32exposed", "ai32_exposed")
    dirname = dirname.replace("noise exposed", "noiseexposed")
    dirname = dirname.replace("noise_exposed", "noiseexposed")
    dirname = dirname.replace("sham_exposed", "shamexposed")
    fstr = re.split("[_ ]+", str(dirname))
    datestr = fstr[0]
    ts, refile = parse_date(datestr, filepath)
    md.date = datetime.datetime.strftime(ts, "%Y.%m.%d")
    # look for age
    r_age = re.compile("^P[0-9]{1,3}", re.IGNORECASE)
    r_sex = re.compile("^[MF]{1}[0-9]*$", re.IGNORECASE)
    r_id = re.compile("^[a-z]{1,3}[0-9]{1,3}$", re.IGNORECASE) # two letters followed by 1 or 2 numbers is usually the ID
    r_exposure = re.compile(r"(?P<dur>[0-9]{1})?(?P<week>wk{0,1})?(?P<un>un{0,1}|noise{0,1}|sham{0,1})?(?P<exp>Exposed{1}|exposed{1}|exposure{1}|control{1})", re.IGNORECASE)
    strains = ['VGAT', 'CBA', 'FVB', 'UBE3A', 'NCAM', 'NrCAM','NF107Ai32', 'NF107', 
        'DDY',  'DBA', 'BK2' 'Math1cre', 'Ai32',
        'CNTNAP2', 'GP43Thy1', 'B2S']
    strains = [s.lower() for s in strains]
    genotypes = ['KO', 'WT', 'Homo', 'Het', 'FF']
    genotypes = [g.lower() for g in genotypes]
    for i, fs in enumerate(fstr):
        fs = fs.strip()
        print("checking fs: ", fs)
        m = r_age.match(fs)
        if m is not None:
            md.age = int(m[0].lower().strip("p"))
            continue
        s = r_sex.match(fs)
        if s is not None:
            md.sex = s[0][0].upper()
            continue
        a = r_id.match(fs)
        if a is not None:
            md.animal_identifier = a[0].upper()
            continue
        exp = r_exposure.match(fs)
        if exp is not None:
            mg = exp.groups()
            # print(exp.group('exp'))
            treat = None
            if exp.group('exp') is not None:
                if exp.group('un') is not None:
                    if exp.group('un') == 'noise':
                        treat = "NoiseExposed"
                    elif exp.group('un') in ['un', 'sham']:
                        treat = "UnExposed"
                else: # descriptor was not included, so key on the actual exposure key:
                    if exp.group('exp') == 'control':
                        treat = "UnExposed"
                    elif exp.group('exp').startswith("expos"):
                        treat = "NoiseExposure"
                    else:
                        raise ValueError(f"Noise exposure key not recognized: {exp.group('exp'):s}")
            else:
                if exp.group('wk') is not None or exp.group('dur') is not None or exp.group('exp') is not None:
                    raise ValueError(f"Unable to parse potential exposure string: {fs:s}")

            md.treatment = treat

        for s in strains:
            if fs.startswith(s) and len(fs) == len(s):
                md.strain = s.upper()
            elif fs.startswith(s):
                md.strain = s.upper()
                gt = fs[len(s):]
                print("gt: ", s, gt, filepath)
                if gt in genotypes:
                    md.genotype = gt.upper()
                else:
                    md.cross = gt.upper() # not a genotype, probably a cross.
                break
        if md.genotype is not None:
            for g in genotypes:
                if fs == g:
                    md.genotype = g.upper()
                    break

        # print(md, dirname)

    return md

def highlight_by_name(row):
    colors = {"Ruili": "#c5d7a5", #"darkseagreen",
            "Ruilis ABRs": "yellow",
            "Ruili ABR data": "yellow",
            "Yong's ABRs": "linen",
            "Reggie_E": "thistle",

            "Tessa_BNE": "lavender",
            "Tessa_CBA": "turquoise",
            "Tessa_FVB": "slategrey",
            "Tessa_CBA_NoiseExposed": "turquoise",
            "Tessa_CNTNAP2": "skyblue",
            "Tessa_CNTNAP2_Het": "skyblue",
            "Tessa_CNTNAP2_Het_GP4.3": "skyblue",
            "Tessa_CNTNAP2_KO": "skyblue",
            "Tessa_CNTNAP2_WT": "skyblue",
            "Tessa_Coate-NrCAM-ABRs_KO": "aqua",
            "Tessa_Coate-NrCAM-ABRs_WT": "aqua",
            "Tessa_Myo10": "powderblue",
            "Tessa_NCAM": "steelblue",
            "Tessa_NCAM_GP4.3_ABR": "steelblue",
            "Tessa_GP4.3-Thy1-NoiseExposed": "lightsteelblue",
            "Tessa_GP4.3-Thy1-Normal": "lightsteelblue",
            "Tessa_NF107": "aliceblue",
            "Tessa_NF107Ai32": "aliceblue",
            "Tessa_VGAT": "lightcyan",
            "Tessa_VGATNIHL": "lightcyan",

            "Xuying": "lightpink",
            "Kasten ABR data": "sienna",
            "JUN's ABRs": "saddlebrown",
            "heather abrs": "blue",
            "Eveleen's ABRs": "lightslategray",
            "Christiann_ABR_data": "darkcyan",
            "Charlie Askew ABR data": "darkgoldenrod",
            "Amber_ABR_data": "lightpink",
            "Xuying ABR data": "lime",
            "ABRstransfer": "mediumorchid",
            "Transgenic-ABRs": "cyan",
    }
    if row.DataSet in colors.keys():
        return [f"background-color: {colors[row.DataSet]:s}" for s in range(len(row))]
    else:
        return [f"background-color: white" for s in range(len(row))]
        
def make_excel(df:object, outfile:Path):
    """cleanup: reorganize columns in spreadsheet, set column widths
    set row colors by a name list

    Args:
        df: object
            Pandas dataframe object
        excelsheet (_type_): _description_
    """
    outfile = Path(outfile)
    if outfile.suffix != '.xlsx':
        outfile = outfile.with_suffix('.xlsx')

    writer = pd.ExcelWriter(outfile, engine='xlsxwriter')

    df.to_excel(writer, sheet_name = "Sheet1")
    # df = organize_columns(df)
    workbook = writer.book
    worksheet = writer.sheets["Sheet1"]
    fdot3 = workbook.add_format({'num_format': '####0.000'})
    fdot3.set_text_wrap()
    wrapit = workbook.add_format()
    wrapit.set_text_wrap()

    df.to_excel(writer, sheet_name = "Sheet1")

    resultno = []

    for i, column in enumerate(df):
        # print('column: ', column)
        if column in resultno:
            writer.sheets['Sheet1'].set_column(first_col=i+1, last_col=i+1,  cell_format=fdot3)
        if column not in ['notes', 'description', 'OriginalTable', 'FI_Curve']:
            coltxt = df[column].astype(str)
            coltxt = coltxt.map(str.rstrip)
            maxcol = coltxt.map(len).max()
            column_width = np.max([maxcol, len(column)]) # make sure the title fits
            if column_width > 100:
                column_width = 100 # but also no super long ones
            #column_width = max(df_new[column].astype(str).map(len).max(), len(column))
        else:
            column_width = 25
        if column_width < 8:
            column_width = 8
        if column in resultno:
            writer.sheets['Sheet1'].set_column(first_col=i+1, last_col=i+1, cell_format=fdot3, width=column_width) # column_dimensions[str(column.title())].width = column_width
            # print(f"formatted {column:s} with {str(fdot3):s}")
        else:
            writer.sheets['Sheet1'].set_column(first_col=i+1, last_col=i+1, width=column_width, cell_format=wrapit) # column_dimensions[str(column.title())].width = column_width

    df = df.style.apply(highlight_by_name, axis=1)
    df.to_excel(writer, sheet_name = "Sheet1")
    writer.close()
    
def make_excel_catalog():
    abrf = toppath.glob("*/*")


    df = pd.DataFrame()
    abr_f = []
    for f in abrf:
        fp = f.parts
        subdir = f.stem
        # print("<",f.suffix.strip(),">")
        if f.suffix.strip() in ['.pdf', '.pzf', '.pzfx', '.txt', '.xls', '.xlsx', '.csv', '.pxp', '.cal', '.m'] or f.name == '.DS_Store':
            continue
        if f.is_dir():
            runs = get_runs(f)
        else:
            runs = np.nan
        mdata = get_metadata(dirname=subdir, filepath=f) # get some metadata from the subdir name: multiple parses... 
        abr_f.append({"DataSet": fp[5], "Date": mdata.date, "Age": mdata.age, "Strain": mdata.strain, "Sex": mdata.sex,
            "animal identifier": mdata.animal_identifier, "treatment": mdata.treatment,"genotype": mdata.genotype, "cross": mdata.cross,
            "DataDirectory": subdir, "Runs": runs, "BasePath": str(Path(*fp[:5]))})
    df = pd.DataFrame(abr_f)
    print(df.head())
    print(df.index)
    make_excel(df, "ABRs.xlsx")


if __name__ == '__main__':
    make_excel_catalog()
