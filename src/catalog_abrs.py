import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

toppath = Path("/Volumes/Pegasus_002/ManisLab_Data3/abr_data")

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

    

def highlight_by_name(row):
    colors = {"Ruili": "#c5d7a5", #"darkseagreen",
            "Tessa_CNTNAP2_Het": "skyblue",
            "Tessa_CNTNAP2_Het_GP4.3": "skyblue",
            "Tessa_CNTNAP2_KO": "skyblue",
            "Tessa_CNTNAP2_WT": "skyblue",
            "Tessa_Coate-NrCAM-ABRs_KO": "skyblue",
            "Tessa_Coate-NrCAM-ABRs_WT": "skyblue",
            "Tessa_Myo10": "skyblue",
            "Tessa_NCAM": "skyblue",
            "Tessa_NCAM_GP4.3_ABR": "skyblue",
            "Tessa_GP4.3-Thy1-NoiseExposed": "skyblue",
            "Tessa_GP4.3-Thy1-Normal": "skyblue",
            "Xuying": "lightpink",
            "Yong's ABRs": "linen",
            "Ruilis ABRs": "yellow",
            "Ruili ABR data": "yellow",
            "Reggie_E": "thistle",
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
    if row.UserName in colors.keys():
        return [f"background-color: {colors[row.UserName]:s}" for s in range(len(row))]
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
        abr_f.append({"UserName": fp[5], "DataDirectory": subdir, "Runs": runs, "BasePath": str(Path(*fp[:4]))})
    df = pd.DataFrame(abr_f)
    print(df.head())
    print(df.index)
    make_excel(df, "ABRs.xlsx")


if __name__ == '__main__':
    make_excel_catalog()
