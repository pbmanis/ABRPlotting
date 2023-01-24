ABR Program WORKFLOW
====================

The scripts in the src directory provide tools for organizing ABR data, performing analysis,
comparing datasets, and plotting ABRs and IO functions. 

The general workflow is as follows:

1. Put the ABR data into folders. Each folder describes a "dataset",
such as data collected from a particular strain, before or after manipulations, or data from
a particular experiment. The names of the directories in the folder should be in the format
mm-dd-yyyy_Pnn_[MF]k_animalid_strain_manipulation. Do not use hyphens except in the date portion
of the field. Underscores may be replace with spaces. This format is specific because the programs
will parse it and generate tables to drive further analysis, and to get information for 
selection criteria Only a few manipulation fields are current recognized.

Do not put subfolders here. If you have multiple datasets that will eventually be compared, collect
them as folders under the top level abr_data folder.

2. Generate a "catalog" of the data. The program "src/catalog_abrs.py" will generate this catalog,
which is an Excel file held in the main directory as "ABRs.xlsx". Because this file will be modified
on the next run, do not try to edit it - all edits will be lost. 

3. Run "plotABRs.py" with the selected datasets passed as a list to the analysis function. The data will
be analyzed and stored in separate excel tables in each of the ABR data folders. Again, these are write-only 
folders and will be overwritten on the next analysis. The program will also plot summaries that include
the IO functions for each subject, the waveforms for each subject, and an overlay IO plot for every subject.

4. Run "plotABRs.py" with the selected datasets passed as a list to the plotting function. The analyzed
data will be plotted across groups (data folders), with selected criteria. 

If you rename any of the data folders (such as to include the actual animal ID, or the condition), then
you need to start at the beginning by regenerating the catalog of the data, and then re-run the analysis.

