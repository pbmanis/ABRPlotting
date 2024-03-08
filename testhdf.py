

fn = "/Volumes/Pegasus_002/ManisLab_Data3/abr_data/Reggie_CBA_Age/CBA_F_p18/20230124-1238.h5"
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as mpl
import numpy as np

import src.abr_reader as ABRR
import src.abr_funcs as ABRF

Reader = ABRR.ABR_Reader()
Reader.setup(datapath=Path(fn).parent, configuration=None)
Reader.characterize_abr_datafiles(configuration=None)
w, t = Reader.read_dataset(
        datapath = Path(fn).parent,  # path to the data (.txt files are in this directory)
        datatype = "tone",
        fnamepos = "20230124-1207-n-16000.000.txt",
        fnameneg = "20230124-1207-p-16000.000.txt",
        lineterm="\r",
    )
print(w.shape)
exit()

f = h5py.File(fn, 'r')
datap = f['/datap']
datan = f['/datan']
stim = f['/stim'].asstr()
hardware = f['/hardware'].asstr()
calibration = f['/calibration'].asstr()
st = json.loads(str(stim[:][0][0]))
hw = json.loads(str(hardware[:][0][0]))
cal = json.loads(str(calibration[:][0][0]))
for s in [st, hw, cal]:
    k = list(s.keys())
    for key in s.keys():
        if key in ["wave", 'ACQPars_tb', 'maxdB', 'dBSPL', 'dBSPL_nf', 'dBSPL_bp', 'Vmeas', 'Vmeas_bp', 'CHK75', 'SPKR']:
            continue
        print(key, s[key])
tb = np.array(st['ACQPars_tb'])
n_stim = len(st["wave"])
ni_tb = np.linspace(0, (1.0/st["NIFreq"])*n_stim, n_stim)
sh = datap.shape
print(sh)
sh2 = int(sh[1]/2)
datac = np.zeros((sh[0], sh2))
for tr in range(datap.shape[0]):
    datac[tr,:] = (datap[tr,:sh2] + datap[tr,sh2:])/4
    datac[tr,:] += (datan[tr,:sh2] + datan[tr,sh2:])/4
f, ax = mpl.subplots(3, 1, figsize=(6,10))
step = 5e-7
for tr in range(datac.shape[0]):
    ax[0].plot(tb[:sh2], datac[tr,:]+step*tr, linewidth=0.5)
    # ax[1].plot(tb[:sh2], datac[tr,:]+step*tr, linewidth=0.5)
    ax[1].plot(ni_tb, st["wave"], linewidth=0.5)
    # ax[2].plot(tb, datap[tr,:]+datan[tr,:], linewidth=0.5)
mpl.show()

