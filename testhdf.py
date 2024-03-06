

fn = "/Volumes/Pegasus_002/ManisLab_Data3/abr_data/Reggie_NIHL/Glyt2-GFP_M_VF1_p56_NE2wks106/20240214-1029.h5"
import h5py
import numpy as np
import json
import matplotlib.pyplot as mpl


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
tb = np.array(st['ACQPars_tb'])*2.0
n_stim = len(st["wave"])
ni_tb = np.linspace(0, (1.0/st["NIFreq"])*n_stim, n_stim)
print(datan.shape)
f, ax = mpl.subplots(3, 1)
for tr in range(datap.shape[0]):
    ax[0].plot(tb, datap[tr,:], linewidth=0.5)
    ax[1].plot(tb, datan[tr,:], linewidth=0.5)
    ax[2].plot(ni_tb, st["wave"], linewidth=0.5)
    # ax[2].plot(tb, datap[tr,:]+datan[tr,:], linewidth=0.5)
mpl.show()

