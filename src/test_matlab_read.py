"""Test reading ABR matlab files
"""
from pathlib import Path
import scipy.io
import numpy as np

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    
    from: `StackOverflow <http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries>`_
    '''
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], scipy.io.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def print_mat_nested(d, indent=0, nkeys=0):
    """Pretty print nested structures from .mat files   
    Inspired by: `StackOverflow <http://stackoverflow.com/questions/3229419/pretty-printing-nested-dictionaries-in-python>`_
    """
    
    # Subset dictionary to limit keys to print.  Only works on first level
    if nkeys>0:
        d = {k: d[k] for k in list(d.keys())[:nkeys]}  # Dictionary comprehension: limit to first nkeys keys.

    if isinstance(d, dict):
        for key, value in d.items():         # iteritems loops through key, value pairs
          print ('\t' * indent + 'Key: ' + str(key))
          print_mat_nested(value, indent+1)

    if isinstance(d,np.ndarray) and d.dtype.names is not None:  # Note: and short-circuits by default
        for n in d.dtype.names:    # This means it's a struct, it's bit of a kludge test.
            print( '\t' * indent + 'Field: ' + str(n))
            print_mat_nested(d[n], indent+1)
            if n == 's2':
                print(d[n])

def dtype_shape_str(x):
    """ Return string containing the dtype and shape of x."""
    return str(x.dtype) + " " + str(x.shape)

testfile = Path("/Volumes/Pegasus_002/ManisLab_Data3/abr_data/Reggie_E/B2S_Math1cre_F_3-3_P30_KO/20220401-1137.mat")
matdata = loadmat(testfile)

print_mat_nested(matdata, nkeys=6)

print(matdata['bigdata']['CAL'])

t = matdata['bigdata']['CAL']
print(dtype_shape_str(t['arr']))
print(dtype_shape_str(t['arr'][0]))

print(t['arr'][0])


# mf = scipy.io.loadmat(testfile, struct_as_record=False, squeeze_me=True)
# print(mf.keys())
# print(mf['bigdata'])
# # print("bigdata types: ", mf['bigdata'].dtype)
# print("bigdata DATA: ", mf['bigdata']['DATA'])
# print("bigdata DATA type: ", mf['bigdata']['DATA'].dtype)
# print('bigd: ', mf['bigdata']['DATA'])
# # print(mf['bigdata']['DATA']['arr'].item())
# print(dir(mf['bigdata']['DATA'])) # ['[b'abr4_calibration_struct'])
# keys = []
# for k, v in mf.items():
#     if isinstance(v, np.ndarray):
#         keys.append(k)

# print(keys)
# d = mf[keys[0]]
# print(d)
# # npd = mf['bigdata']['DATA']
# # print(dir(npd))
# # print(npd.tolist())

