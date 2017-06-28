from __future__ import print_function

import os

import numpy as np
import scipy.io as spio # this is the Scipy module that laods mat-files
from datetime import datetime, date, time
import pandas as pd

import matplotlib.pyplot as plt
from cmath import phase


# for accessing the file in a child folder # generalize this into a function
file_dir = os.path.dirname(os.path.realpath('__file__'))
H1_dir = os.path.join(file_dir, "data/H3") # or data/H3
H1_training_files = filter(lambda x: 'Training' in x, os.listdir(H1_dir))

def data_loader(path):
    return spio.loadmat(path, struct_as_record=False, squeeze_me=True) # these creates numpy arrays much easier to use

loaded_files = [data_loader(os.path.join(H1_dir, file)) for file in os.listdir(H1_dir)][0] ## for testing only

# extract tables
buf = loaded_files['Buffer'] ## investigate all keys
fieldnames = buf._fieldnames

## ravel = untangle something (for concrete objects)
print('*' * 79)
for key in fieldnames:
     print(key + ' ' + str(eval('buf.' + key).shape))
print('*' * 79)

## form pandas data frame for HF
#hf_cols = {n: eval('buf.' + n) for n in fieldnames if 'HF' in n}

## align arrays for concatenation
# def isCompatible(a, b): # a and b arrays
#     return a[0] == b[0] or a[1] == b[1] # try and exception here
#
# def align_arrays(arrays_list):
#     return list(map(lambda x: x if isCompatible(x.shape, arrays_list[0].shape) else x.T, arrays_list))

#hf_df = pd.DataFrame(np.concatenate(align_arrays([v for k, v in hf_cols.items()]), axis=1))
#hf_df.rename(columns={'0': 'TimeTicksHF'}) # accepts a lambda function too
#hf_df[0]=[datetime.fromtimestamp(*ts) for ts in hf_cols['TimeTicksHF']]

## extracting tables
LF1V = buf.LF1V
LF1I = buf.LF1I
LF2V = buf.LF2V
LF2I = buf.LF2I
L1_TimeTicks = buf.TimeTicks1
L2_TimeTicks = buf.TimeTicks2
HF = buf.HF
HF_TimeTicks = buf.TimeTicksHF

taggingInfo = buf.TaggingInfo

# calculate power (by convolution)
L1_P = LF1V * LF1I.conjugate() ## all arithmetics operates elementwise
L2_P = LF2V * LF2I.conjugate()

#
L1_ComplexPower = L1_P.sum(axis=1)
L2_ComplexPower = L2_P.sum(axis=1)

# extract components
L1_Real, L1_Imag, L1_App = L1_ComplexPower.real, L1_ComplexPower.imag, abs(L1_ComplexPower)
L2_Real, L2_Imag, L2_App = L2_ComplexPower.real, L2_ComplexPower.imag, abs(L2_ComplexPower)

#
L1_Pf = [np.cos(phase(L1_P[i,0])) for i in range(len(L1_P[:,0]))]
L2_Pf = [np.cos(phase(L2_P[i,0])) for i in range(len(L2_P[:,0]))]
L1_Pf = np.array(L1_Pf, dtype='float64')
L2_Pf = np.array(L2_Pf, dtype='float64')


## plotting the devices
def add_devices(ax, taggingInfo, H=500, start=0):
    """
   Add a vertical green/red line for every device.
   First device will be at y=bottom, second at y=bottom+step etc
   Device names will be displayed on the left
   """
    N = len(taggingInfo)
    #H = 500
    dy = 500//N
    for i in range(N):
        ax.plot([taggingInfo[i, 2], taggingInfo[i, 2]], [start, H], color='green', linewidth=6)
        ax.plot([taggingInfo[i, 3], taggingInfo[i, 3]], [start, H], color='red', linewidth=6)
        str1 = 'ON-%s' % taggingInfo[i, 1]
        ax.text(taggingInfo[i, 2], i * dy, str1)
        str2 = 'OFF-%s' % taggingInfo[i, 1]
        ax.text(taggingInfo[i , 3], i * dy, str2)

## plotting
# subset is the range of indices of L1_TimeTicks to plot
#subset = range(300000, 360000) ## figure out plotting interval basic on appliance usage

plt.clf()
fig = plt.figure(1)
ax1 = plt.subplot(411) # 4 stacked graphs
#ax1.plot(L1_TimeTicks[subset], L1_Real[subset], color='blue') # this works with datetimes as well
#ax1.plot([datetime.fromtimestamp(ts) for ts in np.nditer(L1_TimeTicks.reshape(-1,1))], L1_Real, color='blue')
ax1.plot(L1_TimeTicks, L1_Real, color='blue')
ax1.set_title('Real Power (W) and ON/OFF Device Category IDS')
fig.set_dpi(150)
fig.set_size_inches(18.5, 50.5)
ax1_H = L1_Real.max() - L1_Real.min()
add_devices(ax1, taggingInfo, ax1_H/10, L1_Real.min())
# This will draw a green line for every device while it is turned on

## Plot Imaginary/Reactive power (VAR)
ax2 = fig.add_subplot(412)
ax2.plot(L1_TimeTicks, L1_Imag)
ax2.set_title('Imaginary/Reactive power (VAR)')
ax2_H = L1_Imag.max() - L1_Imag.min()
add_devices(ax2, taggingInfo, ax2_H/10, L1_Imag.min())

## Plot Power Factor
ax3 = fig.add_subplot(413)
ax3.plot(L1_TimeTicks, L1_Pf)
ax3.set_title('Power Factor')
ax3.set_xlabel('Unix Timestamp')
ax3_H = L1_Pf.max() - L1_Pf.min()
add_devices(ax3, taggingInfo, ax3_H/10, L1_Pf.min())

plt.show()
