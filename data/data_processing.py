import pandas as pd
import numpy as np
from cmath import phase
from datetime import datetime, timedelta
from matplotlib import pyplot
from random import sample, seed
import os

import cProfile

## to profile code
#cProfile.run('your python command here')


# for accessing the file in a child folder # generalize this into a function
base_dir = os.path.dirname(os.path.realpath('__file__'))
data_dir = os.path.join(base_dir, "data/CSV_OUT")

def get_data_dirs(type='Tagged_Training'):
    return [os.path.join(data_dir, dir_name) for dir_name in os.listdir(data_dir) if type in dir_name]

train_dir = get_data_dirs()


# get all time ticks
def get_timestamps(dir_path, type=1): # type = 1, 2, 'HF'
    file_path = dir_path + u'/TimeTicks' + str(type) +  '.csv'
    ts = pd.read_csv(file_path, squeeze=True, names=[u'timestamps'])
    ts = pd.to_datetime(ts, unit='s')
    #ticks = ts.index.apply(lambda ts: datetime.fromtimestamp(*ts))
    return ts

time_ticks_1_train = [get_timestamps(dir_path) for dir_path in train_dir]
time_ticks_2_train = [get_timestamps(dir_path, 2) for dir_path in train_dir]
time_ticks_HF_train = [get_timestamps(dir_path, 'HF') for dir_path in train_dir]


## N X 6 array of fundamental and first 5 harmonics of 60Hz voltage measurement Phase-1
# header = 0
def get_measurements(dir_path, type='LF1V'): # type = 'LF*I', 'LF*V', 'HF' and where * = 1, 2
    N_HARM = 6
    N_FFT = 4096
    path = dir_path + '/' + str(type) + '.csv'
    if 'LF' in type:
        names = [u'E_%s' % k for k in range(0, N_HARM)]
    else:
        names = [u'k_%s' % f for f in range(0, N_FFT)]
    df = pd.read_csv(path, squeeze=True, names=names)
    df_cpx = df.apply(lambda col: col.apply(lambda val: complex(val.replace('i','j'))))
    df_cpx.columns.name = type
    return df_cpx


print('This is the data file we will be working with:', train_dir[0])
#timeTicks1 = pd.read_csv(train_dir[0] + u'/TimeTicks1.csv', squeeze=True, names=[u'timestamps'])
#timeTicks1 = timeTicks1.apply(lambda ts: datetime.fromtimestamp(ts))
#LF1V = pd.read_csv(train_dir[0] + u'/LF1V.csv', squeeze=True, names=[u'E_%s' % k for k in range(0,6)]) # returns series if parsed data only contains one column

# Phase 1
LF1V_train = [get_measurements(dir_path) for dir_path in train_dir]
LF1I_train = [get_measurements(dir_path, 'LF1I') for dir_path in train_dir]

# Phase 2
LF2V_train = [get_measurements(dir_path, 'LF2V') for dir_path in train_dir]
LF2I_train = [get_measurements(dir_path, 'LF2V') for dir_path in train_dir]

def create_indexed_signal(timestamps, signal): # both assumed to be time frames
    signal['timestamps'] = timestamps
    signal_indexed = signal.set_index(['timestamps'])
    return signal_indexed

LF1V_train_indexed = [create_indexed_signal(*pair) for pair in zip(time_ticks_1_train, LF1V_train)]
LF1I_train_indexed = [create_indexed_signal(*pair) for pair in zip(time_ticks_1_train, LF1I_train)]

def compute_power_features(voltage, current): # voltage and current
    power = voltage.values * current.values.conjugate() # formula for alternate currents
    complex_power = power.sum(axis=1)
    real_power, react_power, app_power = complex_power.real, complex_power.imag, abs(complex_power)
    ph = [np.cos(phase(power[i,0])) for i in range(len(power[:, 0]))]
    df = voltage.copy()
    df['real_power'],  df['react_power'], df['app_power'], df['ph'] = real_power, react_power, app_power, ph
    return df[['real_power', 'react_power', 'app_power']]

power_features_1 = [compute_power_features(*pair) for pair in zip(LF1V_train_indexed, LF1I_train_indexed)]

## testing
# LF1V_train_indexed.head(5)

#LF1V_with_time = LF1V.copy(deep=True)
#LF1V_with_time['timeTicks1'] = timeTicks1
#LF1V_with_time.head(5)

#LF1I_with_time = LF1I.copy(deep=True)
#LF1I_with_time['timeTicks1'] = timeTicks1
#LF1I_with_time.head(5) ## wrap this in a class: too much boiler-plate code

## calculate power by convolution
L1_P = LF1V.values * LF1I.values.conjugate()
L1_ComplexPower = L1_P.sum(axis=1)
L1_Real, L1_Imag, L1_App = L1_ComplexPower.real, L1_ComplexPower.imag, abs(L1_ComplexPower)
L1_Pf = [np.cos(phase(L1_P[i,0])) for i in range(len(L1_P[:,0]))]
L1_Pf = np.array(L1_Pf, dtype='float64')

L1_Real_ts = pd.concat([timeTicks1, pd.DataFrame(L1_Real)], axis=1) ## first pandas time series?
L1_Real_ts.columns = ['timeTicks', 'L1_Real']
L1_Real_ts = L1_Real_ts.set_index('timeTicks')
L1_Real_ts.plot()
if not os.path.exists(csv_dir + '/figs'):
    os.makedirs(csv_dir + '/figs')
pyplot.savefig(csv_dir + '/figs/L1_Real.png')
#pyplot.show()

## moving averages and spike detection
L1_Real_resampled = L1_Real_ts.resample('1S').mean() # filter noise: resample each 5 seconds
L1_Real_cum_mean = L1_Real_resampled.expanding().median()
baseline_dev = (L1_Real_resampled.shift(1) - L1_Real_cum_mean)/L1_Real_cum_mean # compute delta from median

## tagging info and estimating spikes
tagging_info = pd.read_csv(training_data[0] + u'/TaggingInfo.csv',
                           squeeze=True,
                           names=['id','appliances','turned_ON', 'turned_OFF'])
tagging_info.columns = ['id', 'appliance', 'ON_time', 'OFF_time']
tagging_info['ON_time'] = tagging_info['ON_time'].apply(lambda ts: datetime.fromtimestamp(ts))
tagging_info['OFF_time'] = tagging_info['OFF_time'].apply(lambda ts: datetime.fromtimestamp(ts))

## searching for the master bathroom fan
def compute_L1_spike(appliance_id, useBuffer=False):
    if useBuffer:
        buffer = timedelta(seconds=5)
    else:
        buffer = timedelta(seconds=0)
    applianceSpiked = (baseline_dev.index >= tagging_info['ON_time'][appliance_id] - buffer) & \
                      (baseline_dev.index <= tagging_info['OFF_time'][appliance_id] + buffer)
    response_ts = baseline_dev.loc[applianceSpiked]
    delta_response = (response_ts.diff(1) / response_ts.shift(-1)).abs()
    delta_std = delta_response.std()
    delta_mean = delta_response.mean()
    #delta_response.plot()
    #print(delta_response.max(), delta_mean + 2*delta_std)
    return delta_response.max() #> delta_mean + 2*delta_std #response_ts.max(), response_ts

#seed(42)
#random_test = sample(range(len(tagging_info)), 10)
#test_cases = [compute_L1_spike(k, True) for k in random_test]
tagging_info['L1_spikes'] = [compute_L1_spike(k, True) for k in range(len(tagging_info))]

## how to fit a gaussian curve to a time series? feature extraction

## create a data frame with

#print(os.listdir(os.path.join(training_data[0])))

# def parser(x):
#     return datetime.strptime('190' + x, '%Y-%m')
#
#
# series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# print(series.head())
# series.plot()
# pyplot.show()

def test():
    # include some asserts here, PN style
    return None