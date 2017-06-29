from cmath import phase
from datetime import datetime, timedelta
from functools import reduce
from random import sample, seed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cProfile
import os
matplotlib.style.use('ggplot') # yhat wrapper
"""
matplotlib.use() must be called *before* pylab, matplotlib.pyplot, or matplotlib.backends 
is imported for the first time
"""
matplotlib.use('Agg')

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
LF1V_train = [get_measurements(dir_path, 'LF1V') for dir_path in train_dir]
LF1I_train = [get_measurements(dir_path, 'LF1I') for dir_path in train_dir]

# Phase 2
LF2V_train = [get_measurements(dir_path, 'LF2V') for dir_path in train_dir]
LF2I_train = [get_measurements(dir_path, 'LF2I') for dir_path in train_dir]

def create_indexed_signal(timestamps, signal): # both assumed to be time frames
    signal['timestamps'] = timestamps
    signal_indexed = signal.set_index(['timestamps'])
    return signal_indexed

LF1V_train_indexed = [create_indexed_signal(*pair) for pair in zip(time_ticks_1_train, LF1V_train)]
LF1I_train_indexed = [create_indexed_signal(*pair) for pair in zip(time_ticks_1_train, LF1I_train)]

LF2V_train_indexed = [create_indexed_signal(*pair) for pair in zip(time_ticks_2_train, LF2V_train)]
LF2I_train_indexed = [create_indexed_signal(*pair) for pair in zip(time_ticks_2_train, LF2I_train)]

def compute_power_features(voltage, current): # voltage and current
    power = voltage.values * current.values.conjugate() # formula for alternate currents
    complex_power = power.sum(axis=1)
    real_power, react_power, app_power = complex_power.real, complex_power.imag, abs(complex_power)
    ph = [np.cos(phase(power[i,0])) for i in range(len(power[:, 0]))]
    df = voltage.copy()
    df['real_power'],  df['react_power'], df['app_power'], df['ph'] = real_power, react_power, app_power, ph
    df.columns.name = 'power_signals'
    return df[['real_power', 'react_power', 'app_power', 'ph']]

power_features_1 = [compute_power_features(*pair) for pair in zip(LF1V_train_indexed, LF1I_train_indexed)]
power_features_2 = [compute_power_features(*pair) for pair in zip(LF2V_train_indexed, LF2I_train_indexed)]

## testing
# LF1V_train_indexed.head(5)

def create_fig_dir(dir_path):
    fig_path = dir_path + '/figs'
    os.system('rm -R %s' % fig_path)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    return fig_path

fig_dir = [create_fig_dir(dir_path) for dir_path in train_dir]


## test dir
#test_dir = train_dir[0] + '/figs'

def plot_signal(signal_ts, dir_path, name=None):
    plt.figure()
    signal_ts.plot(subplots=True, figsize=(6, 6))
    if name:
        label = name
    else:
        label = 'generic'
    fig_path = dir_path + '/' + label
    #plt.title('A signal is being plotted') # needs customisation
    plt.savefig(dir_path + '/' + label)
    plt.close()
    return 'Plot saved to %s' % fig_path

## consider creating a hash/table or dictionary
def resample_and_normalize(signal_ts): # use indicative names
    resampled_signal = signal_ts.resample('1S').mean()
    cum_mean_signal = resampled_signal.expanding().mean()
    return (resampled_signal.shift(-1) - cum_mean_signal)/cum_mean_signal

features_1_transformed = [signal_ts.apply(resample_and_normalize, axis=0) for signal_ts in power_features_1]
features_2_transformed = [signal_ts.apply(resample_and_normalize, axis=0) for signal_ts in power_features_2]

## will need a double-loop here
## WARNING: this loops may hog a lot of time
power_figs_1 = [plot_signal(signal, fig_dir[k], 'power_signals_1') for k, signal in enumerate(features_1_transformed)]
power_figs_2 = [plot_signal(signal, fig_dir[k], 'power_signals_2') for k, signal in enumerate(features_2_transformed)]


## appliance data
def load_and_transform_tagging_info(file_path):
    df = pd.read_csv(file_path, squeeze=True, names=['id','appliance','turned_ON', 'turned_OFF'])
    df[['turned_ON', 'turned_OFF']] = df[['turned_ON', 'turned_OFF']].applymap(lambda ts: datetime.fromtimestamp(ts))
    return df

## test load
file_path = train_dir[0] + u'/TaggingInfo.csv'
tagging_info = [load_and_transform_tagging_info(file_path=(dir_path + '/TaggingInfo.csv')) for dir_path in train_dir]

## searching for the master bathroom fan
def measure_spikes(signal_ts, appliance_data, use_buffer=False, buffer_size=5):
    """
    - consider calling the plotting the signals here too -- for reference 
    """
    if use_buffer: ## remember also that there are delays in the replies
        buffer = timedelta(seconds=buffer_size)
    else:
        buffer = timedelta(seconds=0)

    def zoom_into_appliance_window(appliance_id):
        isApplianceOn = (signal_ts.index >= appliance_data['turned_ON'][appliance_id] - buffer) & \
                       (signal_ts.index <= appliance_data['turned_OFF'][appliance_id] + buffer)
        # for testing only
        #print(sum(isApplianceOn))
        appliance_ts = signal_ts.loc[isApplianceOn]
        appliance_delta_ts = appliance_ts.diff(1).abs().dropna() #(appliance_ts.diff(1) / appliance_ts.shift(-1)).abs()
        # changing the names for reference ## call plotting function here
        appliance_delta_ts.columns = [col + '_delta' for col in appliance_ts.columns]
        jump = appliance_ts.abs().max()
        variance = appliance_delta_ts.abs().sum()
        return pd.concat([jump, variance]).values

    #def compute_total_variance(signal_ts):  # or whichever other measure we want
    #    return signal_ts.diff(1).abs().sum()

    #[zoom_into_appliance_window(k) for k, app in enumerate(appliance_data['appliance'])]
    spike_measures = [zoom_into_appliance_window(k) for k, app in enumerate(appliance_data['appliance'])]
    power_matrix = np.column_stack(spike_measures).transpose()
    names = ['real_power_max', 'react_power_max', 'app_power_max', 'ph_max', 'real_power_var', 'react_power_var', 'app_power_var', 'ph_var']
    power_df = pd.DataFrame(power_matrix, columns=names) # optional: index=appliance_data['appliance']
    #power_reduce = reduce((lambda x,y: pd.concat(x, y)), power_responses)
    #appliance_data['jump_size', 'total_variance'] = power_responses
    return pd.concat([appliance_data, power_df], axis=1) #power_matrix #appliance_data

## test appliance responses
#test_responses = measure_spikes(signal_ts=features_1_transformed[1], appliance_data=tagging_info[1])

## measuring the responses for phase 1 alone
signal_tags_pairs_1 = zip(features_1_transformed, tagging_info) ## nested for loop

#### generates responses per appliance and their deltas
## respose_lists_2 is missing
power_responses_1 = pd.concat([measure_spikes(*p) for p in signal_tags_pairs_1])
power_responses_1.to_csv(path_or_buf=data_dir + '/summary_phase_1.csv', sep='\t')

## how to fit a gaussian curve to a time series? feature extraction
## gaussian processes?

## create a data frame with

#print(os.listdir(os.path.join(training_data[0])))

# def parser(x):
#     return datetime.strptime('190' + x, '%Y-%m')
#
#
# series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# print(series.head())
# series.plot()
# plt.show()

def test():
    # include some asserts here, PN style
    return None