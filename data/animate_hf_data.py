"""
simple example of an animated plot
"""

## always start the project as a virtual environment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

data_dir = '/Users/alex.decastro/development/gitlab/belkin_challenge/data/CSV_OUT/Tagged_Training_02_15_1360915201'

## had resample hf data at every 15 seconds since data frame woudn't fit into memory was it was taking too long
fields = range(0, 80997, 5)
df = pd.read_csv(data_dir + '/HF.csv', delimiter=',', header=None, na_filter=False, dtype={'x': np.float}, usecols=fields)
#dg = df.pct_change
df_filtered = df.rolling(window=3).mean() # filter out noise and compute first order differences
dg = df.diff(1) / df.shift(1) # same as pct_change?

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

## include data transformation here

fig, ax = plt.subplots()
x = range(dg.shape[0])
line, = ax.plot(x, dg.iloc[:, 0])

def animate(i):
    line.set_ydata(dg.iloc[:, i])  # update the data
    return line,

# Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

# plotting first two hundred samples (~ sampled every 1S each)
ani = animation.FuncAnimation(fig, animate, range(dg.shape[0]), init_func=init,
                              interval=50, blit=True)  # interval between frames. Default is 200 ms

ani.save(data_dir + '/hf_animation.mp4', writer=writer)

plt.show()