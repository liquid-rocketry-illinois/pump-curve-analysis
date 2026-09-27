import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# Resolve data file paths

cwd = Path.cwd()
pt1_path = list(cwd.glob('*PT1-W-PROP*'))[0]
pt2_path = list(cwd.glob('*PT2-W-PROP*'))[0]
flow_path = list(cwd.glob('*Flowmatic 3000*'))[0]

# Confirm desired time range

t_approx = [float(input('Approx. start timestamp? ')), float(input('Approx. end timestamp? '))]

# Load data

pt1 = pd.read_csv(pt1_path)
pt1.rename(columns={'data': 'pt1'}, inplace=True)

pt2 = pd.read_csv(pt2_path)
pt2.rename(columns={'data': 'pt2'}, inplace=True)

flow = pd.read_csv(flow_path)
flow.rename(columns={'data': 'flow'}, inplace=True)

# Merge data and fill gaps due to sample rate differences

merged = pd.merge(pd.merge(pt1, pt2, on='relseconds', how='outer'), flow, on='relseconds', how='outer')
merged.interpolate(inplace=True)
merged.set_index('relseconds', inplace=True)

# Resolve actual time range

[i_start, i_end] = merged.index.get_indexer(t_approx, method='nearest')
t_start = merged.index[i_start]
t_end = merged.index[i_end]

merged = merged[i_start:i_end]

# Create low-pass filter

n = merged.shape[0] # Number of samples
t = t_end - t_start # Sampling time (s)
fs = n / t # Sample rate (Hz)
T = 1 / fs # Sample period (s)

cutoff = 0.07 # Low-pass cutoff, Hz
nyq = 0.5 * fs # Nyquist frequency?
Wn = cutoff / nyq

b, a = butter(2, Wn, btype='Low', analog=False) # Generate 2nd order Butterworth filter coefficients

# Filter PT1, PT2, and flow

merged['filtered_pt1'] = filtfilt(b, a, merged['pt1'])
merged['filtered_pt2'] = filtfilt(b, a, merged['pt2'])
merged['filtered_flow'] = filtfilt(b, a, merged['flow'])

merged['filtered_dp'] = merged['filtered_pt2'] - merged['filtered_pt1']

# Plot

# merged.plot(y='filtered_dp', x='filtered_flow', style='.')
# merged.plot(y='filtered_flow')

fig, ax = plt.subplots()
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax.grid(visible=True)

ax.plot('filtered_flow', 'filtered_dp', '.', data=merged)

plt.show()