import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path
from scipy.signal import lfilter,butter,filtfilt
from matplotlib import pyplot
import matplotlib as mpl

def findFiles():
    global rawPt1, rawPt2, rawFlow, rawVelo
    csvFiles = [f.name for f in Path('.').glob('*.csv')]
    for fileName in csvFiles:
        try:
            fileName.index('-0x92-0-PT1-W-PROP')
            rawPt1 = pd.read_csv(fileName)
            rawPt1 = rawPt1.rename(columns={'data': 'Pt1'})
            print("Found PT1 Data")
            continue
        except ValueError:
            pass

        try:
            fileName.index('-0x92-1-PT2-W-PROP')
            rawPt2 = pd.read_csv(fileName)
            rawPt2 = rawPt2.rename(columns={'data': 'Pt2'})
            print("Found PT2 Data")
            continue
        except ValueError:
            pass

        try:
            fileName.index('-0x96-0-Flowmatic 3000')
            rawFlow = pd.read_csv(fileName)
            rawFlow = rawFlow.rename(columns={'data': 'Flow'})
            print("Found Flow Data")
            continue
        except ValueError:
            pass

        try:
            fileName.index('-0x 5-0-Pump Motor')
            rawVelo = pd.read_csv(fileName)
            rawVelo = rawVelo.rename(columns={'speed': 'Velo'})
            print("Found Velo Data")
            continue
        except ValueError:
            pass


def hampel_filter(df, data_col='data', window=7, n_sigmas=3, output_col='dataWithNoSpikes',
                   timestamp_col='timestamp'):
    """
    Remove short spikes/outliers from a dataset with a DataFrame index and
    a timestamp column.

    Parameters
    ----------
    df : pd.DataFrame
        Input with the (existing) DataFrame index preserved, a timestamp
        column, and a data column.
    data_col : str
        Name of the data column to despike.
    window : int
        Rolling window size (centered).
    n_sigmas : float
        Outlier threshold in robust standard deviations.
    output_col : str
        Name of the output (despiked) column.
    timestamp_col : str
        Name of the timestamp column to carry through.

    Returns
    -------
    pd.DataFrame
        Same index as input, columns: <timestamp_col>, <output_col>
    """
    x = df[data_col].astype(float)
    k = 1.4826  # scales MAD to be comparable to a standard deviation

    rolling_median = x.rolling(window, center=True).median()
    mad = x.rolling(window, center=True).apply(
        lambda w: np.median(np.abs(w - np.median(w))), raw=True
    )
    threshold = n_sigmas * k * mad

    diff = (x - rolling_median).abs()
    outlier_mask = diff > threshold

    cleaned = x.copy()
    cleaned[outlier_mask] = rolling_median[outlier_mask]

    out = pd.DataFrame(index=df.index)
    out[timestamp_col] = df[timestamp_col]
    out[output_col] = cleaned

    return out


def findActiveTest(filteredVelo, targetVelo):
    min = 0
    max = 0
    minTimestamp = 0.0
    maxTimestamp = 0.0

    i = 0
    for row in filteredVelo.itertuples():
        if min == 0:
            if abs(row.F_Velo - targetVelo) < 500:
                min = i
                minTimestamp = row.relseconds
        else:
            if abs(row.F_Velo - targetVelo) > 500:
                max = i
                maxTimestamp = row.relseconds
                break
        i += 1

    return [minTimestamp, maxTimestamp]


def filterWithButterworth(inputData):
    t_start = rawCombinedData.relseconds[0]
    t_end = rawCombinedData.relseconds.last_valid_index()

    # Create low-pass filter
    n = rawCombinedData.shape[0] # Number of samples
    t = t_end - t_start # Sampling time (s)
    fs = n / t # Sample rate (Hz)
    T = 1 / fs # Sample period (s)

    # cutoff = 0.05 # Low-pass cutoff, Hz
    # nyq = 0.5 * fs # Nyquist frequency?
    # Wn = cutoff / nyq
    Wn = 0.001

    b, a = butter(2, Wn, btype='Low', analog=False) # Generate 2nd order Butterworth filter coefficients

    filteredData = inputData.copy()

    filteredData['filtered_Pt1'] = filtfilt(b, a, filteredData['Pt1'])
    filteredData['filtered_Pt2'] = filtfilt(b, a, filteredData['Pt2'])
    filteredData['filtered_Flow'] = filtfilt(b, a, filteredData['Flow'])
    filteredData['filtered_Velo'] = filtfilt(b, a, filteredData['Velo'])
    return filteredData


#Find large spikes in motor velocity data
def findSpikes(data):
    # constants
    checkBound = 50
    maxDifference = 75.0

    # Loop through the entire dataset to look for spikes
    spikeTimestamps = []
    i = 0
    while i < data.shape[0]:
        difference = abs(data['Diffs'][i])
        if difference > maxDifference: # Finds a spike
            largestIndex = i
            j = -checkBound

            if i < checkBound: # ensure it doesn't check a negative index
                j = -i

            # checks backwards and forwards for a larger spike
            while j <= checkBound and i+j < data.shape[0]:
                testDiff = abs(data['Diffs'][i+j])
                if testDiff > difference:
                    difference = testDiff
                    largestIndex = i+j
                    break
                j += 1
            
            midSeconds = data['relseconds'][largestIndex]

            # Try to find the largest timestamp before the velo change
            minSeconds = data['relseconds'][0]
            k = -1
            while k + i >= 0:
                if abs(data['Diffs'][i+k]) > 50:
                    k -= 1
                    continue
                h = -1
                allGood = True
                while h >= -20 and k+i+h > 0:
                    if abs(data['Diffs'][i+k+h]) > 50.0:
                        allGood = False
                        break
                    h -= 1
                if allGood:
                    minSeconds = data['relseconds'][i+k]
                    break
                k -= 1

            # Try to find the smallest timestamp after the velo change
            maxSeconds = data['relseconds'][data.shape[0]-1]
            k = 1
            while k + i < data.shape[0]:
                if abs(data['Diffs'][i+k]) > 50:
                    k += 1
                    continue
                h = 1
                allGood = True
                while h <= 20 and h+i+k < data.shape[0]:
                    if abs(data['Diffs'][i+k+h]) > 50.0:
                        allGood = False
                        break
                    h += 1
                if allGood:
                    maxSeconds = data['relseconds'][i+k]
                    break
                k += 1

            spikeTimestamps.append([midSeconds, minSeconds, maxSeconds])

            # Above code concludes that this is the largest difference within checkBound indexes on either side
            i += checkBound # Therefore the next checkBound indexes don't need to be checked.

        i += 1
    
    return spikeTimestamps


viridis = mpl.colormaps['viridis'].resampled(20)
# Adds different colored vertical lines to a plot at the timestamps provided
def addTimestampLinesToPlot(plot, timestamps):
    global viridis
    i = 0
    for timestamp in timestamps:
        plot.axvline(x=timestamp[0], color=viridis(float(i)/len(timestamps)), label=str(i), linestyle='--')
        i += 1


# Adds different colored vertical lines to a plot at the timestamps provided
# Includes the +- timstamps as well
def addTimestampLinesToPlotRecursive(plot, timestamps):
    for spikeGroup in timestamps:
        for timestamp in spikeGroup:
            plot.axvline(x=timestamp, color='r')
































print("Pump Data Graphing!!!!")
print("Auto Detect Files? (y/n)")

# Get Files
userInput = input("> ").strip()
if userInput.capitalize() == "Y":
    findFiles()
else:
    print("Manual Input Not Implemented :(")
    time.sleep(5)
    sys.exit()

# Super Filter Velo and find rapid changes in motor velocity
despikeVelo = hampel_filter(rawVelo, data_col="Velo", window=100, n_sigmas=0.5, output_col="F_Velo", timestamp_col='relseconds')
testVelo = rawVelo.merge(despikeVelo, on='relseconds', how="outer")
testVeloPlot = testVelo.plot(x='relseconds', y=['Velo', 'F_Velo'], color=['green', 'blue'])
testVelo['Diffs'] = testVelo['F_Velo'].diff()
# diffPlot = testVelo.plot(x='relseconds', y='Diffs')
spikeTimestamps = findSpikes(testVelo)
addTimestampLinesToPlot(testVeloPlot, spikeTimestamps)
# addTimestampLinesToPlotRecursive(diffPlot, spikeTimestamps)
testVeloPlot.legend()
pyplot.show(block=False)

# Chose Regions
print("Input desired regions to analyze")
print("To use the data in the region between 1,2 and the region between 5,6")
print("Enter in form of 1:2,5:6")
regionsInput = input("> ").strip().replace(" ", "")
regionsInput = regionsInput.split(",")

# combine data
rawCombinedData = rawPt1.merge(rawPt2, on='relseconds', how='outer',)
rawCombinedData = rawCombinedData.merge(rawFlow, on='relseconds', how='outer')
rawCombinedData = rawCombinedData.merge(rawVelo, on='relseconds',how='outer')
rawCombinedData = rawCombinedData.sort_values('relseconds').reset_index(drop=True)

rawCombinedData = rawCombinedData.interpolate('linear')

# Get Indexes For Regions
regionIndexes = []
for indexes in regionsInput:
    raw = indexes.split(":")
    minMax = [spikeTimestamps[int(raw[0])][2],spikeTimestamps[int(raw[1])][1]]

    i = 0
    for row in rawCombinedData.itertuples(index=True):
        if (row.relseconds > minMax[0]):
            break
        i += 1
    j = 0
    for row in rawCombinedData.itertuples(index=True):
        if (row.relseconds > minMax[1]):
            j -= 1
            break
        j += 1
    regionIndexes.append([i,j])

# Get data for regions
intermediateData = rawCombinedData[regionIndexes[0][0]:regionIndexes[0][1]]
i = 1
while i < len(regionIndexes):
    intermediateData = pd.concat([intermediateData, rawCombinedData[regionIndexes[i][0]:regionIndexes[i][1]]], ignore_index=True)
    i += 1
rawCombinedData = intermediateData

# get front nan values
i = 0
for row in rawCombinedData.itertuples(index=True):
    # Access columns using the dot notation (row.Column_Name)
    if pd.isna(row.Pt1) or pd.isna(row.Pt2) or pd.isna(row.Flow) or pd.isna(row.Velo):
        pass
    else:
        break

    i += 1

# get back nan values
j = rawCombinedData.shape[0]
for row in rawCombinedData.iloc[::-1].itertuples():
    if pd.isna(row.Pt1) or pd.isna(row.Pt2) or pd.isna(row.Flow) or pd.isna(row.Velo):
        pass
    else:
        break

    j -= 1

# remove nan values from front and back
rawCombinedData = rawCombinedData[i+1:j].reset_index(drop=True)

# filter data
filteredData = filterWithButterworth(rawCombinedData)

# Make plots
filteredData.plot(x='relseconds',y=['Pt1', 'Pt2', 'filtered_Pt1', 'filtered_Pt2'], color=['dodgerblue','salmon','blue', 'red'])
filteredData.plot(x='relseconds', y=['Flow', 'filtered_Flow'], color=['mediumOrchid', 'purple'])
filteredData.plot(x='relseconds', y=['Velo', 'filtered_Velo'], color=['mediumaquamarine', 'darkgreen'])

# Calculate deltaP
filteredData['DeltaP'] = filteredData['filtered_Pt2'] - filteredData['filtered_Pt1']

# More plots
filteredData.plot(x='relseconds', y='DeltaP')
filteredData.plot.scatter(x='filtered_Flow',y='DeltaP',s=1)

pyplot.show(block=False)

# Continue To Leakage Prompt
print("Continue to Leakage? (y/n)")
userInput = input("> ").strip()
if userInput.capitalize() != 'Y':
    sys.exit()


print("Input desired regions to use")
print("To use the data in the region between 1,2 and the region between 5,6")
print("Enter in form of 1:2,5:6")
regionsInput = input("> ").strip().replace(" ", "")
regionsInput = regionsInput.split(",")

# turn string input into indexes
# Get Indexes For Regions
totalDeltaTime = 0.0
for indexes in regionsInput:
    raw = indexes.split(":")

    startTime = spikeTimestamps[int(raw[0])][0]
    endTime = spikeTimestamps[int(raw[1])][1]

    totalDeltaTime += endTime-startTime

print("DeltaTime: " + str(totalDeltaTime) + " (s)")

# Get weight of water leaked throught test
print("Input weight of leakage (kg)")
weightInput = float(input("> ").strip())


print("Leakage rate = " + str(weightInput / totalDeltaTime) + " kg/s")

print("close?")
input("> ")