import os
import pickle
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

l_observationWindow = [np.timedelta64(np.timedelta64(2,'W'), 'ms'), # equally observed.
                       np.timedelta64(np.timedelta64(4,'W'), 'ms'),
                       np.timedelta64(np.timedelta64(8,'W'), 'ms')]

# TRAINING
# Can be negative for training. (negative means overlapping windows)
# Reduce this number even further to create more samples.
n_split_separation = [np.timedelta64(np.timedelta64(-6,'D'), 'ms'), # min
                      np.timedelta64(np.timedelta64(-13,'D'), 'ms')] # max.

# TESTING
# n_split_separation = [np.timedelta64(np.timedelta64(3,'W'), 'ms'), # min
#                       np.timedelta64(np.timedelta64(7,'W'), 'ms')] # max.

n_min_datapoint = 2*7-1

def xySplit(lstrFolderPathPartition,
            strFolderPathOutputX,
            strFolderPathLabel,
            strFileHeader,
            strFilePathOutputY,
            l_observationWindow=l_observationWindow,
            n_split_separation=n_split_separation,
            n_min_datapoint=n_min_datapoint):
    '''
    This function take the all the timeseries data in each of the lstrFolderPathPartition,
    split them to X (features) and Y (labels).
    The X are split into chuncks with shorter windows, and each chunck is saved as a separate file.
    All of the chuncks are saved in the strFolderPathOutputX folder.
    The Y are saved in a single file, the strFilePathOutputY file.
    '''
    strTempHeader = 'temporaryfilename_'
    # This is the dataframe that maps the string labels (Y) to the one-hot encoding
    dfLabelMapper = pd.read_csv(strFolderPathLabel, index_col=0)
    n_split_separation = np.array(n_split_separation)
    # The total duration of each chunck group.
    # The number of chuncks are integers multiple of each chunck group
    len_chunk_group = np.array(l_observationWindow).sum() + len(l_observationWindow)*n_split_separation.max()
    # get all of the filenames
    lFilenames = []
    for iFolder in lstrFolderPathPartition:
        filenames = os.listdir(iFolder)
        for filename in filenames:
            lFilenames.append(iFolder+filename)
    # iterate through all of the filenames
    iFile = 0
    lLabels = []
    lTemp = []
    for strFileNameXY in tqdm(lFilenames):
        # Ignore OS files
        if strFileNameXY[-4:] != '.pkl':
            continue
        # Open the file, each containing a single timeseries
        with open(strFileNameXY, 'rb') as file:
            il = pickle.load(file)
        # We remove the absolute time information to prevent information leakage
        il['t'] = il['t']-il['t'][0]
        # GET THE TEMPORAL BOUNDING BOXES FOR EACH CHUNCK
        # This is to split a timeseries into smaller windows.
        total_duration = il['t'][-1]
        # the following line is hard coded for this setup
        n_windows_each = total_duration // len_chunk_group
        if n_windows_each == 0:
            continue
        # array of observation window durations
        aowd = np.array(l_observationWindow)
        aowd = np.repeat(aowd, n_windows_each)
        np.random.shuffle(aowd)
        # array of separation durations
        asd = np.random.rand(len(l_observationWindow)*n_windows_each) # A uniform random [0,1] array of the correct shape
        asd = asd * (n_split_separation.max()-n_split_separation.min()) # correcting the width
        asd += n_split_separation.min() # adding the minimum value
        asd[0] = np.timedelta64(0, 'ms') if asd[0] < np.timedelta64(0, 'ms') else asd[0]
        # array of durations
        ad = np.empty(aowd.size+asd.size, dtype=aowd.dtype)
        ad[0::2] = asd
        ad[1::2] = aowd
        # array of start time
        aost = ad.cumsum()
        # array of Temporal Bounding Boxes
        abb = np.empty((n_windows_each*len(l_observationWindow),2), dtype=aost.dtype)
        abb[:,0] = aost[0::2]
        abb[:,1] = aost[0::2]+aowd
        last_item_in_the_bounding_box = abb[-1,1]
        assert last_item_in_the_bounding_box < total_duration
        # SAVE EACH CHUNCK TO A FILE
        for ii, ibb in enumerate(abb):
            # select each chunck
            time_interval_mask = (ibb[0] <= il['t']) & (il['t'] < ibb[1])
            # ignore if the chunck is too short
            if np.sum(time_interval_mask) < n_min_datapoint:
                continue
            # get the features and labels of each chunk
            features = {'t': il['t'][time_interval_mask],
                        'v': il['v'][time_interval_mask]}
            label = dfLabelMapper.loc[dfLabelMapper.index == il['y']].values[0,:]
            # save the chunck features (X) to a file
            strFileNameX = strTempHeader+str(iFile)+'.pkl'
            with open(strFolderPathOutputX+strFileNameX, 'wb') as file:
                pickle.dump(features, file)
            # record the labels (Y) as a row entry in the label file
            iLabel = [strFileNameX, strFileNameXY, il['y']]
            iLabel.extend(list(label))
            lLabels.append(iLabel)
            iFile += 1
    # Rename all the feature (X) files from the temporary names to the final names
    aLabels = np.array(lLabels)
    np.random.shuffle(aLabels)
    for ii, iFilename in tqdm(enumerate(aLabels[:,0])):
        newFilename = strFileHeader+str(ii)+'.pkl'
        # rename the temporary feature (X) file
        os.rename(strFolderPathOutputX+iFilename,
                  strFolderPathOutputX+newFilename)
        # rename the entry in the label (Y) file
        aLabels[ii,0] = newFilename
    # Save the label (Y) file
    dfLabels = pd.DataFrame(aLabels, columns=['FileNameX', 'FileNameXY', 'strLabel']+list(dfLabelMapper.columns))
    dfLabels.to_csv(strFilePathOutputY)
    return iFile, strFilePathOutputY
