import os
import ast
import pickle
import numpy as np
import pandas as pd
import scipy
import sklearn.metrics
from tqdm.auto import tqdm

MEAN = 1.82207407
STD = 3.08905232

l_observationWindow = [np.timedelta64(np.timedelta64(2,'W'), 'ms'), # equally observed.
                       np.timedelta64(np.timedelta64(4,'W'), 'ms'),
                       np.timedelta64(np.timedelta64(8,'W'), 'ms')]

# TRAINING
# Can be negative for training. (negative means overlapping windows)
# Reduce this number even further to create more samples.
n_split_separation_trn = [np.timedelta64(np.timedelta64(-6,'D'), 'ms'), # min
                          np.timedelta64(np.timedelta64(-13,'D'), 'ms')] # max.

# TESTING
n_split_separation_tst = [np.timedelta64(np.timedelta64(3,'W'), 'ms'), # min
                          np.timedelta64(np.timedelta64(7,'W'), 'ms')] # max.

n_min_datapoint = 2*7-1

ARRAY_SHAPE = (329439, 240)

def xySplit(lstrFolderPathPartition,
            strFolderPathOutputX,
            strFolderPathLabel,
            strFileHeader,
            strFilePathOutputY,
            l_observationWindow=l_observationWindow,
            n_split_separation=n_split_separation_trn,
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
    for strFileNameXY in tqdm(lFilenames):
        # Ignore OS files
        if strFileNameXY[-4:] != '.pkl':
            continue
        # Open the file, each containing a single timeseries
        with open(strFileNameXY, 'rb') as file:
            il = pickle.load(file)
        # Convert to relative time
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
            # later edit: why did I save it as a dict? Why not an array with 2 dims?
            features = {'t': il['t'][time_interval_mask],
                        'v': il['v'][time_interval_mask]}
            label = dfLabelMapper.loc[dfLabelMapper.index == il['y']].values[0,:]
            # We remove the absolute time information to prevent information leakage
            features['t'] = features['t'] - features['t'][0]
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

def load_GlobalFeatureSet1(pathData):
    # This function loads the data and computes the global features
    # The gobal features are custom features for this paper
    # just to get the array shape
    n_ts = 0
    for ifilename in os.listdir(pathData):
        if ifilename[-4:] != '.pkl':
            continue
        n_ts += 1
    # load the data
    l_dir = os.listdir(pathData)
    adata = np.empty((n_ts, 11))
    for ii in tqdm(range(n_ts)):
        ifilename = l_dir[ii]
        if ifilename[-4:] != '.pkl':
            continue
        with open(pathData+ifilename, 'rb') as f:
            idata = pickle.load(f)
        idata['v'] = np.where(idata['v']>1e20, 0, idata['v'])
        # statistical moments
        adata[ii,0] = idata['v'].mean()
        adata[ii,1] = idata['v'].std()
        adata[ii,2] = scipy.stats.skew(idata['v'])
        adata[ii,3] = scipy.stats.kurtosis(idata['v'])
        adata[ii,4] = np.sqrt((idata['v']*idata['v']).mean()) # RMS
        # order statistics
        adata[ii,5] = idata['v'].min()
        adata[ii,6] = idata['v'].max()
        adata[ii,7] = np.median(idata['v'])
        adata[ii,8] = np.quantile(idata['v'], .25)
        adata[ii,9] = np.quantile(idata['v'], .75)
        # window width
        adata[ii,10] =  (idata['t'][1:] - idata['t'][:-1]).mean()/1e9
    adata = np.nan_to_num(adata)
    # log and standardize required to make LR work
    adata = np.log(adata-adata.min()+np.e)
    u = adata.mean(0)
    s = adata.std(0)
    adata = (adata - u) / s
    return adata

def slsScaler(ax):
    # symmetric, log, standard scaling.
    ax = ax.copy()
    ax[ax>10] = np.log10(ax[ax>10])
    ax[ax<-10] = -np.log10(-ax[ax<-10])
    ax = (ax - MEAN) / STD
    return ax

def inverse_slsScaler(ax):
    # slsScaler invertor
    ax = ax.copy()
    ax = ax * STD + MEAN
    ax[ax>1] = 10**ax[ax>1]
    ax[ax<-1] = -(10**-ax[ax<-1])
    return ax

def save_lilrows(dense_array, filename=None):
    # This format save more space than the one-hot encoding
    # We use list-in-list sparse matrix format
    sparse_array = scipy.sparse.lil_array(dense_array)
     # We only get the rows, because all the datas are just TRUE, and a fixed number of rows (datapoints in the test set).
    sparse_array = sparse_array.rows
    # You can also use this function for just conversion without saving
    if filename is not None:
        pd.DataFrame(sparse_array).to_csv(filename)
    return sparse_array

def load_lilrows(filename, array_shape=ARRAY_SHAPE):
    # get the rows
    fromFile = pd.read_csv(filename, index_col=0).values[:,0] # this is an array of string
    fromFile = list(fromFile) # this is a list of string
    fromFile = repr(fromFile) # this is a string of string
    fromFile = fromFile.replace("\'","") # this is a string of numbers
    fromFile = ast.literal_eval(fromFile) # this is a list of list (this line is slow)
    # make the data
    loaded_data = []
    for i in fromFile:
        loaded_data_inner = []
        for j in i:
            loaded_data_inner.append(True)
        loaded_data.append(loaded_data_inner)
    # make the sparse array
    loaded_array = scipy.sparse.lil_array(np.zeros((array_shape), dtype=bool))
    loaded_array.rows[:] = fromFile
    loaded_array.data[:] = loaded_data
    loaded_array = loaded_array.todense()
    return loaded_array

def validate_metrics_calculation_input(y,h):
    if y.shape != h.shape:
        raise ValueError('y and h must have the same shape. Instead y is {} and h is {}'.format(y.shape, h.shape))
    if type(h) != np.ndarray:
        raise ValueError('h must be a numpy array. Instead it is {}'.format(type(h)))
    if not np.issubdtype(h.dtype, np.floating):
        raise ValueError('h must be a float array. Instead it is {}'.format(h.dtype))
    if h.min() < 0 or h.max() > 1:
        raise ValueError('h must be in the range of [0,1]. Instead it is in the range of [{},{}]'.format(h.min(), h.max()))

def mAP(y, h, n_threshold = 11, verbose=True):
    # mAP = mean Average Precision
    # logic from learn open cv website:
    # https://learnopencv.com/mean-average-precision-map-object-detection-model-evaluation-metric/

    validate_metrics_calculation_input(y,h)
    
    # calculate precision and recall for each threshold
    a_precision = np.zeros((n_threshold+2, y.shape[1]))
    a_recall = np.zeros((n_threshold+2, y.shape[1]))
    a_precision[:], a_recall[:] = np.NaN, np.NaN
    for ii in tqdm(range(0,n_threshold,1), disable=not verbose):
        threshold = ii/(n_threshold-1)
        h_ = h > threshold
        true__positive = np.sum(np.logical_and(y==1, h_), axis=0)
        # true__negative = np.sum(np.logical_and(y==-1, ~h_), axis=0) # not needed
        false_positive = np.sum(np.logical_and(y==-1, h_), axis=0)
        false_negative = np.sum(np.logical_and(y==1, ~h_), axis=0)
        assert np.isnan(true__positive).sum() == 0
        # assert np.isnan(true__negative).sum() == 0
        assert np.isnan(false_positive).sum() == 0
        assert np.isnan(false_negative).sum() == 0
        a_precision[ii,:] = true__positive / (true__positive + false_positive)
        a_recall[ii,:] = true__positive / (true__positive + false_negative)
    # a_recall[-2,:] = 0
    # a_precision[-2,:] = 0
    a_recall[-1,:] = 1 # only this is needed as the other are zero by default
    # a_precision[-1,:] = 0

    # NaN to zero due to divide by zero
    a_precision = np.nan_to_num(a_precision)
    a_recall = np.nan_to_num(a_recall)

    # sort recall-axis by increasing order
    isort = np.argsort(a_recall, axis=0)
    a_recall = a_recall[isort, np.arange(a_recall.shape[1])]
    a_precision = a_precision[isort, np.arange(a_recall.shape[1])]

    # make it a strictly decreasing function
    for i in range(a_precision.shape[0]-1,0,-1):
        # we are going from the right to the left
        current_value = a_precision[i,:]
        mask_smaller =  a_precision[i-1,:] < current_value
        a_precision[i-1, mask_smaller] = current_value[mask_smaller]

    # calculate average precision per class
    average_precision = np.trapz(a_precision, a_recall, axis=0)
    return average_precision.mean(),\
           {'precision': a_precision,
            'recall': a_recall,
            'AP':average_precision}

def allMetrics(y, h, threshold = .5, verbose = True, n_threshold = 11):
    validate_metrics_calculation_input(y, h)
    h_ = h > threshold
    # calculate class wise confusion matrix
    tp = np.sum(np.logical_and(y==1, h_), axis=0)
    tn = np.sum(np.logical_and(y==-1, ~h_), axis=0)
    fp = np.sum(np.logical_and(y==-1, h_), axis=0)
    fn = np.sum(np.logical_and(y==1, ~h_), axis=0)
    assert np.isnan(tp).sum() == 0
    assert np.isnan(tn).sum() == 0
    assert np.isnan(fp).sum() == 0
    assert np.isnan(fn).sum() == 0
    # calculate metrics per class
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    # NaN to zero due to divide by zero
    accuracy = np.nan_to_num(accuracy)
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1 = np.nan_to_num(f1)
    # calculate mAP
    mAP_, d_mAP = mAP(y, h, n_threshold, verbose)
    # print metrics
    d_return = {'tp': tp.tolist(),
                'tn': tn.tolist(),
                'fp': fp.tolist(),
                'fn': fn.tolist(),
                'accuracy': accuracy.tolist(),
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1': f1.tolist(),
                'AP': d_mAP['AP'].tolist(),
                '_AP_precision': d_mAP['precision'].tolist(),
                '_AP_recall' : d_mAP['recall'].tolist(),
                }
    if verbose:
        print('Accuracy'.center(12),
              'Precision'.center(12),
              'Recall'.center(12),
              'F1'.center(12),
              'mAP'.center(12), sep='|')
        print(str(round(accuracy.mean(),7)).rjust(12),
              str(round(precision.mean(),7)).rjust(12),
              str(round(recall.mean(),7)).rjust(12),
              str(round(f1.mean(),7)).rjust(12),
              str(round(mAP_,7)).rjust(12))
    return d_return

def parition_wrapper_for_metrics(dfy, h , partitions='combined',
                                 threshold = .5, verbose = True, n_threshold = 11):
    if partitions == 'combined':
        mask = np.ones(dfy.shape[0], dtype=bool)
    elif partitions == 'leaderboard':
        mask = np.array(['/tstLB/tstLB' in x for x in dfy['FileNameXY'].values])
    elif partitions == 'secret':
        mask = np.array(['/tstSC/tstSc' in x for x in dfy['FileNameXY'].values])
    else:
        raise ValueError('partitions must be either combined, leaderboard, or secret')
    return allMetrics(dfy.values[mask,3:],
                      h[mask,:],
                      threshold,
                      verbose,
                      n_threshold)

