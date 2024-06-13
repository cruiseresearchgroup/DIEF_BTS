import numpy as np
import os
import pickle
import pandas as pd
from tqdm import tqdm

def load_pickles(directory, is_individual=False):
    if is_individual:
            datastreams = dict()
            for filename in tqdm(os.listdir(directory)):
                if filename.endswith('.pickle'):
                    filepath = os.path.join(directory, filename)
                    data = dict()
                    with open(filepath, 'rb') as f:
                        meta, timestamps, values = pickle.load(f)
                        meta = meta.split('.')[0]
                        site, streamid = meta[:6], meta[7:]
                        data['site'] = site
                        data['streamid'] = streamid
                        # convert to datetime
                        timestamps = pd.to_datetime(timestamps, unit='s')
                        data['timestamps'] = timestamps
                        data['values'] = values
                datastreams[int(filename.split('.')[0])] = data
            # sort the datastreams by key
            datastreams = dict(sorted(datastreams.items()))
            return datastreams
    else:
        with open(directory, 'rb') as f:
            data = pickle.load(f)
            return data


def clip_by_date(dstr, start_date, end_date, freq='10min'):
    '''
    :param dstr: loaded pickle dictionary
    :param start_date: np.datetime64
    :param end_date: np.datetime64
    :param freq: str, default='10min'
    :return: dictionary indexing by IoT point index
    '''
    df_index = pd.date_range(start=start_date, end=end_date, freq=freq)
    data = dict()
    if isinstance(list, type(dstr)):
        for i, point in enumerate(dstr):
            metadata, timestamps, values = point
            # timestamps = np.array(timestamps, dtype='datetime64[s]')
            timestamps = np.array(pd.to_datetime(timestamps).tz_localize(None), dtype='datetime64[s]')

            mask = (timestamps >= start_date) & (timestamps <= end_date)
            timestamps, values = timestamps[mask], values[mask]

            df = pd.DataFrame({'Timestamp': timestamps, 'Value': values})
            df.set_index('Timestamp', inplace=True)
            df = df.resample(freq).mean()
            df = df.reindex(df_index)
            data[i] = df
    else:
        for i in dstr:
            # Extract stream data
            timestamps, values = dstr[i]['timestamps'], dstr[i]['values']
            # timestamps = np.array(timestamps, dtype='datetime64[s]')
            timestamps = np.array(pd.to_datetime(timestamps).tz_localize(None), dtype='datetime64[s]')

            mask = (timestamps >= start_date) & (timestamps < end_date)
            timestamps, values = timestamps[mask], values[mask]

            df = pd.DataFrame({'Timestamp': timestamps, 'Value': values})
            df.set_index('Timestamp', inplace=True)
            df = df.resample(freq).mean()
            df = df.reindex(df_index)

            data[i] = df

    return data

def remove_outlier(data):
    # remove outliers and replace with mean
    for p_index in data:
        mean = data[p_index]['Value'].mean()
        std = data[p_index]['Value'].std()
        lower_limit = mean - 3 * std
        upper_limit = mean + 3 * std
        data[p_index]['Value'] = data[p_index]['Value'].apply(lambda x: mean if x < lower_limit or x > upper_limit else x)
    return data

def filter_by_ratio(data, threshold=0.01):
    unique_ratio = np.zeros(len(data))
    for p_index in data:
        unique_ratio[int(p_index)] = (data[p_index].nunique() / data[p_index].shape[0]).iloc[0]
    mask = (unique_ratio>threshold)
    idx = np.arange(len(data))
    return list(idx[mask])


def preprocess(data,
               ratio_threshold=0.01,
               fill_method='interpolate',
               save_name=None):
    '''
    Preprocesses data and returns a DataFrame.

    Parameters:
    - data: dictionary indexing by key, where each value is a list or array representing a column
    - fill_method: method for filling missing values, either 'interpolate' or 'nan_to_num'
    - save_name: name for saving the preprocessed DataFrame to a CSV file

    Returns:
    - df: preprocessed DataFrame
    '''
    assert fill_method in ['interpolate', 'nan_to_num'], "fill_method must be 'interpolate' or 'nan_to_num'"

    df = pd.DataFrame()

    data = remove_outlier(data)

    idx = filter_by_ratio(data, ratio_threshold)


    # Interpolation method
    if fill_method == 'interpolate':
        interpolated_data = list()
        for i in idx:
            interpolated_data.append(data[i].interpolate(method='polynomial', order=2).ffill().bfill())

        df = pd.concat(interpolated_data, axis=1)
    elif fill_method == 'nan_to_num':
        df = pd.DataFrame(np.nan_to_num(data))

    # set index
    df.index = data[0].index
    # rename columns
    df.columns = idx
    if save_name is not None:
        df.to_csv(f'{save_name}.csv', index=True)

    return df

