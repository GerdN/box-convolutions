import re, os
import numpy as np
import pandas as pd
import pickle as pkl
import argparse
import torch
from os.path import basename, join

#import hdbscan

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_RUL(dataframe, Lifetime):
    return  Lifetime.loc[(dataframe['unit_id'])] - dataframe['cycle']


def RUL_by_parts(df, RUL=130):
    if df['RUL'] > RUL: return RUL
    if df['RUL'] <= RUL: return df['RUL']

parser = argparse.ArgumentParser(description='Script Variables')
parser.add_argument('--data_path', default='c:/Users/Nollmann/Anaconda3/envs/tf-gpu1/C-MAPSS_Problem-master/notebooks/data/', type=str,
                        help='Folder with txt files')
args = parser.parse_args()

datasets = []
path =  args.data_path # path to .txt files
text_files = [f for f in os.listdir(path) if f.endswith('.txt') and not f.startswith('r')]
dataframe = [os.path.splitext(f)[0] for f in text_files]
sensor_columns = ["sensor {}".format(s) for s in range(1, 22)]
#info_columns = ['dataset_id', 'unit_id', 'cycle', 'setting 1', 'setting 2', 'setting 3']
info_columns = ['unit_id', 'cycle', 'setting 1', 'setting 2', 'setting 3']
label_columns = ['unit_id', 'rul']
settings = ['setting 1', 'setting 2', 'setting 3']

test_data = []
train_data = []
RUL_data = []


for file in text_files:
    print(file)

    if re.match('RUL*', file):
        subset_df = pd.read_csv(path + file, delimiter=r"\s+", header=None)
        unit_id = range(1, subset_df.shape[0] + 1)
        subset_df.insert(0, 'unit_id', unit_id)
        #dataset_id = basename(file).split("_")[1][:5]
        #subset_df.insert(0, 'dataset_id', dataset_id)
        RUL_data.append(subset_df)

    if re.match('test*', file):
        subset_df = pd.read_csv(path + file, delimiter=r"\s+", header=None, usecols=range(26))
        #dataset_id = basename(file).split("_")[1][:5]
        #subset_df.insert(0, 'dataset_id', dataset_id)
        test_data.append(subset_df)

    if re.match('train*', file):
        subset_df = pd.read_csv(path + file, delimiter=r"\s+", header=None, usecols=range(26))
        #dataset_id = basename(file).split("_")[1][:5]
        #subset_df.insert(0, 'dataset_id', dataset_id)
        train_data.append(subset_df)

df_train = pd.concat(train_data, ignore_index=True)
df_train.columns = info_columns + sensor_columns
df_train.sort_values(by=['unit_id', 'cycle'], inplace=True)

df_test = pd.concat(test_data, ignore_index=True)
df_test.columns = info_columns + sensor_columns
df_test.sort_values(by=['unit_id', 'cycle'], inplace=True)

df_RUL = pd.concat(RUL_data, ignore_index=True)
df_RUL.columns = label_columns
df_RUL.sort_values(by=['unit_id'], inplace=True)

RUL_train = df_train.groupby(['unit_id'])['cycle'].max()
RUL_test = df_test.groupby(['unit_id'])['cycle'].max() + df_RUL.groupby(['unit_id'])[
    'rul'].max()

df_train['RUL'] = df_train.apply(lambda r: get_RUL(r, RUL_train), axis=1)
df_test['RUL'] = df_test.apply(lambda r: get_RUL(r, RUL_test), axis=1)

df_train['RUL'] = df_train.apply(lambda r: RUL_by_parts(r, 130), axis=1)
df_test['RUL'] = df_test.apply(lambda r: RUL_by_parts(r, 130), axis=1)

#clusterer = hdbscan.HDBSCAN(min_cluster_size=3000, prediction_data=True).fit(df_train[['setting 1', 'setting 2', 'setting 3']])

#NOL from https://towardsdatascience.com/keras-vs-pytorch-for-deep-learning-a013cb63870d
#For Pytorch, you’ll have to enable the GPU explicitly for every torch tensor and numpy variable. This clutters up the code and can be a bit error prone if you move back and forth between CPU and GPU for different operation.
#For example, to transfer our previous model to run on GPU we have to do the following:
# Get the GPU device
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Transfer the network to GPU
#clusterer.to(device)
# Transfer the inputs and labels to GPU
#df_train, df_test = df_train.to(device), df_test.to(device)

#train_labels, strengths = hdbscan.approximate_predict(clusterer, df_train[['setting 1', 'setting 2', 'setting 3']])
#test_labels, strengths = hdbscan.approximate_predict(clusterer, df_test[['setting 1', 'setting 2', 'setting 3']])

#df_train['HDBScan'] = train_labels
#df_test['HDBScan'] = test_labels

df_train.set_index(['unit_id'], inplace=True)
df_test.set_index(['unit_id'], inplace=True)

#pd.to_pickle(df_train, args.data_path + '/df_train_cluster_piecewise.pkl')
#pd.to_pickle(df_test, args.data_path + '/df_test_cluster_piecewise.pkl')

#df_train1 = df_train.drop('dataset_id', 1)
#df_test1 = df_test.drop('dataset.id', 1)
