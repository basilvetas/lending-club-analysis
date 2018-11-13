from os import getcwd
from os.path import join, exists
from timeit import default_timer as timer
import glob
import numpy as np
import pandas as pd

cache_path = f'{getcwd()}/cache/'
data_path = f'{getcwd()}/data/'

def load_dataframe():
	""" we cache the df for faster dev workflow """
	df_raw_cache = join(cache_path, 'df_raw.hdf')

	start = timer() # cache saves us over a minute
	if not exists(df_raw_cache):
	    print(f'Loading data from csvs...')
	    csv_files = glob.glob(join(data_path, '*.csv'))
	    df_raw = pd.concat((pd.read_csv(f, header=1, low_memory=False) for f in csv_files))
	    df_raw.to_hdf(df_raw_cache, 'df_raw', mode='w')
	else:
	    print(f'Loading data from cache...')
	    df_raw = pd.read_hdf(df_raw_cache, 'df_raw')

	print(f'Fetching data took {timer() - start:.2f} seconds')
	print(f'''Retrieved {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns''')
	return df_raw

def load_data_dic():
	""" load data dictionary """
	print('Loading data dictionary...')
	return pd.read_excel(join(data_path, 'LCDataDictionary.xlsx'))



