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
	    print(f'Loading raw data from csvs...')
	    csv_files = glob.glob(join(data_path, '*.csv'))
	    df_raw = pd.concat((pd.read_csv(f, header=1, low_memory=False) for f in csv_files))
	    df_raw.to_hdf(df_raw_cache, 'df_raw', mode='w')
	    print(f'Fetching and caching raw data took {timer() - start:.2f} seconds')
	else:
	    print(f'Loading raw data from cache...')
	    df_raw = pd.read_hdf(df_raw_cache, 'df_raw')
	    print(f'Fetching raw data from cache took {timer() - start:.2f} seconds')

	print(f'''Retrieved {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns''')
	return df_raw

def load_data_dic():
	""" load data dictionary """
	print('Loading data dictionary...')
	dic_df = pd.read_excel(join(data_path, 'LCDataDictionary.xlsx'))
	print(f'''Retrieved {dic_df.shape[0]:,} fields''')
	return dic_df

def preprocess(df):
	""" preprocess and cache df: clean fields and extract features """
	df_pre_cache = join(cache_path, 'df_pre.hdf')

	start = timer()
	if not exists(df_pre_cache):
		print(f'Preprocessing data...')
		# drop null and remove non-digits from term
		df.dropna(subset=['term'], inplace=True)
		df.term = df.term.str.replace(r'\D+', '').astype(int)

		# convert date fields to datetime and extract age
		df.issue_d = pd.to_datetime(df.issue_d)
		q4_startdate = pd.Timestamp('2018-10-01 00:00:00')
		df['age_in_days'] = q4_startdate - df.issue_d
		df['age_in_months'] = (q4_startdate.to_period('M') - df.issue_d.dt.to_period('M')).astype(int)
		df.to_hdf(df_pre_cache, 'df_pre', mode='w')
		print(f'Preprocessing and caching took {timer() - start:.2f} seconds')
	else:
		print(f'Loading preprocessed data from cache...')
		df = pd.read_hdf(df_pre_cache, 'df_pre')
		print(f'Fetching preprocessed data took {timer() - start:.2f} seconds')

	print(f'''Preprocessed {df.shape[0]:,} rows, {df.shape[1]} columns''')
	return df

