from os import getcwd
from os.path import join, exists
from timeit import default_timer as timer
import glob
import numpy as np
import pandas as pd
from sklearn import preprocessing

cache_path = f'{getcwd()}/cache/'
data_path = f'{getcwd()}/data/'

loan_statuses = pd.api.types.CategoricalDtype([
	    	'Charged Off',
	    	'Current',
	    	'Default',
	    	'Fully Paid',
	    	'In Grace Period',
	    	'Issued',
	    	'Late (16-30 days)',
	    	'Late (31-120 days)'])

loan_terms = pd.api.types.CategoricalDtype(['36', '60'])

dtypes = {'LOAN_ID': int,
          'PERIOD_END_LSTAT': loan_statuses,
          'MOB': int,
          'term': loan_terms }

parse_dates = ['RECEIVED_D', 'Month', 'IssuedDate']

def load_dataframe():
	""" we cache the df for faster dev workflow """
	df_raw_cache = join(cache_path, 'df_raw.hdf')

	start = timer() # cache saves us over a minute
	if not exists(df_raw_cache):
		print(f'Loading raw data from csv...')
		try:
			df_raw = pd.read_csv(join(data_path, 'PMTHIST_ALL_201811.csv'), usecols=dtypes.keys(), dtype=dtypes)
		except Exception:
			# deal with the two versions, temporary
			df_raw = pd.read_csv(join(data_path, 'PMTHIST_INVESTOR_201811.csv'), usecols=dtypes.keys(), dtype=dtypes)
		# chunksize = 1e6
		# for chunk in pd.read_csv(, chunksize=chunksize):
		# 	process(chunk)

		# csv_files = glob.glob(join(data_path, '*.csv'))
		# df_raw = pd.concat((pd.read_csv(f, header=1, low_memory=False) for f in csv_files))
		# df_raw.to_hdf(df_raw_cache, 'df_raw', mode='w')

		# rename the columns
		df_raw.rename(columns={'MOB': 'age_of_loan', 'PERIOD_END_LSTAT': 'loan_status', 'LOAN_ID': 'id'}, inplace=True)
		with pd.HDFStore(df_raw_cache, mode='w') as store:
			store.append('df_raw', df_raw, data_columns=df_raw.columns, format='table')

		print(f'Fetching and caching raw data took {timer() - start:.2f} seconds')
	else:
		print(f'Loading raw data from hdf5 cache...')
		df_raw = pd.read_hdf(df_raw_cache, 'df_raw')
		print(f'Fetching raw data took {timer() - start:.2f} seconds')

	print(df_raw.dtypes)
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

		# encode loan_status and term
		le = preprocessing.LabelEncoder()
		df['loan_status'] = le.fit_transform(df.loan_status)
		df['term'] = le.fit_transform(df.term)

		with pd.HDFStore(df_pre_cache, mode='w') as store:
			store.append('df_pre', df, data_columns= df.columns, format='table')

		print(f'Preprocessing and caching took {timer() - start:.2f} seconds')
	else:
		print(f'Loading preprocessed data from hdf5 cache...')
		df = pd.read_hdf(df_pre_cache, 'df_pre')
		print(f'Fetching preprocessed data took {timer() - start:.2f} seconds')

	print(f'''Preprocessed {df.shape[0]:,} rows, {df.shape[1]} columns''')
	return df

