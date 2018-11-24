from os.path import dirname, realpath, join, exists
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from sklearn import preprocessing
import warnings

# we don't want warnings in our final notebook
warnings.filterwarnings('ignore')

cache_path = join(dirname(realpath(__file__)), '../cache/')
data_path = join(dirname(realpath(__file__)), '../data/')

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

		# rename the columns
		df_raw.rename(columns={'MOB': 'age_of_loan', 'PERIOD_END_LSTAT': 'loan_status', 'LOAN_ID': 'id'}, inplace=True)

		print(f'Caching...')
		with pd.HDFStore(df_raw_cache, mode='w') as store:
			store.append('df_raw', df_raw, data_columns=df_raw.columns, format='table')

		print(f'Fetching and caching raw data took {timer() - start:.2f} seconds')
	else:
		print(f'Loading raw data from hdf5 cache...')
		df_raw = pd.read_hdf(df_raw_cache, 'df_raw')
		print(f'Fetching raw data took {timer() - start:.2f} seconds')

	print(f'''Retrieved {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns''')
	return df_raw


def load_data_dic():
	""" load data dictionary """
	print('Loading data dictionary...')
	dic_df = pd.read_excel(join(data_path, 'LCDataDictionary.xlsx'))
	print(f'''Retrieved {dic_df.shape[0]:,} fields''')
	return dic_df


def reset_axes(matrix):
	"""
	for a given input nxn matrix as pandas dataframe, deletes axis names,
	sorts and renames indices and columns using our mapper and rounds digits
	"""

	# remove index names
	del matrix.index.name
	matrix = matrix.T
	del matrix.index.name
	matrix = matrix.T

	# sort axes
	matrix.sort_index(axis=0, inplace=True)
	matrix.sort_index(axis=1, inplace=True)

	# map loan status names
	matrix.rename(columns=loan_status_mapping, inplace=True)
	matrix.rename(index=loan_status_mapping, inplace=True)

	return matrix.round(2)


def preprocess(df):
	""" preprocess and cache df: clean fields and extract features """
	df_pre_cache = join(cache_path, 'df_pre.hdf')

	print(f'Mapping column names...')
	le = preprocessing.LabelEncoder()

	# store mapping of loan_status categorial names
	le.fit(df.term)
	global term_mapping
	term_mapping = dict(zip(le.transform(le.classes_), le.classes_))

	# store mapping of loan_status categorial names
	le.fit(df.loan_status)
	global loan_status_mapping
	loan_status_mapping = dict(zip(le.transform(le.classes_), le.classes_))

	start = timer()
	if not exists(df_pre_cache):
		print(f'Preprocessing data...')

		# filter out 60 month term loans or any 36 month term loans that went over 36 months
		df = df.loc[(df.term.astype(int) == 36) & (df.age_of_loan.astype(int) <= 36)]

		# encode loan_status and term
		df['term'] = le.fit_transform(df.term)
		df['loan_status'] = le.fit_transform(df.loan_status)

		print(f'Caching...')
		with pd.HDFStore(df_pre_cache, mode='w') as store:
			store.append('df_pre', df, data_columns= df.columns, format='table')

		print(f'Preprocessing and caching took {timer() - start:.2f} seconds')
	else:
		print(f'Loading preprocessed data from hdf5 cache...')
		df = pd.read_hdf(df_pre_cache, 'df_pre')
		print(f'Fetching preprocessed data took {timer() - start:.2f} seconds')

	print(f'''Preprocessed {df.shape[0]:,} rows, {df.shape[1]} columns''')
	return df


def split_data(df):
	"""
	formats df as pivot table to feed to edward inference and mcmc
	and also splits data into a training and test set for criticism
	"""
	df_split_cache = join(cache_path, 'split.hdf')

	start = timer()
	if not exists(df_split_cache):
		print('Pivoting and splitting data...')
		x_data = df.pivot(index='id', columns='age_of_loan', values='loan_status')

		# drop where 0 column is not null - this might be a data error, then drop the 0 column
		# and fill null values by propogating forward the last valid value
		x_data = x_data[x_data[0].isnull()].drop(0, axis=1).fillna(axis=1, method='ffill')

		# 90% train, 10% test
		train = np.random.rand(x_data.shape[0]) < 0.9
		x_train = x_data[train]
		x_test = x_data[~train]

		print(f'Caching...')
		with pd.HDFStore(df_split_cache, mode='w') as store:
			store.append('train', x_train, format='table')
			store.append('test', x_test, format='table')

		print(f'Pivoting, splitting and caching took {timer() - start:.2f} seconds')
	else:
		print(f'Loading training and test data from hdf5 cache...')
		x_train = pd.read_hdf(df_split_cache, 'train')
		x_test = pd.read_hdf(df_split_cache, 'test')
		print(f'Fetching training and test data took {timer() - start:.2f} seconds')

	print(f'''Training on {x_train.shape[0]:,} rows, {x_train.shape[1]} columns''')
	print(f'''Testing on {x_test.shape[0]:,} rows, {x_test.shape[1]} columns''')
	return x_train, x_test
