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

	print(df_raw.dtypes)
	print(f'''Retrieved {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns''')
	return df_raw

def load_data_dic():
	""" load data dictionary """
	print('Loading data dictionary...')
	dic_df = pd.read_excel(join(data_path, 'LCDataDictionary.xlsx'))
	print(f'''Retrieved {dic_df.shape[0]:,} fields''')
	return dic_df

def transition_matrix(df):
	""" return transition matrix based on loan term and age of loan """
	transition_matrix_cache = join(cache_path, 'transition_matrix.hdf')

	start = timer()
	if not exists(transition_matrix_cache):
		print('Building transition matrix...')

		df['previous_month'] = df.age_of_loan - 1
		transitions =  pd.merge(df, df, left_on=['id', 'age_of_loan'], right_on=['id', 'previous_month'])
		transition_matrix = pd.crosstab(transitions['loan_status_x'], transitions['loan_status_y'])

		# if there were no transitions for given state, it will be missing so fill it in
		for i in range(df.loan_status.unique().shape[0]):
			if i not in transition_matrix.index:
				# if no row, create it and set to 0:
				print(f'Filling in empty row {i}...')
				transition_matrix.loc[i] = 0
			if i not in transition_matrix.columns:
				# if no column, create it and set to 0:
				print(f'Filling in empty column {i}...')
				transition_matrix[i] = 0

		transition_matrix.sort_index(axis=0, inplace=True)
		transition_matrix.sort_index(axis=1, inplace=True)
		transition_matrix.rename(columns=loan_status_mapping, inplace=True)
		transition_matrix.rename(index=loan_status_mapping, inplace=True)

		print(f'Caching...')
		with pd.HDFStore(transition_matrix_cache, mode='w') as store:
			store.append('matrix', transition_matrix, data_columns=transition_matrix.columns, format='table')

		print(f'Building transition matrix took {timer() - start:.2f} seconds')
	else:
		print(f'Loading transition matrix from hdf5 cache...')
		transition_matrix = pd.read_hdf(transition_matrix_cache, 'matrix')
		print(f'Fetching transition matrix took {timer() - start:.2f} seconds')

	return transition_matrix

def preprocess(df):
	""" preprocess and cache df: clean fields and extract features """
	df_pre_cache = join(cache_path, 'df_pre.hdf')

	print(f'Mapping transformations...')
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

	print(df.dtypes)
	print(f'''Preprocessed {df.shape[0]:,} rows, {df.shape[1]} columns''')
	return df

