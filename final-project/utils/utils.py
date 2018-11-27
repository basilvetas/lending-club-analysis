import sys
from os import remove, environ
from os.path import dirname, realpath, join, exists
from timeit import default_timer as timer
import argparse
import glob
import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
import warnings

# we don't want warnings in our final notebook
warnings.filterwarnings('ignore')

# hid tensorflow info and warning messages
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

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
          'term': loan_terms}

parse_dates = ['RECEIVED_D', 'Month', 'IssuedDate']


def get_cache_or_execute(name, function, *args, **kwargs):
    """ Checks for cached df, otherwise
    runs function to generate df """
    cached_file = join(cache_path, f'{name}.hdf')
    ed_model = kwargs.get('ed_model', False)
    n_items = kwargs.pop('n_items', 1)
    vals = []

    names = [f'{name}_{i+1}' for i in range(n_items)]

    start = timer()
    if not exists(cached_file):

        # hdf5 caching kwargs
        data_columns = kwargs.pop('data_columns', False)
        _format = kwargs.pop('format', False)

        obj, *vals = function(*args, **kwargs)
        print(f'Caching {name} data...')

        if data_columns:
            kwargs['data_columns'] = obj.columns

        if _format:
            kwargs['format'] = _format

        with pd.HDFStore(cached_file, mode='w') as store:
            for i in range(n_items):
                name_ = names[i] if n_items > 1 else name
                obj_ = obj[i] if n_items > 1 else obj
                store.append(name_, obj_, **kwargs)
    else:
        print(f'Loading {name} data from cache...')

        obj = []
        for i in range(n_items):
            name_ = names[i] if n_items > 1 else name
            obj_ = pd.read_hdf(cached_file, name_)
            if n_items > 1:
                obj.append(obj_)
            else:
                obj = obj_

        if ed_model:
            print(f'Loading cached edward model...')
            try:
                sess = tf.Session()
                saver = tf.train.Saver()
                with sess.as_default():
                    saver.restore(sess, join(cache_path, f'{name}.ckpt'))

                vals.append(sess)
                vals.extend(list(ed_model.values()))
            except tf.errors.NotFoundError:
                print(f'Error: please re-run model and try again.')
                val = len(list(ed_model.values()))+1
                return obj, (*[None]*val)

    if isinstance(obj, pd.DataFrame):
        print(f'''Retrieved {obj.shape[0]:,} rows, '''
              f'''{obj.shape[1]} columns in {timer() - start:.2f} seconds''')
    else:
        print(f'Retrieved data in {timer() - start:.2f} seconds')
    return obj, (*vals)


def load_dataframe():
    """ we cache the df for faster dev workflow """
    def function():
        print('Loading raw data from csv...')
        try:
            df = pd.read_csv(join(data_path, 'PMTHIST_ALL_201811.csv'),
                             usecols=dtypes.keys(), dtype=dtypes)
        except Exception:
            # deal with the two versions, temporary
            df = pd.read_csv(join(data_path, 'PMTHIST_INVESTOR_201811.csv'),
                             usecols=dtypes.keys(), dtype=dtypes)

        # rename the columns
        df.rename(columns={'MOB': 'age_of_loan',
                           'PERIOD_END_LSTAT': 'loan_status',
                           'LOAN_ID': 'id'},
                  inplace=True)

        # remove outliers: loans that have a status for
        # month 0 (3% of the data)
        df = df[~(df.age_of_loan == 0)]
        # remove remaining loans that have status 'Issued',
        # there are only three of them, with 1 unique observation each:
        df = df[df.loan_status != 'Issued']
        return df, None

    kwargs = {'data_columns': True, 'format': 'table'}
    return get_cache_or_execute('raw', function, **kwargs)[0]


def load_data_dic():
    """ load data dictionary """
    print('Loading data dictionary...')
    dic_df = pd.read_excel(join(data_path, 'LCDataDictionary.xlsx'))
    print(f'''Retrieved {dic_df.shape[0]:,} fields''')
    return dic_df


def preprocess(df):
    """ preprocess and cache df: clean fields and extract features """
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
    # add additional status
    loan_status_mapping.update({len(loan_status_mapping): "Done"})

    def function(df):
        print(f'Preprocessing data...')

        # filter out 60 month term loans or any 36 month
        # term loans that went over 36 months
        df = df.loc[(df.term.astype(int) == 36) &
                    (df.age_of_loan.astype(int) <= 36)]

        # encode loan_status and term
        df['term'] = le.fit_transform(df.term)
        df['loan_status'] = le.fit_transform(df.loan_status)

        return df, None

    kwargs = {'format': 'table'}
    return get_cache_or_execute('preprocessed', function, df, **kwargs)[0]


def split_data(df):
    """
    formats df as pivot table to feed to edward inference and mcmc
    and also splits data into a training and test set for criticism
    """
    def function(df):
        print('Pivoting and splitting data...')
        x_data = df.pivot(index='id', columns='age_of_loan',
                          values='loan_status')

        # fill null values by propogating forward the last valid value
        # x_data = x_data.fillna(axis=1, method='ffill')
        # add a state "done" to extrapolate short loans
        new_status = df.loan_status.nunique()
        x_data = x_data.fillna(new_status, axis=1)
        return x_data, None

    kwargs = {'format': 'table'}
    x_data = get_cache_or_execute('split', function, df, **kwargs)[0]

    # 90% train, 10% test
    train = np.random.rand(x_data.shape[0]) < 0.9
    x_train = x_data[train]
    x_test = x_data[~train]
    print(f'Train: {x_train.shape} | Test: {x_test.shape}')
    return x_train, x_test


def pretty_matrix(matrix):
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

    return matrix.round(6)


def clear_cache(extensions):
    """ clears cached files of specified extension types """
    cached_files = []
    if not extensions:
        extensions = ['hdf', 'ckpt', 'pkl', 'tmp']
        cached_files.append(join(cache_path, 'checkpoint'))

    for ext in extensions:
        cached_files.extend(glob.glob(join(cache_path, f'*.{ext}*'),
                                      recursive=True))

    for file in cached_files:
        remove(file)
    return


def parse():
    """ command line arguments """
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '--clear_cache',
        '-c',
        nargs='*',
        type=str.lower,
        metavar='extension',
        help='Filetypes to clear from cache'
    )

    return parser.parse_args()


def get_counts_per_month(df, n_states):
    """
    given a matrix of shape (n_loans, chain_len),
    computes counts of each status per month
    to feed into the multinomial model
    """
    chain_len = df.shape[1]
    value_counts_per_month = [df[i+1].value_counts()
                              for i in range(chain_len)]
    counts_per_month = np.array([[vc[j] if (j in vc) else 0
                                 for j in range(n_states)]
                                 for vc in value_counts_per_month])
    return counts_per_month


if __name__ == '__main__':
    args = parse()
    if isinstance(args.clear_cache, list):
        print('Clearing cache...')
        clear_cache(args.clear_cache)

    sys.exit(0)
