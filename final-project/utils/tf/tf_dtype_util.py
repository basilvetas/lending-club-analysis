""" Adapted from https://github.com/tensorflow/probability """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

__all__ = [
        'common_dtype',
]


def common_dtype(args_list, preferred_dtype=None):
    """Returns explict dtype from `args_list` if there is one."""
    dtype = None
    while args_list:
        a = args_list.pop()
        if hasattr(a, 'dtype'):
            dt = tf.as_dtype(getattr(a, 'dtype')).base_dtype.as_numpy_dtype
        else:
            if isinstance(a, list):
                # Allows nested types, e.g. Normal([np.float16(1.0)], [2.0])
                args_list.extend(a)
            continue
        if dtype is None:
            dtype = dt
        elif dtype != dt:
            raise TypeError(f'Found incompatible dtypes, {dtype} and {dt}.')
    return preferred_dtype if dtype is None else tf.as_dtype(dtype)
