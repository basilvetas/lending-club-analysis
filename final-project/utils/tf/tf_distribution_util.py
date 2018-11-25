from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
From: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distributions/python/ops/distribution_util.py
This is needed by hidden_markov_model, and doesn't have the function we need in r1.7.
I only left the functions we do need.
"""

# TODO add the copyright back

import functools
import hashlib
import numpy as np
# from tensorflow_probability.python.internal import dtype_util
import utils.tf.tf_dtype_util as dtype_util # fix
import tensorflow as tf



# these are not needed for us:
# from tensorflow_probability.python.internal import reparameterization
# from tensorflow.python.ops import control_flow_ops
# from tensorflow.python.util import tf_inspect

def static_value(x):
  """Returns the static value of a `Tensor` or `None`."""
  return tf.contrib.util.constant_value(tf.convert_to_tensor(x))

def prefer_static_rank(x):
  """Return static rank of tensor `x` if available, else `tf.rank(x)`.
  Args:
    x: `Tensor` (already converted).
  Returns:
    Numpy array (if static rank is obtainable), else `Tensor`.
  """
  return prefer_static_value(tf.rank(x))

def prefer_static_value(x):
  """Return static value of tensor `x` if available, else `x`.
  Args:
    x: `Tensor` (already converted).
  Returns:
    Numpy array (if static value is obtainable), else `Tensor`.
  """
  static_x = tf.contrib.util.constant_value(x)
  if static_x is not None:
    return static_x
  return x

def pick_scalar_condition(pred, true_value, false_value, name=None):
  """Convenience function that chooses one of two values based on the predicate.
  This utility is equivalent to a version of `tf.where` that accepts only a
  scalar predicate and computes its result statically when possible. It may also
  be used in place of `tf.cond` when both branches yield a `Tensor` of the same
  shape; the operational difference is that `tf.cond` uses control flow to
  evaluate only the branch that's needed, while `tf.where` (and thus
  this method) may evaluate both branches before the predicate's truth is known.
  This means that `tf.cond` is preferred when one of the branches is expensive
  to evaluate (like performing a large matmul), while this method is preferred
  when both branches are cheap, e.g., constants. In the latter case, we expect
  this method to be substantially faster than `tf.cond` on GPU and to give
  similar performance on CPU.
  Args:
    pred: Scalar `bool` `Tensor` predicate.
    true_value: `Tensor` to return if `pred` is `True`.
    false_value: `Tensor` to return if `pred` is `False`. Must have the
      same shape as `true_value`.
    name: Python `str` name given to ops managed by this object.
  Returns:
    result: a `Tensor` (or `Tensor`-convertible Python value) equal to
      `true_value` if `pred` evaluates to `True` and `false_value` otherwise.
      If the condition can be evaluated statically, the result returned is one
      of the input Python values, with no graph side effects.
  """
  with tf.name_scope(name, "pick_scalar_condition",
                     values=[pred, true_value, false_value]):
    pred_ = static_value(pred)
    if pred_ is None:
      return tf.where(pred, true_value, false_value)
    return true_value if pred_ else false_value

def move_dimension(x, source_idx, dest_idx):
  """Move a single tensor dimension within its shape.

  This is a special case of `tf.transpose()`, which applies
  arbitrary permutations to tensor dimensions.
  Args:
    x: Tensor of rank `ndims`.
    source_idx: Integer index into `x.shape` (negative indexing is
      supported).
    dest_idx: Integer index into `x.shape` (negative indexing is
      supported).
  Returns:
    x_perm: Tensor of rank `ndims`, in which the dimension at original
     index `source_idx` has been moved to new index `dest_idx`, with
     all other dimensions retained in their original order.
  Example:
  ```python
  x = tf.placeholder(shape=[200, 30, 4, 1, 6])
  x_perm = _move_dimension(x, 1, 1) # no-op
  x_perm = _move_dimension(x, 0, 3) # result shape [30, 4, 1, 200, 6]
  x_perm = _move_dimension(x, 0, -2) # equivalent to previous
  x_perm = _move_dimension(x, 4, 2) # result shape [200, 30, 6, 4, 1]
  ```
  """
  ndims = prefer_static_rank(x)
  dtype = dtype_util.common_dtype([source_idx, dest_idx],
                                  preferred_dtype=tf.int32)
  source_idx = tf.convert_to_tensor(source_idx, dtype=dtype)
  dest_idx = tf.convert_to_tensor(dest_idx, dtype=dtype)

  # Handle negative indexing.
  source_idx = pick_scalar_condition(
      source_idx < 0, ndims + source_idx, source_idx)
  dest_idx = pick_scalar_condition(
      dest_idx < 0, ndims + dest_idx, dest_idx)

  # Construct the appropriate permutation of dimensions, depending
  # whether the source is before or after the destination.
  def move_left_permutation():
    return prefer_static_value(
        tf.concat(
            [
                tf.range(0, dest_idx, dtype=dtype), [source_idx],
                tf.range(dest_idx, source_idx, dtype=dtype),
                tf.range(source_idx + 1, ndims, dtype=dtype)
            ],
            axis=0))

  def move_right_permutation():
    return prefer_static_value(
        tf.concat(
            [
                tf.range(0, source_idx, dtype=dtype),
                tf.range(source_idx + 1, dest_idx + 1, dtype=dtype),
                [source_idx],
                tf.range(dest_idx + 1, ndims, dtype=dtype)
            ],
            axis=0))

  def x_permuted():
    return tf.transpose(
        x,
        perm=tf.contrib.framework.smart_cond(
            source_idx < dest_idx,
            move_right_permutation,
            move_left_permutation))

  # One final conditional to handle the special case where source
  # and destination indices are equal.
  return tf.contrib.framework.smart_cond(
      tf.equal(source_idx, dest_idx), lambda: x, x_permuted)

