import collections
import cv2
import json
import numpy as np
import os
import re

import tensorflow as tf
import xml.etree.ElementTree as et
from xml.dom import minidom
import yaml
from yaml.constructor import ConstructorError
from mpl_toolkits.axes_grid1 import ImageGrid
from tensorflow.python.framework import ops

import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

import math
import copy

# # import tensorflow as tf
# import numpy as np
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.axes_grid1 import ImageGrid
#
#
# # Model construction utilities below adapted from
# # https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html#deep-mnist-for-experts
# def weight_variable(shape):
#   initial = tf.truncated_normal(shape, stddev=0.1)
#   return tf.Variable(initial)
#
#
# def bias_variable(shape):
#   initial = tf.constant(0.1, shape=shape)
#   return tf.Variable(initial)
#
#
# def conv2d(x, W):
#   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
#
# def max_pool_2x2(x):
#   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1], padding='SAME')
#
#
# def shuffle_aligned_list(data):
#   """Shuffle arrays in a list by shuffling each array identically."""
#   num = data[0].shape[0]
#   p = np.random.permutation(num)
#   return [d[p] for d in data]
#
#
# def batch_generator(data, batch_size, shuffle=True, iter_shuffle=False):
#   """Generate batches of data.
#
#   Given a list of array-like objects, generate batches of a given
#   size by yielding a list of array-like objects corresponding to the
#   same slice of each input.
#   """
#
#   num = data[0].shape[0]
#   if shuffle:
#     data = shuffle_aligned_list(data)
#
#   batch_count = 0
#   while True:
#     if batch_count * batch_size + batch_size >= num:
#       batch_count = 0
#       if iter_shuffle:
#         print("batch shuffling")
#         data = shuffle_aligned_list(data)
#
#     start = batch_count * batch_size
#     end = start + batch_size
#     batch_count += 1
#     yield [d[start:end] for d in data]
#
#
# def imshow_grid(images, shape=[2, 8]):
#   """Plot images in a grid of a given shape."""
#   fig = plt.figure(1)
#   grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)
#
#   size = shape[0] * shape[1]
#   for i in range(size):
#     grid[i].axis('off')
#     grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.
#
#   plt.show()
#
#
# def plot_embedding(X, y, d, title=None):
#   """Plot an embedding X with the class label y colored by the domain d."""
#   x_min, x_max = np.min(X, 0), np.max(X, 0)
#   X = (X - x_min) / (x_max - x_min)
#
#   # Plot colors numbers
#   plt.figure(figsize=(10, 10))
#   ax = plt.subplot(111)
#   for i in range(X.shape[0]):
#     # plot colored number
#     plt.text(X[i, 0], X[i, 1], str(y[i]),
#              color=plt.cm.bwr(d[i] / 1.),
#              fontdict={'weight': 'bold', 'size': 9})
#
#   plt.xticks([]), plt.yticks([])
#   if title is not None:
#     plt.title(title)

###############
def static_or_dynamic_map_fn(fn, elems, dtype=None,
                             parallel_iterations=32, back_prop=True):
  """Runs map_fn as a (static) for loop when possible.

  This function rewrites the map_fn as an explicit unstack input -> for loop
  over function calls -> stack result combination.  This allows our graphs to
  be acyclic when the batch size is static.
  For comparison, see https://www.tensorflow.org/api_docs/python/tf/map_fn.

  Note that `static_or_dynamic_map_fn` currently is not *fully* interchangeable
  with the default tf.map_fn function as it does not accept nested inputs (only
  Tensors or lists of Tensors).  Likewise, the output of `fn` can only be a
  Tensor or list of Tensors.

  TODO(jonathanhuang): make this function fully interchangeable with tf.map_fn.

  Args:
    fn: The callable to be performed. It accepts one argument, which will have
      the same structure as elems. Its output must have the
      same structure as elems.
    elems: A tensor or list of tensors, each of which will
      be unpacked along their first dimension. The sequence of the
      resulting slices will be applied to fn.
    dtype:  (optional) The output type(s) of fn. If fn returns a structure of
      Tensors differing from the structure of elems, then dtype is not optional
      and must have the same structure as the output of fn.
    parallel_iterations: (optional) number of batch items to process in
      parallel.  This flag is only used if the native tf.map_fn is used
      and defaults to 32 instead of 10 (unlike the standard tf.map_fn default).
    back_prop: (optional) True enables support for back propagation.
      This flag is only used if the native tf.map_fn is used.

  Returns:
    A tensor or sequence of tensors. Each tensor packs the
    results of applying fn to tensors unpacked from elems along the first
    dimension, from first to last.
  Raises:
    ValueError: if `elems` a Tensor or a list of Tensors.
    ValueError: if `fn` does not return a Tensor or list of Tensors
  """
  if isinstance(elems, list):
    for elem in elems:
      if not isinstance(elem, tf.Tensor):
        raise ValueError('`elems` must be a Tensor or list of Tensors.')

    elem_shapes = [elem.shape.as_list() for elem in elems]
    # Fall back on tf.map_fn if shapes of each entry of `elems` are None or fail
    # to all be the same size along the batch dimension.
    for elem_shape in elem_shapes:
      if (not elem_shape or not elem_shape[0]
          or elem_shape[0] != elem_shapes[0][0]):
        return tf.map_fn(fn, elems, dtype, parallel_iterations, back_prop)
    arg_tuples = zip(*[tf.unstack(elem) for elem in elems])
    outputs = [fn(arg_tuple) for arg_tuple in arg_tuples]
  else:
    if not isinstance(elems, tf.Tensor):
      raise ValueError('`elems` must be a Tensor or list of Tensors.')
    elems_shape = elems.shape.as_list()
    if not elems_shape or not elems_shape[0]:
      return tf.map_fn(fn, elems, dtype, parallel_iterations, back_prop)
    outputs = [fn(arg) for arg in tf.unstack(elems)]
  # Stack `outputs`, which is a list of Tensors or list of lists of Tensors
  if all([isinstance(output, tf.Tensor) for output in outputs]):
    return tf.stack(outputs)
  else:
    if all([isinstance(output, list) for output in outputs]):
      if all([all(
          [isinstance(entry, tf.Tensor) for entry in output_list])
              for output_list in outputs]):
        return [tf.stack(output_tuple) for output_tuple in zip(*outputs)]
  raise ValueError('`fn` should return a Tensor or a list of Tensors.')

# Model construction utilities below adapted from
# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html#deep-mnist-for-experts
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def shuffle_aligned_list(data):
  """Shuffle arrays in a list by shuffling each array identically."""
  num = data[0].shape[0]
  p = np.random.permutation(num)
  return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True, iter_shuffle=False):
  """Generate batches of data.
`
  Given a list of array-like objects, generate batches of a given
  size by yielding a list of array-like objects corresponding to the
  same slice of each input.
  """

  num = data[0].shape[0]
  if shuffle:
    data = shuffle_aligned_list(data)

  batch_count = 0
  while True:
    if batch_count * batch_size + batch_size >= num:
      batch_count = 0
      if iter_shuffle:
        print("batch shuffling")
        data = shuffle_aligned_list(data)

    start = batch_count * batch_size
    end = start + batch_size
    batch_count += 1
    yield [d[start:end] for d in data]


###############

def GetFilesRecursively(topdir):
  """Gets all records recursively for some topdir.

  Args:
    topdir: String, path to top directory.
  Returns:
    allpaths: List of Strings, full paths to all leaf records.
  Raises:
    ValueError: If there are no files found for this directory.
  """
  assert topdir
  topdir = os.path.expanduser(topdir)
  allpaths = []
  for path, _, leaffiles in tf.gfile.Walk(topdir):
    if leaffiles:
      allpaths.extend([os.path.join(path, i) for i in leaffiles])
  if not allpaths:
    raise ValueError('No files found for top directory %s' % topdir)
  return allpaths


def NoDuplicatesConstructor(loader, node, deep=False):
  """Check for duplicate keys."""
  mapping = {}
  for key_node, value_node in node.value:
    key = loader.construct_object(key_node, deep=deep)
    value = loader.construct_object(value_node, deep=deep)
    if key in mapping:
      raise ConstructorError('while constructing a mapping', node.start_mark,
                             'found duplicate key (%s)' % key,
                             key_node.start_mark)
    mapping[key] = value
  return loader.construct_mapping(node, deep)


def WriteConfigAsYaml(config, logdir, filename):
  """Writes a config dict as yaml to logdir/experiment.yml."""
  if not tf.gfile.Exists(logdir):
    tf.gfile.MakeDirs(logdir)
  config_filename = os.path.join(logdir, filename)
  with tf.gfile.GFile(config_filename, 'w') as f:
    f.write(yaml.dump(config))
  tf.logging.info('wrote config to %s', config_filename)


def LoadConfigDict(config_paths, model_params):
  """Loads config dictionary from specified yaml files or command line yaml."""

  # Ensure that no duplicate keys can be loaded (causing pain).
  yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                       NoDuplicatesConstructor)

  # Handle either ',' or '#' separated config lists, since borg will only
  # accept '#'.
  sep = ',' if ',' in config_paths else '#'

  # Load flags from config file.
  final_config = {}
  if config_paths:
    for config_path in config_paths.split(sep):
      config_path = config_path.strip()
      if not config_path:
        continue
      config_path = os.path.abspath(config_path)
      tf.logging.info('Loading config from %s', config_path)
      with tf.gfile.GFile(config_path.strip()) as config_file:
        config_flags = yaml.load(config_file)
        final_config = DeepMergeDict(final_config, config_flags)
  if model_params:
    model_params = MaybeLoadYaml(model_params)
    final_config = DeepMergeDict(final_config, model_params)
  tf.logging.info('Final Config:\n%s', yaml.dump(final_config))
  return final_config


def MaybeLoadYaml(item):
  """Parses item if it's a string. If it's a dictionary it's returned as-is."""
  if isinstance(item, six.string_types):
    return yaml.load(item)
  elif isinstance(item, dict):
    return item
  else:
    raise ValueError('Got {}, expected YAML string or dict', type(item))


def DeepMergeDict(dict_x, dict_y, path=None):
  """Recursively merges dict_y into dict_x."""
  if path is None: path = []
  for key in dict_y:
    if key in dict_x:
      if isinstance(dict_x[key], dict) and isinstance(dict_y[key], dict):
        DeepMergeDict(dict_x[key], dict_y[key], path + [str(key)])
      elif dict_x[key] == dict_y[key]:
        pass  # same leaf value
      else:
        dict_x[key] = dict_y[key]
    else:
      dict_x[key] = dict_y[key]
  return dict_x


def ParseConfigsToLuaTable(config_paths, extra_model_params=None,
                           save=False, save_name='final_training_config.yml',
                           logdir=None):
  """Maps config_paths and extra_model_params to a Luatable-like object."""
  # Parse config dict from yaml config files / command line flags.
  config = LoadConfigDict(config_paths, extra_model_params)
  if save:
    WriteConfigAsYaml(config, logdir, save_name)
  # Convert config dictionary to T object with dot notation.
  config = RecursivelyConvertToLuatable(config)
  return config


def RecursivelyConvertToLuatable(yaml_dict):
  """Converts a dictionary to a LuaTable-like T object."""
  if isinstance(yaml_dict, dict):
    yaml_dict = T(yaml_dict)
  # for key, item in yaml_dict.iteritems():
  for key, item in yaml_dict.items():
    if isinstance(item, dict):
      yaml_dict[key] = RecursivelyConvertToLuatable(item)
  return yaml_dict


class T(object):
  """Class for emulating lua tables. A convenience class replicating some lua table syntax with a python dict.

  In general, should behave like a dictionary except that we can use dot notation
  to access keys. Users should be careful to only provide keys suitable for
  instance variable names.

  Nota bene: do not use the key "keys" since it will collide with the method keys.

  Usage example:

  >> t = T(a=5,b='kaw', c=T(v=[],x=33))
  >> t.a
  5
  >> t.z = None
  >> print t
  T(a=5, z=None, c=T(x=33, v=[]), b='kaw')

  >> t2 = T({'h':'f','x':4})
  >> t2
  T(h='f', x=4)
  >> t2['x']
  4.
  """
  def __init__(self, *args, **kwargs):
    if len(args) > 1 or (len(args) == 1 and len(kwargs) > 0):
      errmsg = '''constructor only allows a single dict as a positional argument or keyword arguments'''
      raise ValueError(errmsg)
    if len(args) == 1 and isinstance(args[0], dict):
      self.__dict__.update(args[0])
    else:
      self.__dict__.update(kwargs)

  def __repr__(self):
    fmt = ', '.join('%s=%s' for i in range(len(self.__dict__)))
    kwargstr = fmt % tuple(
      x for tup in self.__dict__.items() for x in [str(tup[0]), repr(tup[1])])
    return 'T(' + kwargstr + ')'

  def __getitem__(self, key):
    return self.__dict__[key]

  def __setitem__(self, key, val):
    self.__dict__[key] = val

  def __delitem__(self, key):
    del self.__dict__[key]

  def __iter__(self):
    return iter(self.__dict__)

  def __len__(self):
    return len(self.__dict__)

  def keys(self):  # Needed for dict(T( ... )) to work.
    return self.__dict__.keys()

  def iteritems(self):
    return self.__dict__.iteritems()

  def items(self):
    return self.__dict__.items()


class FlipGradientBuilder(object):
  def __init__(self):
    self.num_calls = 0

  def __call__(self, x, l=1.0):
    grad_name = "FlipGradient%d" % self.num_calls

    @ops.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
      return [tf.negative(grad) * l]

    g = tf.get_default_graph()
    with g.gradient_override_map({"Identity": grad_name}):
      y = tf.identity(x)

    self.num_calls += 1
    return y
