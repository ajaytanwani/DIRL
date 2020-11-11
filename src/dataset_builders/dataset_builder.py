from dataset_builders import mnist_dataset_builder
from dirl_core import dirl_utils

import numpy as np
import tensorflow as tf


class BaseDataset(object):
  """Abstract Base Class for defining a model"""

  def __init__(self, dataset_config, source_dataset, target_dataset, source_iterator=None, target_iterator=None):

    self._dataset_config = dataset_config

    self._source_dataset = source_dataset

    self._source_iterator = source_iterator

    self._target_dataset = target_dataset

    self._target_iterator = target_iterator

  @property
  def dataset_config(self):
    return self._dataset_config

  @property
  def source_dataset(self):
    return self._source_dataset

  @property
  def source_iterator(self):
    return self._source_iterator

  @property
  def target_dataset(self):
    return self._target_dataset

  @property
  def target_iterator(self):
    return self._target_iterator

  @staticmethod
  def build_dataset(dataset_config, output_directory):
    """Build the dataset from the dataset_config
    """
    # import ipdb; ipdb.set_trace()
    if dataset_config['source_dataset_name'] == "MNIST_Data":
      source_dataset = mnist_dataset_builder.MNISTDatasetBuilder.read_dataset(dataset_config['source_dataset_name'])

      # write tfrecords of source_dataset to output_directory
      mnist_dataset_builder.MNISTDatasetBuilder.write_tfrecords(source_dataset, output_directory, domain='source')

    else:
      raise ValueError("Building dataset method is not defined for: %s", dataset_config['source_dataset_name'])

    target_dataset = None
    if 'target_dataset_name' in dataset_config.keys():
      if dataset_config['target_dataset_name'] == "MNIST_Data":
        target_dataset = mnist_dataset_builder.MNISTDatasetBuilder.read_dataset(dataset_config['target_dataset_name'])

        mnist_dataset_builder.MNISTDatasetBuilder.write_tfrecords(target_dataset, output_directory, domain='target')

        target_iterator = mnist_dataset_builder.MNISTDatasetBuilder.make_initializable_iterator_from_tfrecords(
          output_directory, dataset_config)

    return BaseDataset(dataset_config, source_dataset, target_dataset)

  @staticmethod
  def create_dataset_iterators(dataset_config, input_directory):
    """Create iterators from tfrecords in the input directory.

    Args:
      input_director: path to the directory where tfrecords for training are located.
      dataset_config: A T object containing config parameters for parsing the dataset.
    """
    if dataset_config['source_dataset_name'] == "MNIST_Data":
      source_iterator = mnist_dataset_builder.MNISTDatasetBuilder.make_initializable_iterator_from_tfrecords(
        input_directory, dataset_config, domain='source')
    else:
      raise ValueError("Building dataset method is not defined for: %s", dataset_config['source_dataset_name'])

    target_iterator = None
    if 'target_dataset_name' in dataset_config.keys():
      if dataset_config['target_dataset_name'] == "MNIST_Data":
        target_iterator = mnist_dataset_builder.MNISTDatasetBuilder.make_initializable_iterator_from_tfrecords(
          input_directory, dataset_config, domain='target')

    return source_iterator, target_iterator