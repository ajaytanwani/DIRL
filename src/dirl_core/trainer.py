import numpy as np
import tensorflow as tf

from abc import ABCMeta
from abc import abstractmethod

class Trainer(object):

  __metaclass__ = ABCMeta

  def __init__(self, dataset_fn, model_fn, train_config, train_dir):


    self._dataset_fn = dataset_fn

    self._model_fn = model_fn

    self._train_config = train_config

    self._train_dir = train_dir

  @abstractmethod
  def trainer_routine(self, dataset_fn, model_fn, train_config, train_dir):
    """Initialize the training job.

    Args:
      dataset_fn: A T object containing source_iterator, target_iterator, dataset_config, source and target datasets.
      model_fn:
      model_config: config file for training routine.
      train_dir:
    """

    pass
    # with tf.Graph().as_default():

      ## dequeue the iterator
      # input_queue = create_input_queue(train_config[])

      # preprocess the data

      # define the training graph

      # define the loss function

      # start the training routine
