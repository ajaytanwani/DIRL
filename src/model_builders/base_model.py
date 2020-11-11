import collections
import numpy as np
import tensorflow as tf

from abc import ABCMeta
from abc import abstractmethod

rt_shape_str = '_runtime_shapes'


class BaseModel(object):
  """Abstract Base Class for defining a model"""

  __metaclass__ = ABCMeta

  def __init__(self, model_config, is_training):

    self._model_config = model_config
    self._is_training = is_training
    # self._train_dir = train_dir
    # self._add_summaries = add_summaries

  @property
  def model_config(self):
    return self._model_config

  @property
  def is_training(self):
    return self._is_training

  @abstractmethod
  def preprocess(self, inputs):
    """preprocess the inputs

    Args:
      inputs: a [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: a [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.
    """
    pass

  @abstractmethod
  def predict(self, preprocessed_inputs):
    """Define the prediction model architecture and store them all in a prediction dictionary

    Args:
      preprocessed_inputs: a [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images.

    Returns:
      prediction_dict: a dictionary holding tensors for defining the loss function
    """

    pass

  @abstractmethod
  def postprocess(self, prediction_dict, **params):
    """Convert predicted tensors to final outputs back in the image space.

    State the conventions used. For object recognition:
    * Classes are integers in [0, num_classes); background classes are removed
      and the first non-background class is mapped to 0. If the model produces
      class-agnostic detections, then no output is produced for classes.
    * Boxes are to be interpreted as being in [y_min, x_min, y_max, x_max]
      format and normalized relative to the image window.
    * `num_detections` is provided for settings where detections are padded to a
      fixed number of boxes.
    * We do not specifically assume any kind of probabilistic interpretation
      of the scores --- the only important thing is their relative ordering.
      Thus implementations of the postprocess function are free to output
      logits, probabilities, calibrated probabilities, or anything else.

    Args:
      prediction_dict: a dictionary holding prediction tensors.
      **params: additional keyword arguments for specific implementation of task

    Returns:
      output_dictionary: a dictionary holding the output of the model. For object recognition:
        detection_boxes: [batch, max_detections, 4]
        detection_scores: [batch, max_detections]
        detection_classes: [batch, max_detections]
          (If a model is producing class-agnostic detections, this field may be
          missing)
        instance_masks: [batch, max_detections, image_height, image_width]
          (optional)
        keypoints: [batch, max_detections, num_keypoints, 2] (optional)
        num_detections: [batch]
    """

    pass

  @abstractmethod
  def loss(self, prediction_dict, groundtruth_dict):
    """Compute scalar loss tensors with respect to provided groundtruth.

    Calling this function requires that groundtruth tensors have been
    provided via the provide_groundtruth function.

    Args:
      prediction_dict: a dictionary holding predicted tensors
      groundtruth_dict: a dictionary holding groundtruth tensors for computing the loss

    Returns:
      a dictionary mapping strings (loss names) to scalar tensors representing
        loss values.
    """
    pass

  @abstractmethod
  def create_groundtruth_dictionary(self, class_labels_list, domain_labels_list=None):
    """Store the groundtruth information in a dictionary

    Args:
      class_labels_list: tuple of tensors containing the class labels in a given batch.
      domain_labels_list: tuple of tensors containing the domain labels in a give batch
    """

    pass

  # @abstractmethod
  # def train(self, train_tensor, train_dir):
  #   """Initialize training at the output of the tensor
  #
  #   Args:
  #     train_tensor: tensor defining the loss function
  #     train_dir: output directory where training checkpoints are stored.
  #   """
  #
  #   pass


class BatchQueue(object):

  def __init__(self, tensor_dict, batch_size, batch_queue_capacity, num_batch_queue_threads, prefetch_queue_capacity):
    """Constructs a batch queue holding tensor_dict.

    Args:
      tensor_dict: dictionary of tensors to batch.
      batch_size: size of the training batch.
      batch_queue_capacity: max capacity of queue from which the tensors are batched.
      num_batch_queue_threads: number of threads to use for batching.
      prefetch_queue_capacity: max capacity of the queue used to prefetch assembled batches.

    Returns:
      input_queue: a FIFO prefetcher queue
    """

    self._batch_size = batch_size
    self._static_shapes = collections.OrderedDict(
      {key: tensor.get_shape() for key, tensor in tensor_dict.items()})
    # Remember runtime shapes to unpad tensors after batching.
    runtime_shapes = collections.OrderedDict(
      {(key + rt_shape_str): tf.shape(tensor)
       for key, tensor in tensor_dict.items()})

    tensor_dict.update(runtime_shapes)

    batched_tensors = tf.train.batch(tensor_dict, capacity=batch_queue_capacity,
                                     batch_size=batch_size, dynamic_pad=True,
                                     num_threads=num_batch_queue_threads)

    names = list(batched_tensors.keys())
    dtypes = [t.dtype for t in batched_tensors.values()]
    shapes = [t.get_shape() for t in batched_tensors.values()]

    prefetch_queue = tf.PaddingFIFOQueue(prefetch_queue_capacity, dtypes=dtypes,
                                         shapes=shapes,
                                         names=names,
                                         name='prefetch_queue')
    enqueue_op = prefetch_queue.enqueue(batched_tensors)

    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      prefetch_queue, [enqueue_op]))

    tf.summary.scalar('queue/%s/fraction_of_%d_full' % (prefetch_queue.name,
                                                        prefetch_queue_capacity),
                      tf.to_float(prefetch_queue.size()) * (1. / prefetch_queue_capacity))

    self._queue = prefetch_queue

  def dequeue(self):
    """Dequeues a batch of tensor_dict from the batch_queue method.

    Returns:
      tensor_dict_list: a list containing tensor dictonaries from the input queue.
    """

    assert self._batch_size is not None

    assert self._static_shapes is not None

    assert self._queue is not None

    batched_tensors = self._queue.dequeue()
    # Separate input tensors from tensors containing their runtime shapes.
    tensors = {}
    shapes = {}
    for key, batched_tensor in batched_tensors.items():
      unbatched_tensor_list = tf.unstack(batched_tensor)
      for i, unbatched_tensor in enumerate(unbatched_tensor_list):
        if rt_shape_str in key:
          shapes[(key[:-len(rt_shape_str)], i)] = unbatched_tensor
        else:
          tensors[(key, i)] = unbatched_tensor

    # Undo that padding using shapes and create a list of size `batch_size` that
    # contains tensor dictionaries.
    tensor_dict_list = []
    batch_size = self._batch_size
    for batch_id in range(batch_size):
      tensor_dict = {}
      for key in self._static_shapes:
        tensor_dict[key] = tf.slice(tensors[(key, batch_id)],
                                    tf.zeros_like(shapes[(key, batch_id)]),
                                    shapes[(key, batch_id)])
        tensor_dict[key].set_shape(self._static_shapes[key])
      tensor_dict_list.append(tensor_dict)

    return tensor_dict_list


class SharedBatchQueue(object):

  def __init__(self, source_tensor_dict, target_tensor_dict, batch_size, batch_queue_capacity, num_batch_queue_threads,
               prefetch_queue_capacity):

    self.source_queue = BatchQueue(source_tensor_dict, int(batch_size * 0.5),
                                   batch_queue_capacity,
                                   num_batch_queue_threads,
                                   prefetch_queue_capacity)

    self.target_queue = BatchQueue(target_tensor_dict, int(batch_size * 0.5),
                                   batch_queue_capacity,
                                   num_batch_queue_threads,
                                   prefetch_queue_capacity)

  def dequeue(self):
    batched_tensors_list = self.source_queue.dequeue()
    batched_tensors_list.extend(self.target_queue.dequeue())
    return batched_tensors_list
