import functools
import numpy as np
import os
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from dirl_core import dirl_utils
from dataset_builders import dataset_utils

slim = tf.contrib.slim

# Keys used to annotate Tfrecords
IMAGE_KEY = 'image'
CLASS_LABEL_KEY = 'label'
DOMAIN_LABEL_KEY = 'source_id'

def _write_to_tfrecord(data, labels, filename, source_id, num_examples=None):
  """Tfrecord write function for MNIST dataset.

  data: np data array of shape [num_images, width, height, num_channels]
  labels: one_hot encoding of class labels [num_images, num_class]
  """

  if num_examples is None:
    num_examples = data.shape[0]

  with tf.python_io.TFRecordWriter(filename) as tfrecord_writer:
    with tf.Graph().as_default():
      image = tf.placeholder(dtype=tf.uint8, shape=data.shape[1:])
      encoded_png = tf.image.encode_png(image)

      with tf.Session('') as sess:
        for image_index in range(num_examples):
          data_write = np.concatenate([data[image_index], data[image_index], data[image_index]], 3)

          png_string = sess.run(encoded_png, feed_dict={image: data_write})

          example = dataset_utils.image_to_tfexample(
            png_string, 'png'.encode(), data.shape[1], data.shape[2], np.argmax(labels[image_index], axis=0), source_id)
          tfrecord_writer.write(example.SerializeToString())

# def _decode_tfrecord():
#
#   keys_to_features = {
#     'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
#     'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
#     'image/class/label': tf.FixedLenFeature(
#       [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
#   }
#
#   items_to_handlers = {
#     'image': slim.tfexample_decoder.Image(shape=[28, 28, 1], channels=1),
#     'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
#   }
#
#   decoder = slim.tfexample_decoder.TFExampleDecoder(
#     keys_to_features, items_to_handlers)
#
#   return decoder


class MNISTDatasetBuilder(object):

  @staticmethod
  def read_dataset(dataset_name, dataset_path=None):
    """Load the dataset from the dataset_name and dataset_path"""

    dataset = input_data.read_data_sets(dataset_name, one_hot=True)

    input_dataset = dirl_utils.T()
    input_dataset.train_data = dataset.train.images.reshape(-1, 28, 28, 1)
    # concatenate the dataset for 3 channels input
    input_dataset.train_data = np.concatenate(
      [input_dataset.train_data, input_dataset.train_data, input_dataset.train_data], 3)
    input_dataset.train_labels = dataset.train.labels

    input_dataset.test_data = dataset.test.images.reshape(-1, 28, 28, 1)
    input_dataset.test_data = np.concatenate(
      [input_dataset.test_data, input_dataset.test_data, input_dataset.test_data], 3)
    input_dataset.test_labels = dataset.test.labels

    input_dataset.emb_data = dataset.validation.images.reshape(-1, 28, 28, 1)
    input_dataset.emb_data = np.concatenate(
      [input_dataset.emb_data, input_dataset.emb_data, input_dataset.emb_data], 3)
    input_dataset.emb_labels = dataset.validation.labels

    return input_dataset

  @staticmethod
  def write_tfrecords(input_data, output_directory, domain):
    """Write tfrecords for training and testing set to the output directory

    Args:
      input_data: A T object with keys dataset_config, train_data, test_data.
    """

    if not tf.gfile.Exists(output_directory):
      tf.gfile.MakeDirs(output_directory)

    train_filename = os.path.join(output_directory, domain + '_mnist_train.tfrecord')
    test_filename = os.path.join(output_directory, domain + '_mnist_test.tfrecord')
    emb_filename = os.path.join(output_directory, domain + '_mnist_emb.tfrecord')

    if tf.gfile.Exists(train_filename) and tf.gfile.Exists(test_filename) and tf.gfile.Exists(emb_filename):
      print('Dataset files already exist. Exiting without re-creating them.')
      return
    
    if domain == 'source':
      source_id = 1
    elif domain == 'target':
      source_id = 0
    else:
      raise ValueError('Domain is not defined: %s' %domain)

    source_id = str(source_id).encode('utf8')

    _write_to_tfrecord(input_data.train_data, input_data.train_labels, train_filename, source_id)

    _write_to_tfrecord(input_data.test_data, input_data.test_labels, test_filename, source_id)

    _write_to_tfrecord(input_data.emb_data, input_data.emb_labels, emb_filename, source_id)

    num_classes = 10
    # Finally, write the labels file:
    class_names = [str(class_index) for class_index in range(num_classes)]
    labels_to_class_names = dict(zip(range(num_classes), class_names))

    dataset_utils.write_label_file(labels_to_class_names, output_directory)

  @staticmethod
  def make_initializable_iterator_from_tfrecords(input_directory, dataset_config, domain):

    train_filename = os.path.join(input_directory, domain + '_mnist_train.tfrecord')
    test_filename = os.path.join(input_directory, domain + '_mnist_test.tfrecord')

    if not tf.gfile.Exists(train_filename):
      raise ValueError('File not found: %s' %train_filename)

    if not tf.gfile.Exists(test_filename):
      raise ValueError('File not found: %s' %test_filename)

    dataset = tf.data.Dataset.from_tensor_slices(tf.unstack([train_filename]))

    dataset = dataset.repeat()

    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
      'image/class/label': tf.FixedLenFeature(
        [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
      'image/source_id': tf.FixedLenFeature(
        (), tf.string, default_value=''),
    }

    items_to_handlers = {
      IMAGE_KEY: slim.tfexample_decoder.Image(image_key='image/encoded', format_key='image/format', shape=[28, 28, 3], channels=3),
      CLASS_LABEL_KEY: slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
      DOMAIN_LABEL_KEY: slim.tfexample_decoder.Tensor('image/source_id', shape=[]),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

    def process_fn(value):
      tensors = decoder.decode(value)
      keys = decoder.list_items()
      tensor_dict = dict(zip(keys, tensors))
      return tensor_dict

    file_read_func = functools.partial(tf.data.TFRecordDataset, buffer_size=8 * 1000 * 1000)
    records_dataset = dataset.apply(
      tf.contrib.data.parallel_interleave(
        file_read_func,
        cycle_length=dataset_config.num_readers,
        block_length=dataset_config.read_block_length,
        sloppy=dataset_config.shuffle))

    if dataset_config.shuffle:
      records_dataset = records_dataset.shuffle(dataset_config.shuffle_buffer_size)

    tensor_dataset = records_dataset.map(process_fn)

    # tensor_dataset = proprocesser.preprocess(tensor_dict)
    tensor_dataset = tensor_dataset.prefetch(dataset_config.prefetch_size)

    iterator = tensor_dataset.make_initializable_iterator()

    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    return iterator


  @staticmethod
  def make_initializable_iterator(input_data, dataset_config):
    """Create an iterator and initialize tables.

    Args:
      dataset: A T object with keys dataset_config, train_data, test_data.

    Returns:
      iterator: A 'tf.data.Iterator' object.
    """
    import base64
    dataset = tf.data.Dataset.from_tensor_slices(
      (input_data.train_data, input_data.train_labels))
    # Automatically refill the data queue when empty
    dataset = dataset.repeat()

    if dataset_config.shuffle:
      dataset = dataset.shuffle(dataset_config.shuffle_buffer_size)

    # keys_to_features = {
    #   'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    #   'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
    #   'image/class/label': tf.FixedLenFeature(
    #     [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
    # }
    #
    # items_to_handlers = {
    #   'image': slim.tfexample_decoder.Image(shape=[28, 28, 3], channels=3),
    #   'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
    # }
    #
    # decoder = slim.tfexample_decoder.TFExampleDecoder(
    #   keys_to_features, items_to_handlers)
    #
    # def process_fn(value):
    #   tensors = decoder.decode(value)
    #   keys = decoder.list_items()
    #   tensor_dict = dict(zip(keys, tensors))
    #   return tensor_dict
    #
    # import ipdb;
    # ipdb.set_trace()
    # tensor_dataset = dataset.map(process_fn)

    tensor_dataset = dataset.prefetch(dataset_config.prefetch_size)

    iterator = tensor_dataset.make_initializable_iterator()

    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    return iterator

