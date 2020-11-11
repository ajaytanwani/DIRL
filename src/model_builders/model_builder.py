import tensorflow as tf

from dirl_core import aux_funcs
from model_builders import mnist_model


# # A map of names to SSD feature extractors.
# SSD_FEATURE_EXTRACTOR_CLASS_MAP = {
#     'ssd_mobilenet_v1': SSDMobileNetV1FeatureExtractor,
#     'ssd_mobilenet_v1_fpn': SSDMobileNetV1FpnFeatureExtractor,
# }


def build_model(model_config, is_training, train_dir=None,
          add_summaries=True):
  """Builds a DIRL model based on the model config.

  Args:
    model_config: A model object containing the config for the desired
      DIRL model.
    is_training: True if this model is being built for training purposes.
    add_summaries: Whether to add tensorflow summaries in the model graph.
  Returns:
    DIRL model based on the config.

  Raises:
    ValueError: On invalid meta architecture or model.
  """
  # meta_architecture = model_config.WhichOneof('model')
  meta_architecture = model_config.model_name

  if meta_architecture == 'mnist_net':
    return _build_mnist_net(model_config, is_training, add_summaries,
                             train_dir)
  raise ValueError('Unknown meta architecture: {}'.format(meta_architecture))


def _build_mnist_net(model_config, is_training, add_summaries, train_dir):
  """Builds a DANN model for reconstruction type.

  Args:
    model_config: A model object containing the config for the desired
      DIRL model.
    is_training: True if this model is being built for training purposes.
    add_summaries: Whether to add tensorflow summaries in the model graph.
  Returns:
    DIRL model based on the config.

  Raises:
    ValueError: On invalid meta architecture or model.
  """

  mnist_model_build = mnist_model.MNISTArchitecture(model_config, is_training)

  return mnist_model_build

  # input = tf.placeholder(tf.float32,
  #                    [None, model_config.input_width, model_config.input_height, model_config.input_num_channels])
  #
  # preprocessed_inputs = mnist_model.preprocess(input)
  #
  # predictions_dict = mnist_model.predict(preprocessed_inputs)
  #
  # #TODO(): get groundtruth dictionary here
  # groundtruth_dict = {}
  #
  # loss_dict = mnist_model.loss(preprocessed_inputs, groundtruth_dict)


  # mnist_model.train()


def _build_simple_net(model_config, is_training, add_summaries, train_dir):
  """ Builds a simple feedforward network for classification problem.

  Args:
    model_config: A model object containing the config for the desired
      DIRL model.
    is_training: True if this model is being built for training purposes.
    add_summaries: Whether to add tensorflow summaries in the model graph.
  Returns:
    DIRL model based on the config.

  Raises:
    ValueError: On invalid meta architecture or model.
  """

  # TODO(danielzeng): replace model with parameters from the config file.

  # seems like model_config will be essentially a python dict - correct me if i'm wrong

  import ipdb; ipdb.set_trace()
  x = tf.placeholder(tf.float32, [None, model_config.input_width, model_config.input_height, model_config.input_num_channels])
  #not sure how to get source_dataset_height because only passed in model_config, not input_config

  init = tf.contrib.layers.xavier_initializer()
  tf_out = x
  latent_dim = model_config['reconstruction']['latent_dim']
  
  # encoding
  for i, layer in enumerate(model_config['reconstruction']['encoder']['net_architecture']):
    tf_out = _add_layer(tf_out, i, model_config['reconstruction']['encoder'], init, tf.nn.leaky_relu) 
    #not sure how to specify activations individually - either large if-else for each activation, or something else

  # decoding
  for i, layer in enumerate(model_config['reconstruction']['decoder']['net_architecture']):
    tf_out = _add_layer(tf_out, i, model_config['reconstruction']['decoder'], init, tf.nn.leaky_relu)

  return tf_out


def _add_layer(tf_in, i, layers_config, init, activation_):
  layer_type = _safe_get(layers_config, 'net_architecture', i)
  num_filters_ = _safe_get(layers_config, 'num_filters', i)
  padding_ = _safe_get(layers_config, 'padding', i)
  strides_ = _safe_get(layers_config, 'strides', i)
  kernel_size_ = _safe_get(layers_config, 'kernel_size', i)
  pool_size_ = _safe_get(layers_config, 'pool_size', i)
  output_dim_ = _safe_get(layers_config, 'output_dim', i)
  reshape_ = _safe_get(layers_config, 'reshape', i)

  if layer_type == 'conv2d':
    return tf.layers.conv2d(inputs=tf_in, filters=num_filters_, kernel_size=kernel_size_, padding=padding_,
                            activation=activation_)

  if layer_type == 'max_pooling2d':
    return tf.layers.max_pooling2d(inputs=tf_in, pool_size=pool_size_, strides=strides_)

  if layer_type == 'flatten':
    return tf.layers.flatten(tf_in)

  if layer_type == 'dense':
    return tf.layers.dense(tf_in, output_dim_, activation=activation_, kernel_initializer=init)

  if layer_type == 'conv2d_T':
    return tf.layers.conv2d_transpose(tf_in, num_filters_, kernel_size=kernel_size_, padding=padding_, strides=strides_,
                                      activation=activation_)

  if layer_type == 'reshape':
    return tf.reshape(tf_in, reshape_)


def _safe_get(dict_, layer_name, i):
  try:
    return dict_[layer_name][i]
  except IndexError as e:
    pass
  except KeyError:
    pass
