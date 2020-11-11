import tensorflow as tf
from dirl_core import aux_funcs, dirl_utils, losses
from model_builders import base_model
from dataset_builders import mnist_dataset_builder

slim = tf.contrib.slim

#PREDICTIONS_DICTIONARY KEYS

PREPROCESSED_INPUTS_KEY = 'preprocessed_inputs'
EMBEDDING_FEATURES_KEY = 'embedding_features'
PRED_RECONSTRUCTION_KEY = 'pred_reconstruction'
PRED_DOMAIN_KEY = 'pred_domain'
PRED_CLASSES_KEY = 'pred_classes'


class MNISTArchitecture(base_model.BaseModel):

  def __init__(self, model_config, is_training):

    # super(MNISTArchitecture, self).__init__(model_config, is_training, train_dir, add_summaries)

    super(MNISTArchitecture, self).__init__(model_config, is_training)

  def preprocess(self, inputs):
    """

    Args:
      input: a [batch, height, width, channels] float tensor
        representing a batch of images.
    Returns:
      preprocessed_input: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    # if inputs.dtype is not tf.float32:
    #   raise ValueError('`preprocess` expects a tf.float32 tensor')
    # with tf.name_scope('Preprocessor'):
    #   # TODO(jonathanhuang): revisit whether to always use batch size as
    #   # the number of parallel iterations vs allow for dynamic batching.
    #   outputs = dirl_utils.static_or_dynamic_map_fn(
    #       self._image_resizer_fn,
    #       elems=inputs,
    #       dtype=[tf.float32, tf.int32])
    #   resized_inputs = outputs[0]
    #   true_image_shapes = outputs[1]

    return (inputs * 1.0)

  def postprocess(self, predictions_dict):

    return predictions_dict['preprocessed_inputs'] * 1.0


  def predict(self, preprocessed_inputs):

    latent_dim = self.model_config['reconstruction']['latent_dim']  # test lol (can be lower?)

    batch_size = self.model_config['batch_size']

    init = tf.contrib.layers.xavier_initializer()

    with slim.arg_scope([slim.batch_norm], is_training=(self._is_training)):
      with tf.variable_scope(None, 'embedding_features', [preprocessed_inputs]):
        net = tf.reshape(preprocessed_inputs, (batch_size, 28, 28, 3))
        net = slim.conv2d(inputs=net, num_outputs=32, kernel_size=[3,3], padding='valid', activation_fn=tf.nn.leaky_relu)

        net = slim.conv2d(inputs=net, num_outputs=32, kernel_size=[3, 3], padding='valid', activation_fn=tf.nn.leaky_relu)

        net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], stride=2)

        net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=[3, 3], padding='valid', activation_fn=tf.nn.leaky_relu)

        net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=[3, 3], padding='valid', activation_fn=tf.nn.leaky_relu)

        net = slim.flatten(net)

        net = slim.fully_connected(net, num_outputs=16*64, activation_fn=tf.nn.leaky_relu)

        net = slim.fully_connected(net, num_outputs=latent_dim, activation_fn=tf.nn.leaky_relu)
        
      with tf.variable_scope('dann'):
        d_net = aux_funcs.flip_gradient(net, self.model_config.losses.domain_loss.domain_lambda_scale_factor)
        d_net = slim.fully_connected(d_net, num_outputs=100, activation_fn=tf.nn.leaky_relu)
        d_net = slim.fully_connected(d_net, num_outputs=50, activation_fn=tf.nn.leaky_relu)
        d_net = slim.fully_connected(d_net, num_outputs=2, activation_fn=None)        
        
      with tf.variable_scope('reconstuction'):
        r_net = slim.fully_connected(net, num_outputs=1024, activation_fn=tf.nn.leaky_relu)
        r_net = slim.fully_connected(r_net, num_outputs=7*7*128, activation_fn=tf.nn.leaky_relu)
        r_net = tf.reshape(r_net, (batch_size, 7, 7, 128))
        
        r_net = slim.conv2d_transpose(r_net, num_outputs=128, kernel_size=[4,4], padding='same', stride=2, activation_fn=tf.nn.leaky_relu)
        
        r_net = slim.conv2d_transpose(r_net, num_outputs=1, kernel_size=[4,4], padding='same', stride=2, activation_fn=tf.nn.leaky_relu) 
        
        r_net = slim.flatten(r_net)
        r_net = slim.fully_connected(inputs=r_net, num_outputs=28*28*3, activation_fn=tf.nn.sigmoid)
        r_net = tf.reshape(r_net, (batch_size, 28, 28, 3))
        
        
      with tf.variable_scope('source_classify'):
        s_net = slim.fully_connected(inputs=net, num_outputs=100, activation_fn=tf.nn.leaky_relu)
        s_net = slim.fully_connected(inputs=s_net, num_outputs=50, activation_fn=tf.nn.leaky_relu)
        s_net = slim.fully_connected(inputs=s_net, num_outputs=10, activation_fn=None)        
        
    predictions_dict = {
      PREPROCESSED_INPUTS_KEY: preprocessed_inputs,
      EMBEDDING_FEATURES_KEY: net,
      PRED_RECONSTRUCTION_KEY: r_net,
      PRED_DOMAIN_KEY: d_net,
      PRED_CLASSES_KEY: s_net
    }
    
    return predictions_dict

  def create_groundtruth_dictionary(self, class_labels_list, domain_labels_list=None):
    """Create dictionary for groundtruth labels.

    Args:
      class_labels_list: tuple of tensors containing the class labels in a given batch.
      domain_labels_list: tuple of tensors containing the domain labels in a give batch
    """

    groundtruth_dict = {}

    groundtruth_dict[mnist_dataset_builder.CLASS_LABEL_KEY] = class_labels_list

    if domain_labels_list is not None:
      groundtruth_dict[mnist_dataset_builder.DOMAIN_LABEL_KEY] = domain_labels_list

    return groundtruth_dict

  
  def loss(self, predictions_dict, groundtruth_dict):
    """Define loss functions given the prediction and the groundtruth dictionary of tensors

    Args:
      predictions_dict: a dictionary holding predicted tensors
      groundtruth_dict: a dictionary holding groundtruth tensors for computing the loss

    Returns:
      loss_dict: a dictionary mapping strings (loss names) to scalar tensors representing
        loss values.
    """
    loss_config = self.model_config.losses
    
    batch_size = self.model_config.batch_size

    batch_reduce_factor = 2
    if self.model_config.input_queue_type in ['source_only', 'target_only']:
      batch_reduce_factor = 1

    loss_dict = {
      'triplet_loss': None,
      'domain_loss': None,
      'classify_loss': None,
      'reconstruction_loss': None,
      'wasserstein_loss': None,
      'kl_loss': None
    }

    if loss_config.reconstruction_loss.use_reconstruction_loss:
      reconstruction_loss = losses.reconstruction_loss(inputs=predictions_dict[PREPROCESSED_INPUTS_KEY],
                                                       outputs=predictions_dict[PRED_RECONSTRUCTION_KEY],
                                                       scale=float(loss_config.reconstruction_loss.reconstruction_loss_weight))

      loss_dict['reconstruction_loss'] = reconstruction_loss

    if loss_config.triplet_loss.use_triplet_loss:
      triplet_loss = losses.triplet_loss(labels=tf.argmax(groundtruth_dict[mnist_dataset_builder.CLASS_LABEL_KEY], axis=1),
                                         embeddings=predictions_dict[EMBEDDING_FEATURES_KEY],
                                         margin=float(loss_config.triplet_loss.triplet_loss_margin),
                                         scale=float(loss_config.triplet_loss.triplet_loss_weight),
                                         batch_reduce_factor=batch_reduce_factor)

      loss_dict['triplet_loss'] = triplet_loss

    if loss_config.domain_loss.use_domain_loss:
      domain_loss, domain_accuracy, _ = losses.softmax_cross_entropy_with_logits_v2(
        logits=predictions_dict[PRED_DOMAIN_KEY],
        labels=groundtruth_dict[
          mnist_dataset_builder.DOMAIN_LABEL_KEY],
        scale=float(
          loss_config.domain_loss.domain_loss_weight), loss_name='Loss/domain_loss')

      loss_dict['domain_loss'] = domain_loss
    
      # TODO(): Add classification_accuracy to the tensorboard summary

    if loss_config.source_classification_loss.use_source_classification_loss:

      source_classify_loss, classify_source_accuracy, classify_target_accuracy = losses.softmax_cross_entropy_with_logits_v2(
        logits=predictions_dict[PRED_CLASSES_KEY], labels=groundtruth_dict[mnist_dataset_builder.CLASS_LABEL_KEY],
        scale=float(loss_config.source_classification_loss.source_classification_loss_weight),
        batch_reduce_factor=batch_reduce_factor, loss_name='Loss/source_classify_loss')

      loss_dict['classify_loss'] = source_classify_loss

    # TODO(): Recheck the output of the wasserstein loss, adjust the batch size
    if loss_config.wasserstein_loss.use_wasserstein_loss:
      wasser_loss = -tf.contrib.gan.losses.wargs.wasserstein_discriminator_loss(
        discriminator_real_outputs=predictions_dict[PRED_DOMAIN_KEY][:batch_size // batch_reduce_factor],
        discriminator_gen_outputs=predictions_dict[PRED_DOMAIN_KEY][batch_size // batch_reduce_factor:])
        
      wasser_loss = tf.scalar_mul(loss_config.wasserstein_loss.wasserstein_loss_weight, wasser_loss)

      loss_dict['wasserstein_loss'] = wasser_loss

    if loss_config.kl_divergence_loss.use_kl_divergence_loss:
      kl_loss = losses.kl_div(predictions_dict[EMBEDDING_FEATURES_KEY][:batch_size // batch_reduce_factor],
                              predictions_dict[EMBEDDING_FEATURES_KEY][batch_size // batch_reduce_factor:])
        
      kl_loss = tf.scalar_mul(loss_config.use_kl_divergence_loss.kl_divergence_loss_weight, kl_loss)

      loss_dict['kl_loss'] = kl_loss

      
    # loss = triplet_loss + domain_loss + reconstruction_loss + classify_loss
    #
    # loss_dict = {
    #   'triplet_loss': triplet_loss,
    #   'domain_loss': domain_loss,
    #   'classify_loss': source_classify_loss,
    #   'reconstruction_loss': reconstruction_loss,
    #   'wasserstein_loss': wasser_loss,
    #   'kl_loss': kl_loss
    # }

    return loss_dict
    
    
    
  def create_input_queue(self, dataset_fn, train_config, queue_type=''):

    source_iterator, target_iterator = dataset_fn()

    source_tensor_dict = source_iterator.get_next()

    source_tensor_dict[mnist_dataset_builder.IMAGE_KEY] = tf.to_float(
      tf.expand_dims(source_tensor_dict[mnist_dataset_builder.IMAGE_KEY], 0))

    # source_tensor_dict = self.preprocess(source_tensor_dict, train_config['preprocess'])

    if target_iterator is not None:
      target_tensor_dict = target_iterator.get_next()

      target_tensor_dict[mnist_dataset_builder.IMAGE_KEY] = tf.to_float(
        tf.expand_dims(target_tensor_dict[mnist_dataset_builder.IMAGE_KEY], 0))

      # target_tensor_dict = self.preprocess(source_tensor_dict, train_config['preprocess'])


    if queue_type == 'source_only':

      source_queue = base_model.BatchQueue(source_tensor_dict, train_config['batch_size'],
                                           train_config['batch_queue_capacity'],
                                           train_config['num_batch_queue_threads'],
                                           train_config['prefetch_queue_capacity'])

      return source_queue

    elif queue_type == 'target_only':

      target_queue = base_model.BatchQueue(target_tensor_dict, train_config['batch_size'],
                                           train_config['batch_queue_capacity'],
                                           train_config['num_batch_queue_threads'],
                                           train_config['prefetch_queue_capacity'])
      return target_queue

    elif queue_type == 'shared':

      source_target_queue = base_model.SharedBatchQueue(source_tensor_dict, target_tensor_dict,
                                                        train_config['batch_size'],
                                                        train_config['batch_queue_capacity'],
                                                        train_config['num_batch_queue_threads'],
                                                        train_config['prefetch_queue_capacity'])

      return source_target_queue

    else:
      raise ValueError('Input queue creation is not supported for queue_type: %s', queue_type)

  def create_losses(self, input_queue, train_config):
    """Define training loss functions from the input queue

    Args:
      input_queue: a prefetcher FIFO queue.
      train_config: a T object defining instructions for losses.
    """
    read_data_list = input_queue.dequeue()
    
    def extract_images_and_targets(read_data):
      """Extract images and targets from dequeued data.
      
      Args:
        read_data: dequeud batched tensor dictionary list 
      """
      image = read_data[mnist_dataset_builder.IMAGE_KEY]
      key = ''
      if mnist_dataset_builder.DOMAIN_LABEL_KEY in read_data.keys():
        key = read_data[mnist_dataset_builder.DOMAIN_LABEL_KEY]
      key = tf.string_to_number(key, out_type=tf.int64)
      key = tf.one_hot(key, 2, on_value=1, off_value=0)
      key = tf.reshape(key, (-1, 2))
      classes_gt = tf.cast(read_data[mnist_dataset_builder.CLASS_LABEL_KEY], tf.int32)
      classes_gt = tf.cast(tf.one_hot(classes_gt, train_config.dataset_config['num_classes'], on_value=1, off_value=0),
                           tf.float32)
      classes_gt = tf.reshape(classes_gt, (-1, train_config.dataset_config['num_classes']))

      return (image, key, classes_gt)

    images, domain_labels, class_labels = zip(*map(extract_images_and_targets, read_data_list))

    preprocessed_images = []

    for image in images:
      resized_image = self.preprocess(image)
      preprocessed_images.append(resized_image)

    preprocessed_images = tf.concat(preprocessed_images, 0)
    class_labels = tf.concat(class_labels, 0)
    domain_labels = tf.concat(domain_labels, 0)

    groundtruth_dict = self.create_groundtruth_dictionary(class_labels, domain_labels_list=domain_labels)

    predictions_dict = self.predict(preprocessed_images)

    losses_dict = self.loss(predictions_dict, groundtruth_dict)

    for loss_key, loss_tensor in losses_dict.items():
      if loss_tensor is not None:
        # import ipdb;ipdb.set_trace()
        print('Loss tensor added is %s of type %s' %(loss_key, loss_tensor))
        tf.losses.add_loss(loss_tensor)
#
  # def train(self):
  #   """Train function for the mnist domain adaptation
  #   """
