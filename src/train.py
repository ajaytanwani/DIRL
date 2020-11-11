import functools
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from dirl_core import dirl_utils
from dirl_core import trainer, source_trainer
from model_builders import model_builder
from dataset_builders import dataset_builder
# from loss_builders import loss_builder

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('dataset_name', '', 'Name of the dataset.')
flags.DEFINE_string('train_config_path', '', 'path to the config file.')
flags.DEFINE_string('split_name', '', 'train or test or validation split of the dataset.')
flags.DEFINE_string('model_name', '', 'Name of the model to train the algorithm.')
flags.DEFINE_string('train_dir', '',
                    'Directory to save the checkpoints and training summaries.')
flags.DEFINE_string('train_tfrecord_path', '',
                    'path to the tf record of the training dataset.')
flags.DEFINE_string('test_tfrecord_path', '',
                    'path to the tf record of the testing dataset.')
# flags.DEFINE_float('dann', 0, 'whether to use a domain adversarial classifier')
# flags.DEFINE_boolean('dann_target_labels', False, 'whether to use target domain labels when training the dann')
# flags.DEFINE_boolean('wgan_loss', False, 'whether to use wgan discriminator loss when training the dann')

FLAGS = flags.FLAGS


def main(_):
  assert FLAGS.train_dir, '`train_dir` is missing.'
  assert FLAGS.train_config_path, '`train_config_path` is missing'

  if not tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.MakeDirs(FLAGS.train_dir)

  configs = dirl_utils.ParseConfigsToLuaTable(FLAGS.train_config_path, save=True, logdir=FLAGS.train_dir)

  # Create the model function
  model_fn = functools.partial(
      model_builder.build_model,
      model_config=configs.model_config,
      is_training=True,
      train_dir=FLAGS.train_dir)

  dataset_instance = dataset_builder.BaseDataset.build_dataset(configs.dataset_config, FLAGS.train_dir)

  dataset_fn = functools.partial(dataset_builder.BaseDataset.create_dataset_iterators,
                                 dataset_config=configs.dataset_config, input_directory=FLAGS.train_dir)

  # import ipdb; ipdb.set_trace()

  trainer_instance = source_trainer.SourceTrainer(dataset_fn, model_fn, configs, FLAGS.train_dir)
  trainer_instance.trainer_routine(dataset_fn, model_fn, configs, FLAGS.train_dir)

  # model_instance = model_fn()
  #
  # with tf.Graph().as_default():
  #   input_queue = model_instance.create_input_queue(dataset_fn, configs.model_config,
  #                                                   queue_type=configs.model_config['input_queue_type'])
  #
  #   model_loss = functools.partial(model_instance.create_losses, train_config=configs)

# define the training job
  # trainer_instance = source_trainer.SourceTrainer(dataset_fn, model_fn, configs, FLAGS.train_dir)
  # trainer_instance.trainer_routine(dataset_fn, model_fn, configs, FLAGS.train_dir)

#   # Parameters for a single worker.
#   ps_tasks = 0
#   worker_replicas = 1
#   worker_job_name = 'lonely_worker'
#   task = 0
#   is_chief = True
#   master = ''

  # create optimizer using input_fn, loss_fn, and model_fn

  # write the training function call that takes the model_fn, input_fn, loss_fn

  # trainer.train(
  #     create_input_dict_fn,
  #     model_fn,
  #     train_config,
  #     master,
  #     task,
  #     FLAGS.num_clones,
  #     worker_replicas,
  #     FLAGS.clone_on_cpu,
  #     ps_tasks,
  #     worker_job_name,
  #     is_chief,
  #     FLAGS.train_dir,
  #     graph_hook_fn=graph_rewriter_fn,
  #     dann=FLAGS.dann,
  #     dann_target_labels=FLAGS.dann_target_labels)


if __name__ == '__main__':
  tf.app.run()