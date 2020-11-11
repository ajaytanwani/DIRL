import collections
import functools
import numpy as np
import tensorflow as tf
from deployment import model_deploy
from model_builders import optimizer_builder, variables_helper
from dirl_core import trainer

slim =tf.contrib.slim

class SourceTrainer(trainer.Trainer):

  def __init__(self, dataset_fn, model_fn, train_config, train_dir):

    super(SourceTrainer, self).__init__(dataset_fn, model_fn, train_config, train_dir)

  def trainer_routine(self, dataset_fn, model_fn, train_config, train_dir):
    """Initialize the training job.

    Args:
      dataset_fn: A T object containing dataset_config,
      model_fn:
      model_config:
      train_dir:
    """

    model_instance = model_fn()

    with tf.Graph().as_default():

      # Parameters for a single worker.
      ps_tasks = 0
      worker_replicas = 1
      worker_job_name = 'lonely_worker'
      task = 0
      is_chief = True
      master = ''
      num_clones=1
      clone_on_cpu=False


      assert num_clones is 1

      deploy_config = model_deploy.DeploymentConfig(
        num_clones=num_clones,
        clone_on_cpu=clone_on_cpu,
        replica_id=task,
        num_replicas=worker_replicas,
        num_ps_tasks=ps_tasks,
        worker_job_name=worker_job_name)

      # Place the global step on the device storing the variables.
      with tf.device(deploy_config.variables_device()):
        global_step = slim.create_global_step()

      with tf.device(deploy_config.inputs_device()):

        input_queue = model_instance.create_input_queue(dataset_fn, train_config.model_config,
                                                        queue_type=train_config.model_config['input_queue_type'])

      # collections so that they don't have to be passed around.
      summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
      global_summaries = set([])

      model_loss = functools.partial(model_instance.create_losses,
                                     train_config=train_config)

      clones = model_deploy.create_clones(deploy_config, model_loss, [input_queue])
      first_clone_scope = clones[0].scope

      # Gather update_ops from the first clone. These contain, for example,
      # the updates for the batch_norm variables created by model_fn.
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

      with tf.device(deploy_config.optimizer_device()):
        training_optimizer, optimizer_summary_vars = optimizer_builder.build(
          train_config.model_config.optimizer)
        for var in optimizer_summary_vars:
          tf.summary.scalar(var.op.name, var, family='LearningRate')

      # import ipdb; ipdb.set_trace()
      with tf.device(deploy_config.optimizer_device()):
        regularization_losses = (None if train_config.model_config.losses.add_regularization_loss
                                 else [])

        # # Where variable filters were implemented.
        # trainable_vars = variables_helper.filter_variables(tf.trainable_variables(),
        #                                                   ['grasp'],
        #                                                   invert=False)
        # import ipdb; ipdb.set_trace()

        trainable_vars = tf.trainable_variables()

        total_loss, grads_and_vars = model_deploy.optimize_clones(
          clones, training_optimizer,
          regularization_losses=regularization_losses, var_list=trainable_vars)
        total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan.')

        # Optionally multiply bias gradients by train_config.bias_grad_multiplier.
        if train_config.model_config.bias_grad_multiplier:
          biases_regex_list = ['.*/biases']
          grads_and_vars = variables_helper.multiply_gradients_matching_regex(
            grads_and_vars,
            biases_regex_list,
            multiplier=train_config.bias_grad_multiplier)

        # Optionally freeze some layers by setting their gradients to be zero.
        if train_config.model_config.freeze_variables:
          grads_and_vars = variables_helper.freeze_gradients_matching_regex(
            grads_and_vars, train_config.freeze_variables)

        # Optionally clip gradients
        if train_config.model_config.gradient_clipping_by_norm > 0:
          with tf.name_scope('clip_grads'):
            grads_and_vars = slim.learning.clip_gradient_norms(
              grads_and_vars, train_config.gradient_clipping_by_norm)

        # Create gradient updates.
        grad_updates = training_optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops, name='update_barrier')
        with tf.control_dependencies([update_op]):
          train_tensor = tf.identity(total_loss, name='train_op')

      # Add summaries.
      for model_var in slim.get_model_variables():
        global_summaries.add(tf.summary.histogram('ModelVars/' +
                                                  model_var.op.name, model_var))
      for loss_tensor in tf.losses.get_losses():
        global_summaries.add(tf.summary.scalar('Losses/' + loss_tensor.op.name,
                                               loss_tensor))
      global_summaries.add(
        tf.summary.scalar('Losses/TotalLoss', tf.losses.get_total_loss()))

      # Add the summaries from the first clone. These contain the summaries
      # created by model_fn and either optimize_clones() or _gather_clone_loss().
      summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                         first_clone_scope))
      summaries |= global_summaries

      # Merge all summaries together.
      summary_op = tf.summary.merge(list(summaries), name='summary_op')

      # Soft placement allows placing on CPU ops without GPU implementation.
      session_config = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=False)

      # import ipdb; ipdb.set_trace()
      # Save checkpoints regularly.
      keep_checkpoint_every_n_hours = train_config.model_config.keep_checkpoint_every_n_hours

      # Added Target vars param
      saver = tf.train.Saver(  # target_vars,
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

      # Create ops required to initialize the model from a given checkpoint.
      init_fn = None
      if train_config.model_config.fine_tune_checkpoint and False:
        if not train_config.model_config.fine_tune_checkpoint_type:
          # train_config.from_detection_checkpoint field is deprecated. For
          # backward compatibility, fine_tune_checkpoint_type is set based on
          # from_detection_checkpoint.
          if train_config.model_config.from_detection_checkpoint:
            train_config.fine_tune_checkpoint_type = 'detection'
          else:
            train_config.model_config.fine_tune_checkpoint_type = 'classification'

        var_map = model_instance.restore_map(
          fine_tune_checkpoint_type=train_config.fine_tune_checkpoint_type,
          load_all_detection_checkpoint_vars=(
            train_config.load_all_detection_checkpoint_vars))

        available_var_map = (variables_helper.
          get_variables_available_in_checkpoint(
          var_map, train_config.model_config.fine_tune_checkpoint,
          include_global_step=False))

        #  # Add Target Vars
        # print(available_var_map)
        # available_var_map = variables_helper.filter_variables(available_var_map,
        #                                                   ['grasp'],
        #                                                   invert=False)
        # # target_vars = variables_helper.filter_variables(tf.trainable_variables(), 'source', invert=False)
        # # mapping_vars = variables_helper.filter_variables(tf.trainable_variables(), ['source', '.*dann'], invert=False)

        init_saver = tf.train.Saver(available_var_map)

        def initializer_fn(sess):
          # sess.run(tf.global_variables_initializer())
          init_saver.restore(sess, train_config.fine_tune_checkpoint)

        init_fn = initializer_fn

      slim.learning.train(
        train_tensor,
        logdir=train_dir,
        master=master,
        is_chief=is_chief,
        session_config=session_config,
        startup_delay_steps=train_config.model_config.startup_delay_steps,
        init_fn=init_fn,
        summary_op=summary_op,
        number_of_steps=(train_config.model_config.num_steps if train_config.model_config.num_steps else None),
        save_summaries_secs=120,
        save_interval_secs=120,
        sync_optimizer=None,
        saver=saver)
