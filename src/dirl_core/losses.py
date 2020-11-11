import tensorflow as tf
import numpy as np


def triplet_loss(labels, embeddings, margin=1.0, scale=1.0, batch_reduce_factor=1, loss_name='Loss/triplet_loss'):

  batch_size = int(labels.shape[0])

  triplet_loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(
    labels=labels[:batch_size // batch_reduce_factor],
    embeddings=embeddings[:batch_size // batch_reduce_factor],
    margin=margin)

  triplet_loss = tf.scalar_mul(scale, triplet_loss)
  triplet_loss = tf.identity(triplet_loss, loss_name)

  return triplet_loss


def reconstruction_loss(inputs, outputs, scale=1.0, loss_name='Loss/reconstruction_loss'):

  reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs))

  reconstruction_loss = tf.scalar_mul(scale, reconstruction_loss)

  reconstruction_loss = tf.identity(reconstruction_loss, loss_name)

  return reconstruction_loss


def softmax_cross_entropy_with_logits_v2(logits, labels, scale=1.0, batch_reduce_factor=1, loss_name='Loss/classify_loss'):

  batch_size = int(labels.shape[0])

  classification_loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits[:batch_size // batch_reduce_factor],
                                               labels=labels[:batch_size // batch_reduce_factor]))

  classification_loss = tf.scalar_mul(scale, classification_loss)

  classification_loss = tf.identity(classification_loss, loss_name)

  classification_accuracy_batch1 = tf.reduce_mean(
    tf.cast(tf.equal(tf.argmax(labels[:batch_size // batch_reduce_factor], axis=1),
                     tf.argmax(logits[:batch_size // batch_reduce_factor], axis=1)), tf.float32))

  classification_accuraxcy_batch2 = tf.reduce_mean(
    tf.cast(tf.equal(tf.argmax(labels[batch_size // batch_reduce_factor:], axis=1),
                     tf.argmax(logits[batch_size // batch_reduce_factor:], axis=1)), tf.float32))

  return classification_loss, classification_accuracy_batch1, classification_accuracy_batch2


def kl_div(p_logits, q_logits, loss_name='Loss/kl_loss'):
  # https://github.com/tensorflow/models/blob/645202b1e62d323d79936eea9452ef3b58084826/research/adversarial_text/adversarial_losses.py#L217
  # let p = real, q = fake
  q = tf.nn.softmax(q_logits)
  kl = tf.reduce_mean(q * (tf.nn.log_softmax(q_logits) - tf.nn.log_softmax(p_logits)))

  kl = tf.identity(kl, loss_name)
  #     p = tf.nn.softmax(p_logits)
  #     kl2 = tf.reduce_mean(-tf.nn.softmax_cross_entropy_with_logits(labels=p, logits=p/q))
  return kl