import tensorflow as tf
import numpy as np

def compute_pairwise_distances(x, y):
    """Computes the squared pairwise Euclidean distances between x and y.
    Args:
        x: a tensor of shape [num_x_samples, num_features]
        y: a tensor of shape [num_y_samples, num_features]
    Returns:
        a distance matrix of dimensions [num_x_samples, num_y_samples].
    Raises:
        ValueError: if the inputs do no matched the specified dimensions.
    """

    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)

  # By making the `inner' dimensions of the two matrices equal to 1 using
  # broadcasting then we are essentially substracting every pair of rows
  # of x and y.
  # x will be num_samples x num_features x 1,
  # and y will be 1 x num_features x num_samples (after broadcasting).
  # After the substraction we will get a
  # num_x_samples x num_features x num_y_samples matrix.
  # The resulting dist will be of shape num_y_samples x num_x_samples.
  # and thus we need to transpose it again.
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))

def kl_dist_tf(x, y):
    X = tf.distributions.Categorical(probs=x)
    Y = tf.distributions.Categorical(probs=y)
    return tf.distributions.kl_divergence(X, Y)

def triplet_loss_KL_distribution(embeddings, labels, margin=0.3, sigmas=[1.25]):
    #     sigmas = [0.75]
    #     margin = 0.3

    #     pdist_matrix = pairwise_distance(embeddings, squared=False)
    pdist_matrix = compute_pairwise_distances(embeddings, embeddings)
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    s = tf.matmul(beta, tf.reshape(pdist_matrix, (1, -1)))
    pdist_matrix_pos = tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(pdist_matrix))
    #     pdist_matrix_neg = tf.reshape(tf.reduce_sum(tf.exp(margin - s), 0), tf.shape(pdist_matrix))

    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask_negatives = tf.to_float(tf.logical_not(labels_equal))
    #     negatives_dist = tf.multiply(mask_negatives, pdist_matrix_neg)

    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    mask_positives = tf.to_float(tf.logical_and(indices_not_equal, labels_equal))

    #     positives_dist = tf.multiply(mask_positives, pdist_matrix_pos)
    #     positives_dist = positives_dist / tf.norm(positives_dist, ord=1, axis=1, keepdims=True)

    mask_anchors = tf.to_float(labels_equal)
    #     anchors_dist = tf.multiply(mask_anchors, pdist_matrix_pos)
    anchors_dist = pdist_matrix_pos
    anchors_dist = anchors_dist / tf.norm(anchors_dist, ord=1, axis=1, keepdims=True)

    #     kl_div_tf = kl_divergence_tf(tf.gather_nd(anchors_dist, [tf.constant(0)]), tf.gather_nd(anchors_dist, [tf.constant(0)]))

    #     kl_div_pw_tf = kl_divergence_pairwise_tf(anchors_dist, anchors_dist)
    #     kl_div_pw_tf = pdist_matrix

    rep_anchors_rows = tf.reshape(tf.tile(anchors_dist, [1, tf.shape(anchors_dist)[0]]),
                                  [tf.size(anchors_dist), tf.shape(anchors_dist)[0]])

    rep_anchors_matrices = tf.tile(anchors_dist, [tf.shape(anchors_dist)[0], 1])

    #     ce_loss_pw = tf.nn.softmax_cross_entropy_with_logits_v2(logits=rep_anchors[0], labels=tf.nn.softmax(rep_anchors_v2[1]))
    #     import ipdb; ipdb.set_trace()
    #     kl_loss_pw = kl_divergence_tf(rep_anchors_rows, rep_anchors_matrices)
    kl_loss_pw = kl_dist_tf(rep_anchors_rows + 1E-6, rep_anchors_matrices + 1E-6)
    #     kl_loss_pw = ce_loss(rep_anchors_rows, rep_anchors_matrices)

    #     kl_loss_pw = tf.reduce_sum(kl_loss_pw, axis=1)
    kl_loss_pw = tf.reshape(kl_loss_pw, [tf.shape(anchors_dist)[0], tf.shape(anchors_dist)[0]])

    kl_div_pw_pos = tf.multiply(mask_anchors, kl_loss_pw)
    kl_div_pw_neg = tf.multiply(mask_negatives, kl_loss_pw)

    kl_loss = tf.reduce_mean(kl_div_pw_pos, axis=1, keepdims=True) - tf.reduce_mean(kl_div_pw_neg, axis=1,
                                                                                    keepdims=True) + tf.constant(margin,
                                                                                                                 tf.float32)

    kl_loss = tf.maximum(kl_loss, 0.0)
    # #    kl_loss_filtered = tf.where(tf.less(kl_loss, tf.constant(0., tf.float32)), tf.zeros_like(kl_loss, tf.float32), kl_loss)

    return tf.reduce_mean(kl_loss)