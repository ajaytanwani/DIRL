import argparse
import numpy as np
import os
import imageio
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')  # Need Tk for interactive plots.
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.special import softmax

from dirl_core import triplet_loss_distribution
from dirl_core import dirl_utils
from dirl_core import plot_utils

from sklearn.neighbors import KDTree
from collections import Counter
from scipy.special import softmax

# cpu_cores = [12,11,10,9,8,7] # Cores (numbered 0-11)
# os.system("taskset -pc {} {}".format(",".join(str(i) for i in cpu_cores), os.getpid()))

from tensorflow.python.client import device_lib
tf.ConfigProto().gpu_options.allow_growth = False
print([x.name for x in device_lib.list_local_devices()if x.device_type == 'GPU'])


def sample_data(mean, cov0=None, num_dim=2, num_classes=2, num_instances=100, class_ratio=0.5, noise_sf=0.15):
    std = 0.1
    N_class = []
    cov_mat = []
    X_source = np.empty((0, num_dim), float)
    Y_source = np.asarray([], dtype=int)

    N_class.append(int(num_instances * class_ratio))
    N_class.append(int(num_instances * (1.0 - class_ratio)))
    for class_id in range(num_classes):

        if cov0 is None:
            cov = np.eye(num_dim) * std
            cov_noise = np.random.randn(num_dim, num_dim) * noise_sf
            cov_noise = np.dot(cov_noise, cov_noise.transpose())
            cov += cov_noise
            cov_mat.append(cov)
        else:
            cov_mat.append(cov[class_id])

        x, y = np.random.multivariate_normal(mean[class_id], cov_mat[class_id], N_class[class_id]).T

        X = np.concatenate([x.reshape(len(x), 1), y.reshape(len(y), 1)], axis=1)
        X_source = np.append(X_source, X, axis=0)
        Y_source = np.append(Y_source, [class_id] * N_class[class_id], axis=0)

    return X_source, Y_source

def _random_choice(inputs, n_samples):
    """
    With replacement.
    Params:
      inputs (Tensor): Shape [n_states, n_features]
      n_samples (int): The number of random samples to take.
    Returns:
      sampled_inputs (Tensor): Shape [n_samples, n_features]
    """
    weights_init = tf.ones_like(inputs[:, 0], dtype=tf.float32)

    weight_per_sample = tf.div(weights_init, tf.reduce_sum(weights_init, axis=0))

    log_probs = tf.log(tf.expand_dims(weight_per_sample, 0))

    rand_indices = tf.cast(tf.squeeze(tf.multinomial(log_probs, n_samples), 0), tf.int32)

    sampled_inputs = tf.gather(inputs, rand_indices)

    return sampled_inputs


def smooth(x_list, k = 2):
    return np.array([1 / (2 * k) * np.sum([x_list[i + j] for j in range(-k, k)]) for i in range(k, len(x_list) - k)])


def sample_class_dann_logits(class_features, class_labels, class_id, batch_size, num_target_labels_tf,
                             mean_source_filtered, mean_target_filtered, class_batch_size):

    n_samples = class_batch_size // 2

    #     batch_size = tf.shape(class_features)[0]
    class_source_labels_ids = tf.equal(class_labels[:batch_size // 2], class_id)
    class_target_labels_ids = tf.equal(
        class_labels[batch_size // 2: batch_size // 2 + num_target_labels_tf],
        class_id)

    #     class_source_labels_ids = tf.Print(class_source_labels_ids, [class_id, class_source_labels_ids], "class_source_labels_ids: ", summarize=40)
    #     class_target_labels_ids = tf.Print(class_target_labels_ids, [class_id, class_target_labels_ids], "class_target_labels_ids: ", summarize=40)

    class_features_source = tf.boolean_mask(class_features[:batch_size // 2], class_source_labels_ids)
    class_features_target = tf.boolean_mask(
        class_features[batch_size // 2: batch_size // 2 + num_target_labels_tf],
        class_target_labels_ids)

    class_features_source_size = tf.equal(tf.shape(class_features_source)[0], class_id)
    class_features_source_filtered = tf.cond(class_features_source_size, lambda: mean_source_filtered[
                                                                                 class_id * class_batch_size // 2: (
                                                                                                                               class_id + 1) * class_batch_size // 2],
                                             lambda: class_features_source)

    class_features_target_size = tf.equal(tf.shape(class_features_target)[0], 0)
    class_features_target_filtered = tf.cond(class_features_target_size, lambda: mean_target_filtered[
                                                                                 class_id * class_batch_size // 2: (
                                                                                                                               class_id + 1) * class_batch_size // 2],
                                             lambda: class_features_target)

    class_features_size_bool = tf.logical_or(class_features_source_size, class_features_target_size)

    samples_class_source = _random_choice(class_features_source_filtered, n_samples)
    samples_class_target = _random_choice(class_features_target_filtered, n_samples)

    samples_class_source_labels = tf.tile(tf.constant([1, 0], dtype=tf.int64), tf.shape(samples_class_source[:, 0]))
    samples_class_source_labels = tf.reshape(samples_class_source_labels, [tf.shape(samples_class_source[:, 0])[0], 2])

    samples_class_target_labels = tf.tile(tf.constant([0, 1], dtype=tf.int64), tf.shape(samples_class_target[:, 0]))
    samples_class_target_labels = tf.reshape(samples_class_target_labels, [tf.shape(samples_class_target[:, 0])[0], 2])

    class_dann_logits = tf.concat([samples_class_source, samples_class_target], axis=0)

    class_dann_logits_labels = tf.concat([samples_class_source_labels, samples_class_target_labels], axis=0)

    return class_dann_logits, class_dann_logits_labels, class_features_size_bool


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default="dirl",
                        choices=["source_only", "triplet", "dann", "dirl"])
    parser.add_argument('-num_target_labels', type=int, default=8)
    parser.add_argument('-num_iterations', type=int, default=None)
    parser.add_argument('-config_path', type=str, default='configs/dirl_synthetic_2d_config.yml')
    parser.add_argument('-save_results', type=bool, default=True)
    opts = parser.parse_args()

    # import ipdb; ipdb.set_trace()
    logdir = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    if not os.path.exists(os.path.join(logdir, 'animations')):
        os.mkdir(os.path.join(logdir, 'animations'))
        os.mkdir(os.path.join(logdir, 'animations/gifs'))

    if not os.path.exists(os.path.join(logdir, 'figs')):
        os.mkdir(os.path.join(logdir, 'figs'))

    # Load the config file in T struct
    config = dirl_utils.ParseConfigsToLuaTable(opts.config_path, save=opts.save_results, logdir=logdir, save_name='final_training_config_2D_' + opts.mode + '-' + str(opts.num_target_labels) + '.yml')

    # Generate 2D synthetic data
    mean_source = config.dataset.source.class_means

    X_source, Y_source = sample_data(mean_source, num_dim=2, num_classes=2, num_instances=config.dataset.source.num_instances_train, class_ratio=config.dataset.source.class_ratio, noise_sf=config.dataset.source.noise_sf)

    X_source_test, Y_source_test = sample_data(mean_source, num_dim=2, num_classes=2, num_instances=config.dataset.source.num_instances_test, class_ratio=config.dataset.source.class_ratio, noise_sf=config.dataset.source.noise_sf)

    ### Target Dataset Parameters
    mean_target = config.dataset.target.class_means

    X_target, Y_target = sample_data(mean_target, num_dim=2, num_classes=2, num_instances=config.dataset.target.num_instances_train, class_ratio=config.dataset.target.class_ratio, noise_sf=config.dataset.target.noise_sf)

    num_target_labels = opts.num_target_labels
    # class_batch_size = num_target_labels

    X_labeled_target, Y_labeled_target = sample_data(mean_target, num_dim=2, num_classes=2, num_instances=num_target_labels, class_ratio=config.dataset.target.class_ratio, noise_sf=config.dataset.source.noise_sf)
    X_target_test, Y_target_test = sample_data(mean_target, num_dim=2, num_classes=2, num_instances=config.dataset.target.num_instances_test, class_ratio=config.dataset.target.class_ratio, noise_sf=config.dataset.target.noise_sf)

    # Plot the original data for visualization
    plot_utils.plot_2d_data_raw(X_source, Y_source, X_target, Y_target, X_labeled_target, Y_labeled_target)

    # Create placeholders for tensors
    x_input = tf.placeholder(tf.float32, shape=[None, 2], name='X_input')
    y_labels = tf.placeholder(tf.float32, shape=[None, 2], name='Y_labels')
    domain = tf.placeholder(tf.int64, [None, 2])
    class_domain_labels = tf.placeholder(tf.int64, [None, 2])
    l_value = tf.placeholder(tf.float32, [])
    num_target_labels_tf = tf.placeholder(tf.int32, [])

    mean_source_filtered = tf.placeholder(tf.float32, shape=[None, 2])
    mean_target_filtered = tf.placeholder(tf.float32, shape=[None, 2])

    # Define the architecture
    latent_dim = config.model.embedding_size

    # initializer = tf.contrib.layers.xavier_initializer()
    initializer = tf.keras.initializers.he_normal(seed=None)
    flip_gradient = dirl_utils.FlipGradientBuilder()
    x_in = x_input

    hidden_units = config.model.hidden_units

    with tf.variable_scope("embedder", reuse=tf.AUTO_REUSE):
        G_W1 = tf.get_variable('W1', [2, hidden_units], initializer=initializer)
        G_b1 = tf.get_variable('b1', shape=[hidden_units], initializer=tf.zeros_initializer())

        G_W2 = tf.get_variable('W2', [hidden_units, hidden_units], initializer=initializer)
        G_b2 = tf.get_variable('b2', shape=[hidden_units], initializer=initializer)

        G_W3 = tf.get_variable('W3', [hidden_units, latent_dim], initializer=initializer)
        G_b3 = tf.get_variable('b3', shape=[latent_dim], initializer=initializer)

    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        D_W1 = tf.get_variable('W1', shape=[latent_dim, hidden_units], initializer=initializer)
        D_b1 = tf.get_variable('b1', shape=[hidden_units], initializer=initializer)

        D_W2 = tf.get_variable('W2', shape=[hidden_units, hidden_units], initializer=initializer)
        D_b2 = tf.get_variable('b2', shape=[hidden_units], initializer=initializer)

        D_W3 = tf.get_variable('W3', shape=[hidden_units, 2], initializer=initializer)
        D_b3 = tf.get_variable('b3', shape=[2], initializer=initializer)

    with tf.variable_scope("source_classify", reuse=tf.AUTO_REUSE):
        S_W1 = tf.get_variable('W1', shape=[latent_dim, hidden_units], initializer=initializer)
        S_b1 = tf.get_variable('b1', shape=[hidden_units], initializer=initializer)

        S_W2 = tf.get_variable('W2', shape=[hidden_units, hidden_units], initializer=initializer)
        S_b2 = tf.get_variable('b2', shape=[hidden_units], initializer=initializer)

        S_W3 = tf.get_variable('W3', shape=[hidden_units, 2], initializer=initializer)
        S_b3 = tf.get_variable('b3', shape=[2], initializer=initializer)

    with tf.variable_scope("class_discriminator_A", reuse=tf.AUTO_REUSE):
        C1_W1 = tf.get_variable('W1', shape=[2, hidden_units], initializer=initializer)
        C1_b1 = tf.get_variable('b1', shape=[hidden_units], initializer=initializer)

        C1_W2 = tf.get_variable('W2', shape=[hidden_units, hidden_units], initializer=initializer)
        C1_b2 = tf.get_variable('b2', shape=[hidden_units], initializer=initializer)

        C1_W3 = tf.get_variable('W3', shape=[hidden_units, 2], initializer=initializer)
        C1_b3 = tf.get_variable('b3', shape=[2], initializer=initializer)

    with tf.variable_scope("class_discriminator_B", reuse=tf.AUTO_REUSE):
        C2_W1 = tf.get_variable('W1', shape=[2, hidden_units], initializer=initializer)
        C2_b1 = tf.get_variable('b1', shape=[hidden_units], initializer=initializer)

        C2_W2 = tf.get_variable('W2', shape=[hidden_units, hidden_units], initializer=initializer)
        C2_b2 = tf.get_variable('b2', shape=[hidden_units], initializer=initializer)

        C2_W3 = tf.get_variable('W3', shape=[hidden_units, 2], initializer=initializer)
        C2_b3 = tf.get_variable('b3', shape=[2], initializer=initializer)

    # Embedding
    G_h1 = tf.nn.relu(tf.matmul(x_in, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_logit = tf.matmul(G_h2, G_W3) + G_b3
    x_features = G_logit

    # DANN discriminator
    ads_features = flip_gradient(x_features, l_value)
    D_h1 = tf.nn.relu(tf.matmul(ads_features, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)

    # Source classifier 1
    S_h1 = tf.nn.relu(tf.matmul(x_features, S_W1) + S_b1)
    S_h2 = tf.nn.relu(tf.matmul(S_h1, S_W2) + S_b2)
    S_logit = tf.matmul(S_h2, S_W3) + S_b3
    S_prob = tf.nn.sigmoid(S_logit)

    batch_size = tf.shape(x_features)[0]
    class_labels = tf.argmax(y_labels[:batch_size // 2 + num_target_labels_tf], axis=1)

    class_A_dann_logits, class_A_dann_logits_labels, class_A_labels_size_bool = sample_class_dann_logits(x_features,
                                                                                                         class_labels,
                                                                                                         0,
                                                                                                         batch_size,
                                                                                                         num_target_labels_tf,
                                                                                                         mean_source_filtered,
                                                                                                         mean_target_filtered,
                                                                                                         num_target_labels)

    class_B_dann_logits, class_B_dann_logits_labels, class_B_labels_size_bool = sample_class_dann_logits(x_features,
                                                                                                         class_labels,
                                                                                                         1,
                                                                                                         batch_size,
                                                                                                         num_target_labels_tf,
                                                                                                         mean_source_filtered,
                                                                                                         mean_target_filtered,
                                                                                                         num_target_labels)

    # class dann A
    C1_features = flip_gradient(class_A_dann_logits, l_value)
    C1_h1 = tf.nn.relu(tf.matmul(C1_features, C1_W1) + C1_b1)
    C1_h2 = tf.nn.relu(tf.matmul(C1_h1, C1_W2) + C1_b2)
    C1_logit = tf.matmul(C1_h2, C1_W3) + C1_b3

    # class dann B
    C2_features = flip_gradient(class_B_dann_logits, l_value)
    C2_h1 = tf.nn.relu(tf.matmul(C2_features, C2_W1) + C2_b1)
    C2_h2 = tf.nn.relu(tf.matmul(C2_h1, C2_W2) + C2_b2)
    C2_logit = tf.matmul(C2_h2, C2_W3) + C2_b3

    # define loss functions here

    # try n-pairs loss instead, add team of classifiers, add MCD for DANN
    # triplet_weight = config.loss.triplet_weight
    # triplet_margin = config.loss.triplet_margin
    norm_embeddings = tf.nn.l2_normalize(x_features, axis=1)

    triplet_KL_margin = config.loss.triplet_KL_margin
    triplet_KL_weight = config.loss.triplet_KL_weight

    triplet_loss_class = triplet_loss_distribution.triplet_loss_KL_distribution(
        norm_embeddings[:(batch_size // 2) + num_target_labels_tf],
        labels=tf.argmax(y_labels[:batch_size // 2 + num_target_labels_tf], axis=1),
        margin=triplet_KL_margin, sigmas=[0.01, 0.1])
    # triplet_loss_domain = triplet_loss_distribution.triplet_loss_KL_distribution(norm_embeddings, labels=tf.argmax(domain,axis=1), margin=triplet_KL_margin, sigmas=[0.01, 0.1])
    triplet_loss = triplet_loss_class  # + triplet_loss_domain
    triplet_loss = tf.scalar_mul(triplet_KL_weight, triplet_loss)

    # triplet_loss_class = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels=tf.argmax(y_labels[:batch_size // 2 + num_target_labels], axis = 1), embeddings=norm_embeddings[:(batch_size // 2) + num_target_labels_tf], margin=triplet_margin)
    # triplet_loss_domain = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels=tf.argmax(domain, axis = 1), embeddings=norm_embeddings, margin=triplet_margin)
    # triplet_loss = triplet_loss_class + # triplet_loss_domain
    # triplet_loss = tf.scalar_mul(triplet_weight, triplet_loss)

    # domain_loss
    domain_weight = config.loss.domain_weight
    domain_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=D_logit, labels=domain))
    domain_loss = tf.scalar_mul(domain_weight, domain_loss)

    domain_accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(domain, axis=1), tf.argmax(D_prob, axis=1)), tf.float32))

    # conditional dann loss
    class_domain_weight = config.loss.class_dann_weight
    class_domain_A_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=C1_logit, labels=class_A_dann_logits_labels))
    class_domain_B_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=C2_logit, labels=class_B_dann_logits_labels))

    class_domain_A_loss = tf.cond(class_A_labels_size_bool, lambda: tf.constant(0.0), lambda: class_domain_A_loss)
    class_domain_B_loss = tf.cond(class_B_labels_size_bool, lambda: tf.constant(0.0), lambda: class_domain_B_loss)

    class_domain_A_loss = tf.scalar_mul(class_domain_weight, class_domain_A_loss)
    class_domain_B_loss = tf.scalar_mul(class_domain_weight, class_domain_B_loss)

    class_domain_A_accuracy_v2 = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(class_A_dann_logits_labels, axis=1), tf.argmax(C1_logit, axis=1)), tf.float32))
    class_domain_B_accuracy_v2 = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(class_B_dann_logits_labels, axis=1), tf.argmax(C2_logit, axis=1)), tf.float32))

    class_domain_A_accuracy_v2 = tf.cond(class_A_labels_size_bool, lambda: tf.constant(0.0),
                                         lambda: class_domain_A_accuracy_v2)
    class_domain_B_accuracy_v2 = tf.cond(class_B_labels_size_bool, lambda: tf.constant(0.0),
                                         lambda: class_domain_B_accuracy_v2)

    # # get class IDs
    labels_gt = tf.argmax(y_labels[:batch_size // 2 + num_target_labels_tf], axis=1)

    # classify loss: Note that we do not take target labels into account for classification loss in this example to evaluate the alignment.
    classify_weight = config.loss.classify_weight
    #     classify_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=c_logits[:batch_size // 2 + num_target_labels_tf], labels=y_labels[:batch_size // 2  + num_target_labels_tf]))
    classify_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=S_logit[:batch_size // 2],
                                                       labels=y_labels[:batch_size // 2]))
    classify_loss = tf.scalar_mul(classify_weight, classify_loss)
    #     classify_source_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_labels[:batch_size // 2 + num_target_labels_tf], axis=1), tf.argmax(c_logits[:batch_size // 2 + num_target_labels_tf], axis=1)), tf.float32))
    #     classify_target_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_labels[batch_size // 2 + num_target_labels_tf:], axis=1), tf.argmax(c_logits[batch_size // 2 + num_target_labels_tf:], axis=1)), tf.float32))

    classify_source_accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(y_labels[:batch_size // 2], axis=1), tf.argmax(S_logit[:batch_size // 2], axis=1)),
        tf.float32))
    classify_target_accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(y_labels[batch_size // 2:], axis=1), tf.argmax(S_logit[batch_size // 2:], axis=1)),
        tf.float32))

    # minimize the regularized entropy on the target data
    reg_entropy_weight = config.loss.entropy_weight
    reg_entropy_loss = -tf.reduce_sum(
        -tf.nn.softmax(S_logit[batch_size // 2:]) * tf.nn.log_softmax(S_logit[batch_size // 2:]))
    reg_entropy_loss = tf.scalar_mul(reg_entropy_weight, reg_entropy_loss)

    if opts.mode == 'source_only':
        dirl_loss  = classify_loss
    elif opts.mode == 'triplet':
        dirl_loss = classify_loss + triplet_loss
    elif opts.mode == 'dann':
        dirl_loss = classify_loss + domain_loss
    elif opts.mode == 'dirl':
        dirl_loss = domain_loss + classify_loss + class_domain_A_loss + class_domain_B_loss + triplet_loss
    else:
        print('Model not supported.')

    ###

    # Define the optimizer and training pipeline
    train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(dirl_loss)

    if opts.num_iterations is None:
        num_iterations = config.model.num_iterations
    else:
        num_iterations = opts.num_iterations

    batch_factor = config.model.batch_factor #1 for both source and target, 2 for just source domain labels
    num_trials = config.model.num_trials
    output_name = '2d_synthetic_' + opts.mode + '-' + str(opts.num_target_labels)
    save_gif_images = True
    display_step = int(num_iterations / 10)
    running_avg_num_steps = 100

    batch_size = config.model.batch_size

    l_v = config.loss.reverse_grad_scale_factor

    all_metrics_across_trials = []
    overall_metrics_across_trials = []
    all_sessions = []

    mean_source_filtered_data = np.zeros((num_target_labels, 2))
    mean_target_filtered_data = np.zeros((num_target_labels, 2))

    for class_id in range(2):
        mean_source_filtered_data[class_id * num_target_labels // 2: (class_id + 1) * num_target_labels // 2] = np.tile(
            mean_source[class_id], [num_target_labels // 2, 1])
        mean_target_filtered_data[class_id * num_target_labels // 2: (class_id + 1) * num_target_labels // 2] = np.tile(
            mean_target[class_id], [num_target_labels // 2, 1])

    domain_labels = np.vstack([np.tile([1, 0], [batch_size // 2, 1]), np.tile([0, 1], [batch_size // 2, 1])])
    batch_class_domain_labels = np.vstack(
        [np.tile([1, 0], [num_target_labels // 2, 1]), np.tile([0, 1], [num_target_labels // 2, 1])])

    for trial_num in range(num_trials):
        print('trial:', trial_num)
        print('dirl_loss', 'domain_loss', 'class_dann_loss', 'classify_loss', 'triplet_loss', 'reg_entropy')

        iter_list = []
        all_metrics = []
        gif_images_list = []

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        for i in range(num_iterations + 1):
            # import ipdb; ipdb.set_trace()
            source_ids = np.random.choice(len(X_source), batch_size // 2)
            sample_source = X_source[source_ids]
            sample_source_labels = np.eye(2)[Y_source[source_ids]]

            if num_target_labels > 0:
                target_labeled_ids = np.random.choice(len(X_labeled_target), num_target_labels)
                sample_labeled_target = X_labeled_target[target_labeled_ids]
                sample_labeled_target_labels = np.eye(2)[Y_labeled_target[target_labeled_ids]]

            target_ids = np.random.choice(len(X_target), batch_size // 2 - num_target_labels)
            sample_target = X_target[target_ids]
            sample_target_labels = np.eye(2)[Y_target[target_ids]]

            if num_target_labels > 0:
                batch_xs = np.vstack([sample_source, sample_labeled_target])
                batch_ys = np.vstack([sample_source_labels, sample_labeled_target_labels])
            else:
                batch_xs = sample_source
                batch_ys = sample_source_labels

            batch_xs = np.vstack([batch_xs, sample_target])

            batch_ys = np.vstack([batch_ys, sample_target_labels])

            acc_ops = [classify_source_accuracy, classify_target_accuracy, domain_accuracy]
            loss_ops = [dirl_loss, domain_loss, class_domain_A_loss, class_domain_B_loss, classify_loss, triplet_loss, reg_entropy_loss]

            ops = [train_op] + acc_ops + loss_ops

            metrics = sess.run(ops, feed_dict={x_input: batch_xs, y_labels: batch_ys, domain: domain_labels,
                                               class_domain_labels: batch_class_domain_labels, l_value: l_v,
                                               num_target_labels_tf: num_target_labels,
                                               mean_source_filtered: mean_source_filtered_data,
                                               mean_target_filtered: mean_target_filtered_data})[1:]

            if save_gif_images and i % display_step == 0:
                plot_utils.overall_figure(sess, S_logit, G_logit, S_prob, X_source_test, Y_source_test, X_target_test,
                                          Y_target_test, logdir + '/animations/' + output_name + str(
                        trial_num) + '_' + str(i), x_input, X_labeled_target, Y_labeled_target, iter_count=i,
                                          save_gif_images=save_gif_images)
                image_handle = imageio.imread(logdir + '/animations/' + output_name + str(trial_num) + '_' + str(i) + '.png')
                if i == num_iterations:
                    gif_images_list.extend([image_handle] * 5)
                else:
                    gif_images_list.append(image_handle)

            all_metrics.append(metrics)

            iter_list.append(i)

            if i % display_step == 0:
                running_metrics = np.array(all_metrics)
                running_metrics = np.array(all_metrics[-running_avg_num_steps:])
                running_metrics = np.round(np.mean(running_metrics, axis=0), decimals=2)
                print(i, '\t accs:', running_metrics[:len(acc_ops)], '\t losses:', running_metrics[len(acc_ops):])

        all_metrics_across_trials.append(all_metrics)

        if opts.save_results:
            # plot loss functions
            plt.figure(figsize=(10, 5))
            plt.plot(range(num_iterations+1), [plot_metrics[len(acc_ops):] for plot_metrics in all_metrics])
            plt.legend(['total_loss', 'domain_loss', 'cond_loss_A', 'cond_loss_B', 'classify_loss', 'triplet_loss', 'reg_entropy'])
            plt.savefig(
                logdir + '/figs/losses_' + output_name + '-' + str(trial_num) + '.png', format='png', bbox_inches='tight',
                pad_inches=2)

        if save_gif_images:
            # Save GIF
            imageio.mimsave(logdir + '/animations/gifs/' + output_name + str(trial_num) + '.gif',
                            gif_images_list, format='GIF', duration=1.0)

        print('')

        X_source_trans, X_source_g, Y_source_pred = sess.run([S_logit, G_logit, S_prob], feed_dict={x_input: X_source})
        X_target_trans, X_target_g, Y_target_pred = sess.run([S_logit, G_logit, S_prob], feed_dict={x_input: X_target})

        source_accuracy = accuracy_score(Y_source, np.argmax(Y_source_pred, axis=1))
        target_accuracy = accuracy_score(Y_target, np.argmax(Y_target_pred, axis=1))

        overall_metrics = [source_accuracy, target_accuracy]
        print('\t overall:', overall_metrics)
        overall_metrics_across_trials.append(overall_metrics)

        plot_utils.overall_figure(sess, S_logit, G_logit, S_prob, X_source_test, Y_source_test, X_target_test, Y_target_test,
                       logdir + '/figs/' + output_name + '-' + str(trial_num), x_input, X_labeled_target,
                       Y_labeled_target, iter_count=i)

        all_sessions.append(sess)

        print('')

    print('average across trials')

    all_metrics_across_trials = np.array(all_metrics_across_trials)
    all_metrics_averaged = np.mean(all_metrics_across_trials, axis=0)
    for i in range(0, len(all_metrics_averaged), display_step):
        start_idx = max(-running_avg_num_steps + i, 0)
        end_idx = i + 1
        running_metrics = np.array(all_metrics_averaged[start_idx:end_idx])
        running_metrics = np.round(np.mean(running_metrics, axis=0), decimals=2)
        print(i, '\t accs:', running_metrics[:len(acc_ops)], '\t losses:', running_metrics[len(acc_ops):])
    print('')

    overall_metrics_averaged = np.mean(overall_metrics_across_trials, axis=0)
    print('\t overall: ', overall_metrics_averaged)

    if opts.save_results:
        # Write the accuracy metrics to the training config file
        config_output_file = os.path.join(logdir, 'final_training_config_2D_' + opts.mode + '-' + str(opts.num_target_labels) + '.yml')

        outfile = open(config_output_file, "a")  # append mode

        for trial_num, overall_metrics in enumerate(overall_metrics_across_trials):
            outfile.write('\n \n trial: {} ,\t source_accuracy {}, \t target_accuracy {}'.format(trial_num, overall_metrics[0], overall_metrics[1]))