import argparse
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops as ops_tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Need Tk for interactive plots.
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.manifold import TSNE
from skimage.transform import resize
import scipy.io as sio
import scipy.stats
import h5py
import random
# from utils import *
from dataset_builders.dataset_loader import MNISTDataset
from dirl_core import triplet_loss_distribution
from dirl_core import dirl_utils
from sklearn.neighbors import KDTree
from sklearn.metrics import accuracy_score, silhouette_score, normalized_mutual_info_score

tf.logging.set_verbosity(tf.logging.INFO)


# flags = tf.app.flags
# flags.DEFINE_string('dataset_name', '', 'Name of the dataset.')
# flags.DEFINE_string('train_config_path', '', 'path to the config file.')
# flags.DEFINE_string('split_name', '', 'train or test or validation split of the dataset.')
# flags.DEFINE_string('model_name', '', 'Name of the model to train the algorithm.')
# flags.DEFINE_string('train_dir', '',
#                     'Directory to save the checkpoints and training summaries.')
# flags.DEFINE_string('train_tfrecord_path', '',
#                     'path to the tf record of the training dataset.')
# flags.DEFINE_string('test_tfrecord_path', '',
#                     'path to the tf record of the testing dataset.')

def digitsNet(x_in, config, l_value, l_value2):

    l2_weight = config.model.l2_regularizer_weight
    emb_size = config.model.embedding_size
    flip_gradient = dirl_utils.FlipGradientBuilder()
    init = tf.contrib.layers.variance_scaling_initializer
    is_training = True
    # define model architecture
    with tf.variable_scope("gen"):
        net = x_in
        if config.model.input_normalize:
            mean = tf.reduce_mean(net, [1, 2], True)
            std = tf.reduce_mean(tf.square(net - mean), [1, 2], True)
            net = (net - mean) / (std + 1e-5)

        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                activation_fn=tf.nn.leaky_relu,
                weights_regularizer=slim.l2_regularizer(l2_weight)):
            with slim.arg_scope([slim.dropout], is_training=is_training):
                net = slim.conv2d(net, 32, [3, 3], scope='conv1')
                net = slim.conv2d(net, 32, [3, 3], scope='conv1_2')
                net = slim.conv2d(net, 32, [3, 3], scope='conv1_3')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 14
                net = slim.conv2d(net, 64, [3, 3], scope='conv2_1')
                net = slim.conv2d(net, 64, [3, 3], scope='conv2_2')
                net = slim.conv2d(net, 64, [3, 3], scope='conv2_3')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 7
                net = slim.conv2d(net, 128, [3, 3], scope='conv3')
                net = slim.conv2d(net, 128, [3, 3], scope='conv3_2')
                net = slim.conv2d(net, 128, [3, 3], scope='conv3_3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 3
                net = slim.flatten(net, scope='flatten')

                with slim.arg_scope([slim.fully_connected], normalizer_fn=None):
                    emb = slim.fully_connected(net, emb_size, scope='fc1')

    x_features = emb

    # DANN
    # l_value = tf.placeholder(tf.float32, [])
    ads_feats = flip_gradient(x_features, l_value)
    d_classify = tf.layers.dense(ads_feats, 100, activation=tf.nn.leaky_relu, kernel_initializer=init())
    d_classify = tf.layers.dense(d_classify, 50, activation=tf.nn.leaky_relu, kernel_initializer=init())
    d_classify = tf.layers.dense(d_classify, 2, activation=None, kernel_initializer=init())

    # classify
    s_classify = tf.layers.dense(x_features, 100, activation=tf.nn.leaky_relu, kernel_initializer=init())
    s_classify = tf.layers.dense(s_classify, 50, activation=tf.nn.leaky_relu, kernel_initializer=init())
    s_classify = tf.layers.dense(s_classify, num_classes, activation=None, kernel_initializer=init())

    class_danns = []
    # l_value2 = tf.placeholder(tf.float32, [])
    ads_feats_cdann = flip_gradient(x_features, l_value2)
    with tf.variable_scope("c_classify"):
        for i in range(num_classes):
            c_classify = tf.layers.dense(ads_feats_cdann, 100, activation=tf.nn.leaky_relu, kernel_initializer=init())
            c_classify = tf.layers.dense(c_classify, 50, activation=tf.nn.leaky_relu, kernel_initializer=init())
            c_classify = tf.layers.dense(c_classify, 2, activation=None, kernel_initializer=init())
            class_danns.append(c_classify)

    return x_features, d_classify, s_classify, class_danns

def few_labels(data, labels, num_pts):
    num = data.shape[0]
    data_subset = []
    labels_subset = []
    for i in range(10):
        filterr = np.where(np.argmax(labels, axis = 1) == i)
        data_subset.append(data[filterr][0:num_pts])
        labels_subset.append(labels[filterr][0:num_pts])
    return np.concatenate(data_subset, axis = 0), np.concatenate(labels_subset, axis = 0)

def plot_embedding(X, y, d, title=None, save_fig_path='tmp.png'):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    print(x_min, x_max)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(7,7))
#     ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'size': 14})

    plt.xticks([]), plt.yticks([])
    plt.xlim(min(X[:,0]), max(X[:,0])+0.05)
    plt.ylim(min(X[:,1]), max(X[:,1]) + 0.05)
    if title is not None:
        plt.title(title)

    plt.savefig(save_fig_path, format='png', bbox_inches='tight', pad_inches=2)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default="dirl",
                        choices=["source_only", "triplet", "dann", "dirl"])
    parser.add_argument('-source', type=str, default='mnist',
                        choices=["mnist", "mnistm", "svhn", "usps"])
    parser.add_argument('-target', type=str, default='mnistm',
                        choices=["mnist", "mnistm", "svhn", "usps"])
    parser.add_argument('-num_target_labels', type=int, default=10)
    parser.add_argument('-num_iterations', type=int, default=None)
    parser.add_argument('-config_path', type=str, default='configs/dirl_digits_config.yml')
    parser.add_argument('-save_results', type=bool, default=False)
    opts = parser.parse_args()


    # Load the config file in T struct
    logdir= os.path.join(os.getcwd(), 'results')

    if not os.path.exists(logdir):
        os.mkdir(logdir)

    config = dirl_utils.ParseConfigsToLuaTable(opts.config_path, save=opts.save_results, logdir=logdir, save_name='final_training_config_' + opts.mode + '-' + opts.source + '-' + opts.target + '-' + str(opts.num_target_labels) + '.yml')

    # Load source and target dataset
    datasets_directory = os.path.join(os.getcwd(), config.dataset.datasets_directory)
    mnd = MNISTDataset(datasets_directory)
    source_data, source_data_test, source_labels, source_labels_test = mnd.get_dataset(opts.source)
    target_data, target_data_test, target_labels, target_labels_test = mnd.get_dataset(opts.target)


    curr_pts = opts.num_target_labels
    num_classes = config.dataset.num_classes
    target_sup_size = curr_pts * num_classes
    sizing = [target_sup_size * 4, target_sup_size, target_sup_size * 3]
    batch_size = sum(sizing)
    print(batch_size, sum(sizing[0:2]))

    target_data_sup, target_labels_sup = few_labels(target_data, target_labels, curr_pts)

    gen_batch_source = dirl_utils.batch_generator([source_data, source_labels], sizing[0], iter_shuffle=False)
    gen_batch_target = dirl_utils.batch_generator([target_data, target_labels], sizing[2], iter_shuffle=True)

    # 0 = real, 1 = fake
    domain_labels = np.vstack([np.tile([1, 0], [batch_size // 2, 1]), np.tile([0, 1], [batch_size // 2, 1])])
    adv_labels = np.tile([1, 0], [batch_size, 1])


    # Define TF Placeholders
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, config.dataset.input_dim, config.dataset.input_dim, 3])
    y = tf.placeholder(tf.float32, [None, config.dataset.num_classes])
    domain = tf.placeholder(tf.float32, [None, 2])
    adv_domain = tf.placeholder(tf.float32, [None, 2])
    l_value = tf.placeholder(tf.float32, [])
    l_value2 = tf.placeholder(tf.float32, [])

    x_features, d_classify, s_classify, class_danns = digitsNet(x, config, l_value, l_value2)


    #define loss functions and accuracy metrics
    batch_size_tf = tf.shape(x_features)[0]

    ## domain loss
    domain_weight = config.loss.domain_weight
    domain_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_classify, labels=domain))
    domain_accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(domain, axis=1), tf.argmax(d_classify, axis=1)), tf.float32))

    adv_domain_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_classify[batch_size // 2:],
                                                                             labels=adv_domain[batch_size // 2:]))

    # Only impose the loss on the source domain and fewshot target
    classify_weight = config.loss.classify_weight
    classify_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=s_classify[:sum(sizing[0:2])], labels=y[:sum(sizing[0:2])]))

    interim = tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(s_classify, axis=1)), tf.float32)
    classify_source_accuracy = tf.reduce_mean(interim[:sizing[0]])
    classify_target_sup_accuracy = tf.reduce_mean(interim[sizing[0]:sizing[0] + sizing[1]])
    classify_target_accuracy = tf.reduce_mean(interim[sizing[0]:])

    # Triplet loss
    # triplet_weight = config.loss.triplet_weight
    # triplet_margin = config.loss.triplet_margin #0.7  # 1.0 #1e-4
    # triplet_loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(
    #     labels=tf.argmax(y, axis=1)[:sum(sizing[0:2])],
    #     embeddings=tf.nn.l2_normalize(x_features[:sum(sizing[0:2])], axis=1), margin=triplet_margin)

    triplet_KL_margin = config.loss.triplet_KL_margin
    triplet_KL_weight = config.loss.triplet_KL_weight

    norm_embeddings = tf.nn.l2_normalize(x_features[:sum(sizing[0:2])], axis = 1)
    triplet_loss = triplet_loss_distribution.triplet_loss_KL_distribution(norm_embeddings,labels=tf.argmax(y, axis = 1)[:sum(sizing[0:2])], margin=triplet_KL_margin, sigmas=[0.01, 0.1, 0.5, 1.1])
    triplet_loss = tf.scalar_mul(triplet_KL_weight, triplet_loss)

    # Class dann loss
    c_dann_weight = config.loss.class_dann_weight
    c_dann_weight_b = config.loss.class_dann_weight
    c_dann_losses = []
    c_dann_losses_B = []
    for i in range(num_classes):
        is_class_source = tf.equal(tf.argmax(y[:sizing[0]], axis=1), i)
        masked_domain_source = tf.boolean_mask(domain[:sizing[0]], is_class_source)
        masked_class_dann_source = tf.boolean_mask(class_danns[i][:sizing[0]], is_class_source)

        masked_adv_domain_source = tf.boolean_mask(adv_domain[:sizing[0]], is_class_source)

        is_class_target_l = tf.equal(tf.argmax(y[sizing[0]:sum(sizing[0:2])], axis=1), i)
        masked_domain_target_l = tf.boolean_mask(domain[sizing[0]:sum(sizing[0:2])], is_class_target_l)
        masked_class_dann_target_l = tf.boolean_mask(class_danns[i][sizing[0]:sum(sizing[0:2])], is_class_target_l)

        masked_adv_domain_target_l = tf.boolean_mask(adv_domain[sizing[0]:sum(sizing[0:2])], is_class_target_l)

        m_shape_ratio = tf.maximum(tf.shape(masked_domain_source)[0] // tf.shape(masked_domain_target_l)[0], 1)

        masked_domain = tf.concat([masked_domain_source, masked_domain_target_l], axis=0)
        masked_class_dann = tf.concat([masked_class_dann_source, masked_class_dann_target_l], axis=0)

        masked_domain_adv = tf.concat([masked_adv_domain_source, masked_adv_domain_target_l], axis=0)

        class_dann_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=masked_class_dann, labels=masked_domain))

        class_dann_loss_B = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=masked_class_dann_target_l, labels=masked_adv_domain_target_l))

        c_dann_losses.append(class_dann_loss)
        c_dann_losses_B.append(class_dann_loss_B)

    c_dann_losses_mean = tf.reduce_mean(c_dann_losses)
    c_dann_losses_mean_B = tf.reduce_mean(c_dann_losses_B)

    source_accuracy_fin = tf.reduce_mean(interim[:batch_size_tf // 2])
    target_accuracy_fin = tf.reduce_mean(interim[batch_size_tf // 2:])

    # Define optimizers
    stepB_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'gen')

    if opts.mode == 'source_only':
        loss_A  = classify_weight * classify_loss
        loss_B = 0.0 * classify_loss
    elif opts.mode == 'triplet':
        loss_A  = classify_weight * classify_loss
        loss_B = triplet_loss
    elif opts.mode == 'dann':
        loss_A = classify_weight * classify_loss + domain_weight * domain_loss
        loss_B = domain_weight * adv_domain_loss
    elif opts.mode == 'dirl':
        loss_A = classify_weight * classify_loss + c_dann_weight * c_dann_losses_mean + domain_weight * domain_loss
        loss_B = c_dann_weight_b * c_dann_losses_mean_B + domain_weight * adv_domain_loss + triplet_loss
    else:
        print('Model not supported.')

    train_A = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss_A)
    train_B = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss_B, var_list=stepB_vars)

    # Initialize session and run the optimization loop
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print_strings = []
    the_loss_classify = []
    the_loss_domain = []
    the_loss_cd_domain = []
    the_acc_target = []
    triplet_loss_list = []
    triplet_loss_KL = []
    the_l = []
    plot_losses = []
    epoch = 0
    max_test_run = 0

    num_test = config.model.test_size  # tsne
    num_test = min(num_test, source_data_test.shape[0], target_data_test.shape[0])

    source_random_indices = list(range(source_data_test.shape[0]))
    target_random_indices = list(range(target_data_test.shape[0]))
    random.shuffle(source_random_indices)
    random.shuffle(target_random_indices)
    source_test_indices = source_random_indices[:num_test]
    target_test_indices = target_random_indices[:num_test]

    combined_test_imgs = np.vstack([source_data_test[source_test_indices], target_data_test[target_test_indices]])
    combined_test_labels = np.vstack([source_labels_test[source_test_indices], target_labels_test[target_test_indices]])
    combined_test_domain = np.vstack([np.tile([1, 0.], [num_test, 1]),
                                      np.tile([0, 1.], [num_test, 1])])

    # Run training loop and log loss functions

    # Training loop is decomposed into two steps:
    # Step 1) Train entire network with classification loss, and minimize class dann domain loss on classifiers only
    # Step 2) Maximize class dann domain loss on generator only, by using inverted domain labels
    # Note: loss is configured such that the target is encouraged to move towards source, but source no penality

    safety = 0
    epoch = 0
    if opts.num_iterations is None:
        num_iterations = config.model.num_iterations
    else:
        num_iterations = opts.num_iterations

    for epoch in range(num_iterations):
        batch_xs_source, batch_ys_source = next(gen_batch_source)
        batch_xs_target, batch_ys_target = next(gen_batch_target)
        batch_xs = np.vstack([batch_xs_source, target_data_sup, batch_xs_target])
        batch_ys = np.vstack([batch_ys_source, target_labels_sup, batch_ys_target])

        if epoch > 300:
            _ = sess.run(train_B, feed_dict={x: batch_xs, y: batch_ys,
                                             domain: domain_labels, adv_domain: adv_labels, l_value: -1.0, l_value2: -1.0})

        _, domain_loss_f, domian_accu, class_l_f, class_s_a, class_t_sup_a, class_t_a, c_dann_all, c_dann_mean, triplet_loss_f = \
            sess.run([train_A, domain_loss, domain_accuracy,
                      classify_loss, classify_source_accuracy,
                      classify_target_sup_accuracy, classify_target_accuracy, c_dann_losses, c_dann_losses_mean,
                      triplet_loss],
                     feed_dict={x: batch_xs, y: batch_ys, domain: domain_labels, adv_domain: adv_labels, l_value: 0.0,
                                l_value2: 0.0})

        the_loss_domain.append(domain_loss_f)
        the_loss_classify.append(class_l_f)
        the_acc_target.append(class_t_a)
        the_loss_cd_domain.append(c_dann_mean)
        triplet_loss_list.append(triplet_loss_f)
        plot_losses.append([domain_loss_f, class_l_f, c_dann_mean, triplet_loss_f])

        if epoch % 100 == 0:
            epoch_string = "{} Batch Loss: {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
                epoch, domain_loss_f, domian_accu, class_l_f, class_s_a, class_t_sup_a, class_t_a, c_dann_mean,
                triplet_loss_f)
            print_strings.append(epoch_string)
            print(epoch_string)

            if epoch > 300:
                max_test_run = max(ab[1], max_test_run)

            ab = sess.run([source_accuracy_fin, target_accuracy_fin],
                          feed_dict={x: combined_test_imgs, y: combined_test_labels, domain: combined_test_domain})
            print(ab, sess.run([m_shape_ratio, tf.shape(masked_domain_source)[0], tf.shape(masked_domain_target_l)[0]],
                               feed_dict={x: batch_xs, y: batch_ys, domain: domain_labels}), max_test_run)

    if opts.save_results:
    # plot loss functions
        plt.figure(figsize=(10, 5))
        plt.plot(range(num_iterations), plot_losses)
        plt.legend(['domain_loss', 'classify_loss', 'class_dann_loss', 'triplet_loss'])
        plt.savefig(os.getcwd() + '/results/dirl_digits_losses_' + opts.mode + '-' + opts.source + '-'+ opts.target + '-' + str(opts.num_target_labels) + '.png', format='png', bbox_inches='tight',
                    pad_inches=2)

    print("Max Training Accuracy: ", max_test_run)
    # evaluate accuracy on source and target domains
    c1s, c1t = sess.run([source_accuracy_fin, target_accuracy_fin],
                        feed_dict={x: combined_test_imgs, y: combined_test_labels, domain: combined_test_domain})

    print("classification 1:\nsource:", c1s, "\ntarget:", c1t)

    # domain accuracy
    domain_accur_both = sess.run(domain_accuracy, feed_dict={x: combined_test_imgs, domain: combined_test_domain})
    domain_accur_source = sess.run(domain_accuracy,
                                   feed_dict={x: source_data_test[source_test_indices], domain: np.tile([1, 0.], [num_test, 1])})
    domain_accur_target = sess.run(domain_accuracy,
                                   feed_dict={x: target_data_test[target_test_indices], domain: np.tile([0, 1.], [num_test, 1])})

    print("Domain: both", domain_accur_both, "\nsource:", domain_accur_source, "\ntarget:", domain_accur_target)

    # class accuracy
    # domain_accur_both = sess.run(domain_accuracy, feed_dict={x: combined_test_imgs, domain: combined_test_domain})
    class_accur_source = sess.run(classify_source_accuracy, feed_dict={x: combined_test_imgs, y: combined_test_labels})
    class_accur_target = sess.run(classify_target_accuracy, feed_dict={x: combined_test_imgs, y: combined_test_labels})

    print("classification:\nsource:", class_accur_source, "\ntarget:", class_accur_target)

    # KNN accuracy
    neighbors = config.model.k_neighbours
    source_emb = sess.run(x_features, feed_dict={x: source_data_test[source_test_indices]})
    target_emb = sess.run(x_features, feed_dict={x: target_data_test[target_test_indices]})
    kdt = KDTree(source_emb, leaf_size=30, metric='euclidean')
    neighbor_idx = kdt.query(target_emb, k=1 + neighbors, return_distance=False)[:, 1:]
    print(neighbor_idx.shape)
    neighbor_label = source_labels_test[source_test_indices][neighbor_idx]
    neighbor_label_summed = np.sum(neighbor_label, axis=1)
    knn_accuracy = np.mean(np.argmax(target_labels_test[target_test_indices], 1) == np.argmax(neighbor_label_summed, 1))
    print(neighbor_label.shape, neighbor_label_summed.shape)
    print("Accuracy of knn on labels in z space (source)(on test)", knn_accuracy)

    # plot t-sne embedding
    num_test = config.model.tsne_test_size  # tsne
    num_test = min(num_test, source_data_test.shape[0], target_data_test.shape[0])
    source_test_indices = source_random_indices[:num_test]
    target_test_indices = target_random_indices[:num_test]
    combined_test_imgs = np.vstack([source_data_test[source_test_indices], target_data_test[target_test_indices]])
    combined_test_labels = np.vstack([source_labels_test[source_test_indices], target_labels_test[target_test_indices]])
    combined_test_domain = np.vstack([np.tile([1, 0.], [num_test, 1]),
                                      np.tile([0, 1.], [num_test, 1])])

    dann_emb = sess.run(x_features, feed_dict={x: combined_test_imgs, domain: combined_test_domain})
    dann_tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=400).fit_transform(dann_emb)
    if opts.save_results:
        plot_embedding(dann_tsne, combined_test_labels.argmax(1), combined_test_domain.reshape(-1), '', save_fig_path=os.getcwd() + '/results/dirl_digits_tsne_plot_' + opts.mode + '-' + opts.source + '-' + opts.target + '-' + str(opts.num_target_labels) + '.png')
    else:
        plot_embedding(dann_tsne, combined_test_labels.argmax(1), combined_test_domain.reshape(-1), '')

    if opts.save_results:
        # Write the accuracy metrics to the training config file
        config_output_file = os.getcwd() + '/results/final_training_config_' + opts.mode + '-' + opts.source + '-' + \
                             opts.target + '-' + str(opts.num_target_labels) + '.yml'

        outfile = open(config_output_file, "a")  # append mode

        outfile.write('\n \n Max Accuracy on target test: \t' + str(max_test_run) + '\n')
        outfile.write('Final Accuracy on source, target test: \t' + str(c1s) + ',\t' + str(c1t) + '\n')
        outfile.write('Domain Accuracy on both, source, target test: \t' + str(domain_accur_both) + ',\t' + str(domain_accur_source) + ',\t' + str(domain_accur_target) + '\n')
        outfile.write('Classicication Accuracy on source, target test: \t' + str(class_accur_source) + ',\t' + str(class_accur_target) + '\n')
        outfile.write('KNN accuracy on target test: \t' + str(knn_accuracy) + '\n')
