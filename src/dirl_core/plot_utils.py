import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec
import pandas as pd
from tensorflow.python.framework import ops
from dirl_core import dirl_utils
from sklearn.metrics import accuracy_score


colormap = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])

def imshow_grid(images, shape=[2, 8]):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

    plt.show()


def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def colored_contour(x, y, Z, levels, cmap, alpha):
    def contour_plot(*args, **kwargs):
        args = (x, y, Z, levels)
        kwargs['cmap'] = cmap
        kwargs['alpha'] = alpha

        plt.contourf(*args, **kwargs)

    return contour_plot


def colored_contour_line(x, y, Z, levels, cmap, alpha):
    def contour_plot(*args, **kwargs):
        args = (x, y, Z, levels)
        kwargs['cmap'] = cmap
        kwargs['alpha'] = alpha
        plt.contour(*args, **kwargs)

    return contour_plot


def colored_density(x, y, c=None, class_name=None, domain=None):
    def density_plot(*args, **kwargs):
        args = (x, y)
        kwargs['cmap'] = c
        kwargs['shade'] = False
        kwargs['alpha'] = .7
        kwargs['n_levels'] = 2
        kwargs['shade_lowest'] = False
        kwargs['legend'] = True
        sns.kdeplot(*args, **kwargs)

    return density_plot


def colored_scatter(x, y, c=None, domain=None):
    def scatter(*args, **kwargs):
        args = (x, y)
        if domain is 'unlabeled' or domain is "target_g" or domain is "target_trans" or domain is "target_pred":
            kwargs['facecolors'] = 'none'
            kwargs['edgecolors'] = c
            kwargs['s'] = 120
            kwargs['linewidths'] = 1.1
            kwargs['alpha'] = 1.0  # scatter_alpha
        elif c is not None:
            kwargs['c'] = c
            kwargs['s'] = 120
            kwargs['alpha'] = 0.6  # scatter_alpha
        plt.scatter(*args, **kwargs)

    return scatter


def colored_markers(x, y, c=None, domain=None):
    def scatter(*args, **kwargs):
        args = (x, y)
        if domain is 'unlabeled':
            kwargs['color'] = c
        elif c is not None:
            kwargs['color'] = c
        kwargs['alpha'] = 1.0  # scatter_alpha
        plt.plot(*args, **kwargs)

    return markers


def plot_2d_data_raw(X_source, Y_source, X_target, Y_target, X_labeled_target, Y_labeled_target):
    Y_labeled_target_plot = [y + 3 if y == 0 else y + 1 for y in Y_labeled_target]
    df1 = pd.DataFrame(X_source, columns=['x1', 'x2'])
    if len(X_labeled_target) == 0:
        df2 = pd.DataFrame(X_target, columns=['x1', 'x2'])
    else:
        df2 = pd.DataFrame(np.vstack([X_target, X_labeled_target]), columns=['x1', 'x2'])
    df3 = pd.DataFrame(X_labeled_target, columns=['x1', 'x2'])
    df1['kind'] = 'labeled'
    df2['kind'] = 'unlabeled'
    df3['kind'] = 'ztarget_labeled'

    df1['label_name'] = 'source'
    df2['label_name'] = 'target'
    df3['label_name'] = 'target_labeled'

    df1['label'] = Y_source
    if len(X_labeled_target) == 0:
        df2['label'] = Y_target
    else:
        df2['label'] = np.concatenate([Y_target, Y_labeled_target])
    df3['label'] = Y_labeled_target_plot
    df1['color'] = "r"
    df2['color'] = "g"

    if len(X_labeled_target) == 0:
        df = pd.concat([df1, df2], sort=False)
    else:
        df = pd.concat([df1, df2, df3], sort=False)

    g = sns.JointGrid(x='x1', y='x2', data=df)

    #     import ipdb; ipdb.set_trace()
    legends = []
    for name, df_group in df.groupby('kind'):

        g.plot_joint(
            colored_scatter(df_group['x1'].values, df_group['x2'].values, c=colormap[df_group['label']], domain=name),
        )

        if name != 'ztarget_labeled':
            legends.append(name)
            g.plot_joint(colored_density(df_group['x1'].values[df_group['label'] == 0],
                                         df_group['x2'].values[df_group['label'] == 0], c='Blues_r'),
                         )

            g.plot_joint(colored_density(df_group['x1'].values[df_group['label'] == 1],
                                         df_group['x2'].values[df_group['label'] == 1], c='Reds_r'),
                         )

            if name == "unlabeled":
                color = "r"
            else:
                color = "g"

            sns.distplot(
                df_group['x1'].values,
                ax=g.ax_marg_x,
                color=color,
                label=df_group['label_name']
            )

            sns.distplot(
                df_group['x2'].values,
                ax=g.ax_marg_y,
                color=color,
                vertical=True,
                label=df_group['label_name']
            )

        plt.legend(legends)

    plt.setp(g.ax_marg_x.legend(), visible=True)

def draw_plot_df(g, df, draw_marginals=False, draw_class_contours=False):
    legends = []
    for name, df_group in df.groupby('kind'):

        if draw_class_contours and name is not "ztarget_labeled":

            if name is 'unlabeled' or name is "target_g" or name is "target_trans" or name is "target_pred":
                g.plot_joint(
                    colored_density(df_group['x1'].values[df_group['label'] == 0],
                                    df_group['x2'].values[df_group['label'] == 0], c="Blues_r"),
                )

                g.plot_joint(
                    colored_density(df_group['x1'].values[df_group['label'] == 1],
                                    df_group['x2'].values[df_group['label'] == 1], c="Reds_r"),
                )
            else:
                g.plot_joint(
                    colored_density(df_group['x1'].values[df_group['label'] == 0],
                                    df_group['x2'].values[df_group['label'] == 0], c="Blues_r"),
                )

                g.plot_joint(
                    colored_density(df_group['x1'].values[df_group['label'] == 1],
                                    df_group['x2'].values[df_group['label'] == 1], c="Reds_r"),
                )

            legends.append(name)

        g1 = g.plot_joint(
            colored_scatter(df_group['x1'].values, df_group['x2'].values, c=colormap[df_group['label']], domain=name),
        )

        if name == "unlabeled" or name == "target_g" or name == "target_trans" or name == "target_pred":
            color = "r"
        else:
            color = "g"

        if draw_marginals and name is not "ztarget_labeled":
            sns.distplot(
                df_group['x1'].values,
                ax=g.ax_marg_x,
                color=color,
                label=df_group['label_name']
            )

            sns.distplot(
                df_group['x2'].values,
                ax=g.ax_marg_y,
                color=color,
                vertical=True,
                label=df_group['label_name']
            )
    if not draw_marginals:
        plt.setp(g.ax_marg_x.axes, visible=False)
        plt.setp(g.ax_marg_y.axes, visible=False)


#     else:
#         plt.setp(g.ax_marg_x.legend(), visible=True)


def set_style():
    sns.set_context("paper")

    # Set the font to be serif, rather than sans
    sns.set(font='serif')

    # Make the background white, and specify the
    # specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


def get_min_max(source_data, target_data, labeled_target_data, margin=0.5):
    stacked_data = np.vstack([source_data, target_data])
    stacked_data = np.vstack([stacked_data, labeled_target_data])
    xmin = np.min(stacked_data, axis=0) - margin
    xmax = np.max(stacked_data, axis=0) + margin

    return xmin, xmax


def overall_figure(sess, S_logit, G_logit, S_prob, X_source_test, Y_source_test, X_target_test, Y_target_test,
                   fig_name='', x_input=None, X_labeled_target=[], Y_labeled_target=[], Y_target=[], iter_count=None,
                   save_gif_images=False):
    set_style()
    sns.set_context("notebook", font_scale=1.4, rc={"lines.linewidth": 1.7})

    X_source_trans, X_source_g, Y_source_pred = sess.run([S_logit, G_logit, S_prob], feed_dict={x_input: X_source_test})
    X_target_trans, X_target_g, Y_target_pred = sess.run([S_logit, G_logit, S_prob], feed_dict={x_input: X_target_test})

    target_accuracy = accuracy_score(Y_target_test, np.argmax(Y_target_pred, axis=1))

    if len(X_labeled_target) > 0:
        X_target_labeled_trans, X_target_labeled_g, Y_target_labeled_pred = sess.run([S_logit, G_logit, S_prob],
                                                                                     feed_dict={
                                                                                         x_input: X_labeled_target})
    else:
        X_target_labeled_trans, X_target_labeled_g, Y_target_labeled_pred = [], [], []

    #### draw distribution plots
    df1 = pd.DataFrame(X_source_test, columns=['x1', 'x2'])
    if len(X_labeled_target) > 0:
        df2 = pd.DataFrame(np.vstack([X_target_test, X_labeled_target]), columns=['x1', 'x2'])
    else:
        df2 = pd.DataFrame(X_target_test, columns=['x1', 'x2'])

    df3 = pd.DataFrame(X_labeled_target, columns=['x1', 'x2'])

    df4 = pd.DataFrame(X_source_g, columns=['x1', 'x2'])
    df5 = pd.DataFrame(X_target_g, columns=['x1', 'x2'])

    df5a = pd.DataFrame(X_target_labeled_g, columns=['x1', 'x2'])

    df6 = pd.DataFrame(X_source_trans, columns=['x1', 'x2'])
    df7 = pd.DataFrame(X_target_trans, columns=['x1', 'x2'])

    df7a = pd.DataFrame(X_target_labeled_trans, columns=['x1', 'x2'])

    df8 = pd.DataFrame(Y_source_pred, columns=['x1', 'x2'])
    df9 = pd.DataFrame(Y_target_pred, columns=['x1', 'x2'])

    df9a = pd.DataFrame(Y_target_labeled_pred, columns=['x1', 'x2'])

    df1['kind'] = 'labeled'
    df2['kind'] = 'unlabeled'
    df3['kind'] = 'ztarget_labeled'
    df4['kind'] = 'source_g'
    df5['kind'] = 'target_g'
    df5a['kind'] = 'ztarget_labeled'

    df6['kind'] = 'source_trans'
    df7['kind'] = 'target_trans'
    df7a['kind'] = 'ztarget_labeled'

    df8['kind'] = 'source_pred'
    df9['kind'] = 'target_pred'
    df9a['kind'] = 'ztarget_labeled'

    df1['label_name'] = 'source'
    df2['label_name'] = 'target'
    df3['label_name'] = 'target_labeled'
    df4['label_name'] = 'source_g'
    df5['label_name'] = 'target_g'
    df5a['label_name'] = 'target_labeled'

    df6['label_name'] = 'source_trans'
    df7['label_name'] = 'target_trans'
    df7a['label_name'] = 'target_labeled'

    df8['label_name'] = 'source_pred'
    df9['label_name'] = 'target_pred'
    df9a['label_name'] = 'target_labeled'

    Y_labeled_target_plot = [y + 3 if y == 0 else y + 1 for y in Y_labeled_target]

    df1['label'] = Y_source_test
    if len(X_labeled_target) > 0:
        df2['label'] = np.concatenate([Y_target_test, Y_labeled_target])
    else:
        df2['label'] = Y_target_test
    df3['label'] = Y_labeled_target_plot
    df4['label'] = Y_source_test
    df5['label'] = Y_target_test

    df5a['label'] = Y_labeled_target_plot

    df6['label'] = Y_source_test
    df7['label'] = Y_target_test
    df7a['label'] = Y_labeled_target_plot

    df8['label'] = Y_source_test
    df9['label'] = Y_target_test
    df9a['label'] = Y_labeled_target_plot

    df1['color'] = "r"
    df2['color'] = "g"

    if len(X_labeled_target) > 0:
        df = pd.concat([df1, df2, df3], sort=False)
    else:
        df = pd.concat([df1, df2], sort=False)

    #### draw the boundary plot

    if len(X_labeled_target) > 0:
        xmin, xmax = get_min_max(X_source_test, X_target_test, X_labeled_target)
    else:
        xmin, xmax = get_min_max(X_source_test, X_target_test, X_target_test)

    #     g = sns.JointGrid(x='x1', y='x2', data=df, ylim={xmin[1],xmax[1]}, xlim={xmin[0],xmax[0]})
    g = sns.JointGrid(x='x1', y='x2', data=df)

    x_min, x_max = -4, 4
    y_min, y_max = -3.5, 3.5
    h = .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    levels = np.linspace(0, 1, 3)
    cm = plt.cm.coolwarm
    # cm = plt.cm.Paired
    Z = [None] * 2
    Z[0] = sess.run(S_prob, feed_dict={x_input: np.c_[xx.ravel(), yy.ravel()]})[:, 0].reshape(xx.shape)
    Z[1] = sess.run(S_prob, feed_dict={x_input: np.c_[xx.ravel(), yy.ravel()]})[:, 1].reshape(xx.shape)

    g.plot_joint(
        colored_contour(xx, yy, Z=1 - Z[0], levels=levels, cmap=cm, alpha=0.6),
    )

    g.plot_joint(
        colored_contour_line(xx, yy, Z=1 - Z[0], levels=levels, cmap=plt.cm.gist_gray, alpha=0.7),
    )

    plt.legend('')

    draw_plot_df(g, df, draw_marginals=False, draw_class_contours=False)

    #### draw the embedding visualization

    if len(X_labeled_target) > 0:
        xmin_g, xmax_g = get_min_max(X_source_g, X_target_g, X_target_labeled_g)
    else:
        xmin_g, xmax_g = get_min_max(X_source_g, X_target_g, X_target_g)

    if len(X_labeled_target) > 0:
        dfa = pd.concat([df4, df5, df5a], sort=False)
    else:
        dfa = pd.concat([df4, df5], sort=False)

    #     g2 = sns.JointGrid(x='x1', y='x2', data=dfa, ylim={xmin_g[1],xmax_g[1]}, xlim={xmin_g[0],xmax_g[0]})
    g2 = sns.JointGrid(x='x1', y='x2', data=dfa)

    draw_plot_df(g2, dfa, draw_marginals=True, draw_class_contours=True)

    if len(X_labeled_target) > 0:
        dfb = pd.concat([df1, df2, df3], sort=False)
    else:
        dfb = pd.concat([df1, df2], sort=False)

    #     g3 = sns.JointGrid(x='x1', y='x2', data=dfb, ylim={xmin[1],xmax[1]}, xlim={xmin[0],xmax[0]})
    g3 = sns.JointGrid(x='x1', y='x2', data=dfb)
    draw_plot_df(g3, dfb, draw_marginals=True, draw_class_contours=True)

    g.ax_joint.set_xticks([-3, 0, 3])
    g.ax_joint.set_yticks([-3, 0, 3])

    g2.ax_joint.set_xticks([int(xmin_g[0]), 0, int(xmax_g[0])])
    g2.ax_joint.set_yticks([int(xmin_g[1]), 0, int(xmax_g[1])])

    g3.ax_joint.set_xticks([-3, 0, 3])
    g3.ax_joint.set_yticks([-3, 0, 3])

    from matplotlib import rc
    rc('text', usetex=True)

    g.ax_joint.axes.set_ylabel('')
    g.ax_joint.axes.set_xlabel('')
    #     g.ax_joint.axes.set_title(r'\textbf{output space $h \circ g(X)$}', y=1.2, fontsize = 16)
    g.ax_joint.axes.set_title('output space $f \circ g(X)$', y=1.2, fontsize=14)

    g2.ax_joint.axes.set_ylabel('')
    g2.ax_joint.axes.set_xlabel('')
    #     g2.ax_joint.axes.set_title(r'\textbf{feature space $g(X)$}', y=1.2, fontsize = 16)
    g2.ax_joint.axes.set_title('feature space $g(X)$', y=1.2, fontsize=14)

    g3.ax_joint.axes.set_ylabel('')
    g3.ax_joint.axes.set_xlabel('')
    #     g3.ax_joint.axes.set_title(r'\textbf{input space $X$}', y=1.2, fontsize = 16)
    g3.ax_joint.axes.set_title('input space $X$', y=1.2, fontsize=14)

    #     g3.ax_joint.axes.set_title('input space $X$', y=1.2, fontsize = 14)
    #     g.ax_joint.axes.text(-0.2, -3.2, "Iter: " + str(), horizontalalignment='left', size='medium', color='black', weight='semibold')
    if iter_count is not None:
        #         g.ax_joint.axes.text(-3.5, 3.65, "Iter: " + str(iter_count) + '\t, Acc: ' + str(target_accuracy), horizontalalignment='left', size='medium', color='black', weight='semibold')
        g.ax_joint.axes.text(-3.5, 3.65, "Iter: %d, \t Acc: %0.3f" % (iter_count, target_accuracy),
                             horizontalalignment='left', size='medium', color='black', weight='semibold')

    rc('text', usetex=False)
    fig = plt.figure(figsize=(20, 6), dpi=110)
    gs = gridspec.GridSpec(1, 3, wspace=0.1)

    mg0 = SeabornFig2Grid(g3, fig, gs[0])
    mg1 = SeabornFig2Grid(g2, fig, gs[1])
    mg2 = SeabornFig2Grid(g, fig, gs[2])

    plt.savefig(fig_name + '.png', format='png', bbox_inches='tight', pad_inches=2)
    if save_gif_images:
        plt.close()


class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig, subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
                isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n, m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i, j], self.subgrid[i, j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h = self.sg.ax_joint.get_position().height
        h2 = self.sg.ax_marg_x.get_position().height
        r = int(np.round(h / h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r + 1, r + 1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        # https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure = self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())
