import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from sklearn.manifold import TSNE

viz_path = '../results/visualization/'

def draw_bar_plot(labels, included_clusters, save_name):
    '''
    Draw bar charts for decision points by labels

    Input
    labels : draw histogram of cluster label counts
    '''
    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=10)

    mapping = {
        x: y for x, y in zip(included_clusters, range(len(included_clusters)))
    }
    new_label = [mapping[x] for x in labels if x in included_clusters]
    K_num = len(np.unique(new_label))
    plt.figure(figsize=(10,4))
    plt.hist(new_label, bins=K_num, edgecolor='black')
    plt.xticks(np.arange(0.5, K_num + .5, 1 - 1 / K_num)[:K_num], range(K_num))
    plt.xlabel('Cluster number')
    plt.ylabel('Size')
    plt.savefig(os.path.join(viz_path, save_name), bbox_inches='tight')
    plt.show()

def draw_box_plot(feats_name, S_by_label, S_orig, included_clusters, save_name):
    '''
    Input
    feats_name : list of feature names
    S_by_label: list of S matrix by cluster labels
    S_orig: S matrix with features in original scale
    '''
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)
    mapping = {
        x: y for x, y in zip(included_clusters, range(len(included_clusters)))
    }
    K_num = len(S_by_label)
    feats_num = len(feats_name)
    fig, _ = plt.subplots(figsize=(50,50), sharex=True, sharey=True)
    fig.text(0.5, 0.05, 'Cluster', ha='center', fontsize=30)
    fig.text(0.07, 0.5, 'Value', va='center', rotation='vertical', fontsize=30)
    for i in range(len(feats_name)):
        plt.subplot(feats_num//2+1,2,i+1)
        plt.boxplot([S_by_label[c][:, i] for c in range(K_num) if c in included_clusters], positions=np.arange(len(included_clusters)), sym='+', autorange=True)
        plt.ylim(0, np.percentile(S_orig[:, i], 95))
        plt.axhline(S_orig[:, i].mean())
        plt.title(feats_name[i], fontsize=30)
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(os.path.join(viz_path, save_name), bbox_inches='tight')
    plt.show()

def draw_colormap(column_names, S_orig_by_label, included_clusters, save_name, width, height):
    '''
    Input
    column_names : list of columns to be plotted
    S_orig_by_label : S in original state space grouped by clusters
    save_name : name for the fig to be saved
    K_CONST : number of clusters
    '''
    mapping = {
        x: y for x, y in zip(included_clusters, range(len(included_clusters)))
    }
    col_len = len(column_names)
    K_CONST = len(included_clusters)
    mean_mat = np.zeros((col_len, K_CONST))
    normailized_mean_mat = np.zeros((col_len, K_CONST))
    for d in range(col_len):
        for c in included_clusters:
            mean_mat[d, mapping[c]] = np.mean(S_orig_by_label[c][:, d])
        normailized_mean_mat[d] = (mean_mat[d] - mean_mat[d].min()) / (mean_mat[d].max() - mean_mat[d].min())

    plt.figure(figsize=(height,width))
    im = plt.matshow(normailized_mean_mat, cmap='GnBu', fignum=1)
    for d in range(col_len):
        for c in included_clusters:
            plt.text(mapping[c], d, '{:0.0f}'.format(mean_mat[d, mapping[c]]), ha='center', va='center', fontsize=10)
    cbar = plt.colorbar(im,fraction=0.0239, pad=0.02)
    cbar.ax.set_ylabel('Relative (normalized) value', rotation=270, fontsize=15, labelpad=40)
    plt.xticks(range(K_CONST), range(0, K_CONST), fontsize=15)
    plt.yticks(range(col_len), column_names, fontsize=15)
    plt.tick_params(axis="x", labelbottom=True, labeltop=False)
    plt.title('Mean Feature Value For Each Decision Region', fontsize=20)
    plt.xlabel('Decision Regions', fontsize=18)
    plt.grid(False)
    plt.savefig(os.path.join(viz_path, save_name), bbox_inches='tight')
    plt.show()

def draw_tsne(Z, S_labels_ordered, save_name, PERPLEX=120, sample_size=200, random_state=888):
    '''
    Input
    Z : features transformed into Z space
    S_labels_ordered : S labeled by cluster number in original order
    save_name : name for the fig to be saved
    perplexity : perplexity to call TSNE
    sample_size : number of samples to collect from each cluster
    '''
    np.random.seed(random_state)

    K_CONST = len(np.unique(S_labels_ordered))
    sample_before_tsne = []

    for i in range(K_CONST):
        sample_idx = np.random.choice(np.where(S_labels_ordered==i)[0], sample_size)
        Z_label_sample = Z[sample_idx]
        sample_before_tsne.append(Z_label_sample)
    sample_before_tsne = np.vstack([sample for sample in sample_before_tsne])

    # TSNE cluster
    sample_after_tsne = TSNE(n_components=2,
                             perplexity=PERPLEX,
                             random_state=random_state).fit_transform(sample_before_tsne)

    # plot
    xs, ys, positions  = {}, {}, {}
    for i in range(K_CONST):
        xs[i] = sample_after_tsne[i*sample_size:(i+1)*sample_size, 0]
        ys[i] = sample_after_tsne[i*sample_size:(i+1)*sample_size, 1]
        positions[i] = np.array([xs[i].mean(), ys[i].mean()])

    fig = plt.figure(figsize=(30,22))
    ax = plt.gca()
    ax.axis('equal')
    cm = plt.get_cmap('tab20')
    ax.set_prop_cycle(color=[cm(1.*i/K_CONST) for i in range(K_CONST)])
    for i in range(K_CONST):
        ax.scatter(xs[i], ys[i], s=30, alpha=0.5, label=i+1)
        ax.text(*positions[i], i+1, fontsize=40)
    plt.legend(markerscale=3, prop={'size': 30})
    plt.savefig(os.path.join(viz_path, save_name))
    plt.show()

def draw_action_tsne(Z, S_labels_ordered, A, save_name, perplex=120, sample_size=200, random_state=888, cmap='tab20'):
    '''
    Input
    Z : features transformed into Z space
    A : actions (pre-summarized)
    S_labels_ordered : S labeled by cluster number in original order
    save_name : name for the fig to be saved
    perplexity : perplexity to call TSNE
    sample_size : number of samples to collect from each cluster
    '''
    np.random.seed(random_state)

    K_CONST = len(np.unique(S_labels_ordered))
    sample_before_tsne = []
    sample_actions = []

    for i in range(K_CONST):
        sample_idx = np.random.choice(np.where(S_labels_ordered==i)[0], sample_size)
        Z_label_sample = Z[sample_idx]
        sample_before_tsne.append(Z_label_sample)
        sample_actions.append(A[sample_idx])
    sample_before_tsne = np.vstack([sample for sample in sample_before_tsne])
    sample_actions = np.hstack([sample for sample in sample_actions])

    # TSNE cluster
    sample_after_tsne = TSNE(n_components=2,
                             perplexity=perplex,
                             random_state=random_state).fit_transform(sample_before_tsne)

    fig = plt.figure(figsize=(30,22))
    ax = plt.gca()
    ax.axis('equal')
    cm = plt.get_cmap(cmap)
    num_a = np.max(A) + 1
    action_colors = cm(sample_actions / num_a)[:, :3]

    plt.scatter(sample_after_tsne[:, 0], sample_after_tsne[:, 1], s=30, alpha=0.5, c=action_colors)
    plt.legend(handles=[Line2D([0], [0], marker='o', color='w', label=i,
                          markerfacecolor=cm(i/num_a)) for i in range(num_a)],
               markerscale=3, prop={'size': 30})
    plt.savefig(os.path.join(viz_path, save_name))
    plt.show()

def draw_tiles(T, save_path, vis_thresh=0.5):
    '''
    Draw the transition probability from source cluster to destination cluster after taking given actions.

    Arguments
    ---------------
    T: transition matrix. shape = [k, 4, k] where k is number of clusters. 4 is number of actions
    '''
    cmap = {
        0: "blue",
        1: "orange",
        2: "green",
        3: "red"
    }

    sides = ["up", "down"]

    custom_lines = [Line2D([0], [0], color='blue', lw=4),
                Line2D([0], [0], color='orange', lw=4),
                Line2D([0], [0], color='green', lw=4),
                Line2D([0], [0], color='red', lw=4)]

    rect_w, rect_h = .1, .1

    def draw_rectangle(ax, name, position, w=rect_w, h=rect_h):
        # Draw a single rectangle at speicified position
        x, y = position
        rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='r', facecolor='blue', alpha=0.7)
        cx, cy = x + w/2, y + h/2
        ax.add_artist(rect)
        ax.annotate(name, (cx, cy), color='w', weight='bold',
                    fontsize=14, ha='center', va='center')

    def draw_arror(ax, start_pos, end_pos, action, proba, camp):
        side = 'up' if action%2 == 0 else 'down'
        order = action // 2
        style = "Simple,tail_width=0.5,head_width=8,head_length=8"
        if side == 'up':
            cstyle = f"arc3,rad={-.2 - .15 * action}"
            start_pos = (start_pos[0], start_pos[1] + rect_h/2)
            end_pos = (end_pos[0], end_pos[1] + rect_h/2)
            txt_pos = (end_pos[0], end_pos[1] + .05 * (1 + order)) #avoid text overlap
        else:
            cstyle = f"arc3,rad={.2 + .15 * action}"
            start_pos = (start_pos[0], start_pos[1] - rect_h/2)
            end_pos = (end_pos[0], end_pos[1] - rect_h/2)
            txt_pos = (end_pos[0], end_pos[1] - .05 * (1 + order))
        kwargs = dict(arrowstyle=style, linewidth=proba*10, color=cmap[action], alpha=0.6)
        arrow = patches.FancyArrowPatch(start_pos, end_pos, connectionstyle=cstyle, **kwargs)
        ax.add_artist(arrow)
        ax.annotate(f'{round(proba,2)}', (txt_pos[0], txt_pos[1]), color='black', weight='bold',
                    fontsize=12, ha='center', va='center')

    s, a, s_prime = T.shape
    # Iterate through every starting state
    for i in range(s-2):
        fig = plt.figure(figsize=(15,5))
        ax = fig.add_subplot(111)
        heavy_trans_cluster = []
        cluster_position = {}
        for j in range(a):
            dest_group_idx = np.where(T[i, j, :] > vis_thresh)[0]
            dest_group_proba = T[i, j, :][dest_group_idx]
            for m, n in zip(dest_group_idx, dest_group_proba):
                heavy_trans_cluster.append([j, m, n]) # Action, destination, proba
        all_dest_group = list(set([k[1] for k in heavy_trans_cluster]))

        # Draw source rectangle on the left, and destination rectangles on the right
        draw_rectangle(ax, f'cluster {i}', (0.05, 0.5))
        start_pos = (0.05 + rect_w/2, 0.5 + rect_h/2)
        for j, c in enumerate(all_dest_group):
            if c == 15: name = 'Alive'
            elif c == 16: name = 'Death'
            else: name = f'cluster {c}'
            draw_rectangle(ax, name, (0.2 + j*0.15, 0.5))
            cluster_position[name] = (0.2 + j*0.15 + rect_w/2, 0.5 + rect_h/2)

        # Draw lines to connect source with destination
        for action, dest, proba in heavy_trans_cluster:
            if dest == 15:name = 'Alive'
            elif dest == 16: name = 'Death'
            else: name = f'cluster {dest}'
            end_pos = cluster_position[name]
            draw_arror(ax, start_pos, end_pos, action, proba, cmap)

        ax.legend(custom_lines, ['No Treatment', 'Fluid Only', 'Vaso Only', 'Fluid&Vaso'])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #fig.axes.append(ax)
        plt.savefig(save_path + f'Transition After Action - Cluster {i}', bbox_inches='tight')
        plt.show()
