import time

from collections import defaultdict
import numpy as np
import pandas as pd
import pickle
import pylab
import seaborn as sns
import torch
import networkx as nx

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import matplotlib.patheffects
from matplotlib.lines import Line2D
import matplotlib.patches

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


from learn_kernel import LearningKernel
from pipeline_mimic import feature_selection, get_train_test_action_pred

np.random.seed(888)
torch.manual_seed(888)




def load_data(s_path, a_path, r_path,
              feat_indices, clinician_feats,
              original=True, include_action_states=True, transform=False):
    '''Create Train/teste split data for S1, S2, A, R'''

    # load files
    states = pickle.load(open(s_path, 'rb'))
    actions = pickle.load(open(a_path, 'rb'))
    rewards = pickle.load(open(r_path, 'rb'))

    # prepare train/test data
    pid = list(states.keys())
    S1_train, S1_test, \
    S2_train, S2_test, \
    y_train, y_test, \
    R_train, R_test, \
    orig_pid_train, orig_pid_test, \
    hr_train, hr_test, \
    S1_train_map_below65_indices, S1_test_map_below65_indices = get_train_test_action_pred(pid, states, actions, rewards,
                                      original=original,
                                      include_action_states=include_action_states,
                                      transform=transform)

    # select non-indicator features
    # columns = states[pid[0]].columns[2:].append(actions[pid[0]].columns[3:7])
    columns = states[pid[0]].columns.append(actions[pid[0]].columns[1:7])
    column_names = columns[feat_indices]
    feat_indices_noind = [i for i in feat_indices if 'ind' not in columns[i]]

    # Add MAP if not already added
    map_idx = list(columns).index('map')
    if map_idx not in feat_indices_noind:
        feat_indices_noind = [map_idx] + feat_indices_noind

    # switch map to the first position of feature
    map_pos = feat_indices_noind.index(map_idx)
    if map_pos != 0:
        feat_indices_noind[map_pos], feat_indices_noind[0] = feat_indices_noind[0], feat_indices_noind[map_pos]
    column_names_noind = columns[feat_indices_noind]

    # manually add clinician requred features if not selected by model
    # just for visualization
    # Additional feature: add MAP to beginning
    columns_mod = states[pid[0]].columns
    vis_feat_indices = []
    for feat, idx in clinician_feats.items():
        if feat not in column_names_noind:
            vis_feat_indices.append(idx)
    column_names_vis = columns_mod[vis_feat_indices]

    # output standardized S1_train but non_standardized S1_no_ind
    s1_scaler = StandardScaler()
    s2_scaler = StandardScaler()
    S1_train_norm = s1_scaler.fit_transform(S1_train)
    S1_test_norm = s1_scaler.transform(S1_test)
    S2_train_norm = s2_scaler.fit_transform(S2_train)
    S2_test_norm = s2_scaler.transform(S2_test)

    # output
    S1_train_kernel = S1_train_norm[:, feat_indices]
    S2_train_kernel = S2_train_norm[:, feat_indices]
    S1_train_viz = np.hstack([S1_train[:, feat_indices_noind], S1_train[:, vis_feat_indices]])
    S2_train_viz = np.hstack([S2_train[:, feat_indices_noind], S2_train[:, vis_feat_indices]])
    A_train = y_train
    PID_train = orig_pid_train

    S1_test_kernel = S1_test_norm[:, feat_indices]
    S2_test_kernel = S2_test_norm[:, feat_indices]
    S1_test_viz = np.hstack([S1_test[:, feat_indices_noind], S1_test[:, vis_feat_indices]])
    S2_test_viz = np.hstack([S2_test[:, feat_indices_noind], S2_test[:, vis_feat_indices]])
    A_test = y_test
    PID_test = orig_pid_test

    return S1_train_kernel, S2_train_kernel, S1_train_viz, S2_train_viz, \
            A_train, PID_train, column_names, list(column_names_noind) + list(column_names_vis), \
            S1_test_kernel, S2_test_kernel, S1_test_viz, S2_test_viz, A_test, PID_test, \
            S1_train_map_below65_indices, S1_test_map_below65_indices

def load_kernel(kernel_path, input_dim, output_dim, rand_feat_num, interpret=True):
    kernel_dict = torch.load(kernel_path)
    kernel = LearningKernel(input_dim, output_dim, rand_feat_num, interpret=interpret)
    kernel.load_state_dict(kernel_dict)
    return kernel

def find_dp(P):
    '''Find indices corresponding to decision points'''
    #P = pickle.load(open(P_path, 'rb'))
    dp_indicators = np.sum(P, axis=1) > 1
    dp_idx = np.where(dp_indicators)[0]
    return dp_idx

def select_K(Z, samplesize, repeat, nc_ls):
    '''Select most reasonable number as K for clustering'''
    inertia = np.zeros((repeat, len(nc_ls)))
    silhoutte = np.zeros((repeat, len(nc_ls)))

    for i in range(repeat):
        sample_dp_idx = np.random.choice(range(len(Z)), samplesize, replace=False)
        sample_Z = Z[sample_dp_idx]
        for j, nc in enumerate(nc_ls):
            #print(f'Trying cluster number {nc}')
            start = time.time()
            kmeans = KMeans(n_clusters=nc, init='k-means++')
            centers = kmeans.fit_predict(sample_Z)
            inertia[i, j] = kmeans.inertia_
            silhoutte[i, j] = silhouette_score(sample_Z, centers)
            end = time.time()
            #print(f'Time: {round(end - start, 2)}')

    plt.plot(nc_ls, inertia.mean(axis=0))
    plt.title('Reconstruction error as a function of k')
    plt.ylabel('Error')
    plt.xlabel('k')
    plt.show()
    plt.plot(nc_ls, silhoutte.mean(axis=0))
    plt.title('Silhouette score as a function of k')
    plt.ylabel('Score')
    plt.xlabel('k')
    plt.show()

def cluster(Z, samplesize, k_size, seed=888):
    np.random.seed(seed)
    sample_idx = np.random.choice(range(len(Z)), samplesize, replace=False)
    Z_sample = Z[sample_idx]
    kmeans = KMeans(n_clusters=k_size, init='k-means++', random_state=888)
    kmeans.fit(Z_sample)
    Z_labels = kmeans.predict(Z)
    return Z_labels

def sort_by_feat(feat_idx, novaso_labels, vaso_labels,
                 S1_novaso_orig, S1_vaso_orig,
                 vaso_8hrs_N, vaso_8hrs_Y):
    '''
    Sort clusters by feature values and assign new cluster numbers

    Input -
    feat_idx : index of the feature to be sorted
    novaso_labels : cluster labels for DP with no vaso in past 8 hrs
    vaso_labels : cluster labels for DP given vaso in past 8 hrs
    S1_novaso_orig : S1 matrix of original features with no vaso in past 8 hrs
    S1_vaso_orig : S1 matrix of original features given vaso in past 8 hrs
    vaso_8hrs_N : indices of DP with no vaso in past 8 hrs
    vaso_8hrs_Y : indices of DP given vaso in past 8 hrs


    Return -
    S1_orig_by_label : DPs by cluster
    S1_labels : cluster label ordered by feature values assigned back to S1_orig in original order
    '''
    # increment vaso label by number of non-vaso labels
    novaso_K, vaso_K = len(np.unique(novaso_labels)), len(np.unique(vaso_labels))
    vaso_labels = vaso_labels + novaso_K
    total_K = novaso_K + vaso_K

    # concatenate non-vaso labels with vaso labels
    S1_orig_new = np.vstack((S1_novaso_orig, S1_vaso_orig))
    label_new = np.concatenate((novaso_labels, vaso_labels))
    S1_by_label = [S1_orig_new[label_new==c] for c in range(total_K)]

    # Sort by a feature value
    idx_map = sorted(range(total_K),
                    key=lambda x: S1_by_label[x][:, feat_idx].mean(),
                    reverse=True)

    # Reassign clusters and cluster Labels
    inverted_idx_map = {}
    for i, j in enumerate(idx_map):
        inverted_idx_map[j] = i
    S1_orig_by_label = [S1_by_label[i] for i in idx_map]

    # Map labels back to original S1 orders
    S1_labels_ordered = np.zeros_like(label_new)

    for idx, c in zip(vaso_8hrs_N, novaso_labels):
        S1_labels_ordered[idx] = inverted_idx_map[c]

    for idx, c in zip(vaso_8hrs_Y, vaso_labels):
        S1_labels_ordered[idx] = inverted_idx_map[c]

    return S1_orig_by_label, S1_labels_ordered, inverted_idx_map
