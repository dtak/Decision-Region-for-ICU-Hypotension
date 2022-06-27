import numpy as np
import math
import pickle
from collections import Counter
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.kernel_approximation import RBFSampler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

###############################################
# Data Processing Functions
def get_train_test_action_pred(sample_id, states, actions, rewards, original=True, transform=True,
                               include_action_states=True):
    '''
    Functions that give the train test split of data for action prediction.

    Input:
    states: Dictionary of patients condition dataframe
    actions: Dictionary of patients treatment dataframe
    binary: Whether to convert actions to binary - receive treatment or not
    original: Whether to convert actions to original set of features


    Return:
    Train and test feature, label, patientid, admission hour
    '''

    def create_data(pid_list):
        '''Create dataset based on pid input'''
        orig_pid, hr, R = [], [], []
        S1, S2, Y = [], [], []
        #S1_mod, S2_mod = [], []
        for pid in pid_list:
            orig_pid = orig_pid + [pid] * (len(states[pid]) - 1)
            hr.append(states[pid].values[:-1, 0])
            if include_action_states:
                S1.append(np.hstack([states[pid].values[:-1], actions[pid].values[:-1, 1:7]]))
                S2.append(np.hstack([states[pid].values[1:], actions[pid].values[1:, 1:7]]))
            else:
                S1.append(states[pid].values[:-1])
                S2.append(states[pid].values[1:])
            #S1_mod.append(states_mod[p].values[:-1, :])
            #S2_mod.append(states_mod[p].values[1:, :])
            Y.append(actions[pid].values[1:, -3])
            R.append(rewards[pid].values[:, 1])
        return np.concatenate(S1, axis=0), np.concatenate(S2, axis=0), \
                np.concatenate(Y, axis=0), np.concatenate(R, axis=0), \
                np.concatenate(hr, axis=0), np.array(orig_pid)

    pid_train, pid_test = train_test_split(sample_id, random_state=11)
    S1_train, S2_train, y_train, R_train, hr_train, orig_pid_train = create_data(pid_train)
    S1_test, S2_test, y_test, R_test, hr_test, orig_pid_test = create_data(pid_test)

    # Find indices where map is below 65
    S1_train_map_below65_indices = np.where(S1_train[:, 55] < 65)[0]
    S1_test_map_below65_indices = np.where(S1_test[:, 55] < 65)[0]

    if transform:
        s1_scaler = StandardScaler()
        s2_scaler = StandardScaler()
        S1_train = s1_scaler.fit_transform(S1_train)
        S1_test = s1_scaler.transform(S1_test)
        S2_train = s2_scaler.fit_transform(S2_train)
        S2_test = s2_scaler.transform(S2_test)

    if not original:
        y_train = convert_actions(y_train)
        y_test = convert_actions(y_test)

    return S1_train, S1_test, S2_train, S2_test, \
           y_train, y_test, R_train, R_test, \
           orig_pid_train, orig_pid_test, hr_train, hr_test, \
           S1_train_map_below65_indices, S1_test_map_below65_indices

def convert_actions(y):
    '''Bin actions to 4 classes'''
    some_fluid = y % 4 > 0
    some_vaso = y // 4 > 0

    return some_fluid + 2 * some_vaso

def get_class_weight(y_train):
    '''
    return loss of weight related to each class
    here we assume all classes appear in data...
    '''
    num_of_classes = len(np.unique(y_train))
    class_count = np.bincount(y_train)
    loss_weight = np.zeros(num_of_classes)
    total = len(y_train)
    for y, count in zip(np.unique(y_train), class_count):
        loss_weight[y] = total / count
    return loss_weight

def feature_selection(X_train, y_train, clf, max_feat_num):
    '''
    Return the indices of features that are important using L1 regularization

    Input:
    X_train: Features in original forms
    y_train: Labels
    clf: type of classifier being used to select features

    Ouptut:
    Indices of important features
    '''
    # clf.fit(X_train, y_train)
    # Sort feature importance
    thresholds = sorted(clf.feature_importances_[clf.feature_importances_>0])[::-1]
    # Select top important features
    feat_num = min(max_feat_num, len(thresholds))
    threshold = thresholds[feat_num]
    # get indices of those features
    feat_indices = np.where(clf.feature_importances_ > threshold)[0]
    # feat_indices = np.where(clf.coef_[0] > 0)
    return feat_indices

def trans_x_to_z(X, kernel_weight, model):
    '''Transform X features to Z space given optimized kernel weights, omega and b'''
    omega, b, D = model.omega.numpy(), model.b.numpy(), model.D
    X = kernel_weight * X
    Z = np.sqrt(2/D)*np.cos((np.dot(X, omega) + b))
    return Z

def multiclass_roc_auc_score(y_test, y_pred, average="weighted"):
    '''Calculate the AUC score for multiclass classifier'''
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

def evaluate_clf(X_train, X_test, y_train, y_test, clf):
    print("Training AUC is {0}".format(multiclass_roc_auc_score(y_train, clf.predict(X_train), 'weighted')))
    print("Testing AUC is {0}".format(multiclass_roc_auc_score(y_test, clf.predict(X_test),'weighted')))
    print("Training Accuracy is {0}".format(accuracy_score(y_train, clf.predict(X_train))))
    print("Testing Accuracy is {0}".format(accuracy_score(y_test, clf.predict(X_test))))


# Functions for selecting decision point on MIMIC according to entropy
def cross_entropy(p, eps=1e-9):
    return -np.dot(p, np.log(p + eps))

def entropy(labels, eps=1e-9):
    if len(labels) < 1: return 0
    p = np.bincount(labels) / len(labels)
    if len(p) <= 1: return 0
    return -np.dot(p, np.log(p + eps))

def find_neighbors_mimic(zs, idx, y_actual, pid, hr, delta):
    '''Find neighbors of specified point of a patient'''
    distance = np.dot(zs, zs[idx].reshape(-1,1))
    #distance = np.dot(zs, zs[idx].reshape(-1,1))
    distance[distance < 0] = 0.
    distance[distance > 1] = 1.
    distance = distance.reshape(-1,1)
    # points within delta
    candidates = np.hstack((distance, y_actual.reshape(-1,1), pid.reshape(-1,1), hr.reshape(-1,1)))
    #candidates = np.vstack((distance, y_actual, pid, hr)).T
    indices = np.where(candidates[:,0] >= delta)[0]
    candidates = candidates[indices]
    candidates = candidates[candidates[:,0].argsort()][::-1]
    # exclude self and multiple neighbor points from same patient
    self_pid = pid[idx]
    candidates = candidates[candidates[:,2]!=self_pid]
    candidate_pid = np.unique(candidates[:,2])
    neighbor_num = len(candidate_pid)
    neighbor = {}
    for row in candidates:
        if len(neighbor) == neighbor_num:
            break
        if row[2] not in neighbor:
            neighbor[row[2]] = row[1]
    return list(neighbor.values())

def select_points_w_high_uncertainty_mimic(ps, quantile):
    return np.argsort(np.array([cross_entropy(p) for p in ps]))[-int(quantile * len(ps)):]

def select_uncertain_points_mimic(zs, ps, y, pid, hr, delta, quantile=0.1):
    return [(idx, y[idx], ps[idx], pid[idx], hr[idx], find_neighbors_mimic(zs, idx, y, pid, hr, delta)) \
            for idx in select_points_w_high_uncertainty_mimic(ps, quantile)]

def select_decision_points_mimic(uncertain_points, neighbors_threshold, entropy_threshold):
    decision_points = []
    for point in uncertain_points:
        if len(point[5]) >= neighbors_threshold and entropy(point[5]) > entropy_threshold:
            decision_points.append(point)
    return decision_points

def select_decision_points(cand_states, all_states, A, r, n=None, batch_size = 1000):
    '''Find decision points among candidate states.

    Input:
    cand_states: array of states we are interested in identifying decision points among. Provided in RFF space.
    all_states: array of all states in training dataset. Provided in RFF space.
    A: corresponding actions for all states.
    r: threshold of distance (radius of ball) within which we identify neighbors
    n: optional. Determine the minimum number of neighbors who take an action to make it allowable.

    Return:
    Number of decision points in cand_states and Available actions for candidate states
    '''
    k, D = cand_states.shape
    all_states_T = np.transpose(all_states)

    # convert A to one-hot matrix
    onehot = np.zeros((len(A), max(A) + 1))
    onehot[np.arange(len(A)), A] = 1

    batch_num = k // batch_size + 1
    P = np.zeros((1, max(A) + 1))

    for i in range(batch_num):
        batch_cand = cand_states[i * batch_size:min((i + 1) * batch_size, k)]
        batch_dist = np.dot(batch_cand, all_states_T)
        batch_neighbor = batch_dist > r
        h = np.dot(batch_neighbor, onehot)
        if not n:
            h = (h > 0).astype(int)
        else:
            h = (h >= n).astype(int)
        P = np.vstack((P, h))

    P = P[1:]
    dp_num = np.sum(P.sum(axis=1) > 1)

    return dp_num, P

#########################################################
## Set of functions to evaluate output
def calc_dp_num(P):
    return np.sum(P.sum(axis=1) > 1)

def calc_avg_action(P):
    return P.sum() / len(P)

def calc_Q(model, P, eval_states, mask=False):
    if not mask:
        Q = model.Q(eval_states)
    else:
        Q = np.where(P == 1, model.Q(eval_states), -np.inf)
    return Q

def calc_ope(model, P, eval_states, mask=False):
    Q = calc_Q(model, P, eval_states, mask=False)
    return np.mean(Q.max(axis=1))

def calc_action_diff(fqi, model, P, eval_states, mask=False):
    Q = calc_Q(model, P, eval_states, mask=False)
    return np.sum(Q.argmax(axis=1) != fqi.Q(eval_states).argmax(axis=1))

def calc_aggresive_diff(fqi, model, P, eval_states, mask=False):
    fqi_Q = fqi.Q(S1_train_subset).argmax(axis=1)
    modle_Q = calc_Q(model, P, eval_states, mask=False)
    diff_indices = model_Q.argmax(axis=1) != fqi_Q
    diff_action_model = model_Q.argmax(axis=1)[diff_indices]
    diff_action_fqi = fqi_Q[diff_indices]
    diff = np.sign(diff_action_fqi - diff_action_model)
    num_fqi_aggre = np.sum(diff > 0)
    num_model_aggre = np.sum(diff < 0)
    return num_fqi_aggre, num_model_aggre
