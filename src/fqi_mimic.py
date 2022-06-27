import time
import pickle
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from learn_kernel import train_kernel

class SafeBatchFittedQIteration(object):
    def __init__(self, A, r=0.95, n=10, regressor=None, model=None):
        """Initialize simulator and regressor. Can optionally pass a custom
        `regressor` model (which must implement `fit` and `predict` -- you can
        use this to try different models like linear regression or NNs)"""
        self.num_actions = len(np.unique(A))
        self.fitted = False
        self.r = r
        self.n = n
        self.model = model
        if regressor is None:
            self.regressor = ExtraTreesRegressor()
        else:
            self.regressor = regressor

    def Q(self, states):
        """Return the Q function estimate of `states` for each action"""
        if not self.fitted:
            # If not fitted, return 0s in the right shape
            return np.zeros((len(states), self.num_actions))
        else:
            return np.array([self.regressor.predict(self.encode(states, action))
                             for action in range(self.num_actions)]).T

    def fit_Q(self, S1, S2, A, R, P=None, num_iters=100, discount=0.98):
        """Fit and re-fit the Q function using historical data for the
        specified number of `iters` at the specified `discount` factor"""

        inputs = self.encode(S1, A)
        for _ in range(num_iters):
            start = time.time()
            curr_Q = self.Q(S2) if P is None else np.where(P == 1, self.Q(S2), -np.inf)
            # print(curr_Q)
            targets = R + discount * curr_Q.max(axis=1)
            self.regressor.fit(inputs, targets)
            end = time.time()
            #print("It takes {} seconds for one iteration of fit".format(end-start))
            self.fitted = True

    def encode(self, states, actions):
        """Encode states and actions as a single input array as onehot."""
        if isinstance(actions, int):
            actions = [actions] * len(states)
        onehot = np.zeros((len(actions), self.num_actions))
        onehot[np.arange(len(actions)), actions] = 1
        return np.hstack([states, onehot])

    def evaluate_final_policy(self, eval_states, all_states, A, r=None, n=None, P=None, mask=False):
        if not r: r = self.r
        if not n: n = self.n
        if not mask:
            return self.Q(eval_states)
        else:
            if not P:
                _, P = self.find_possible_actions(eval_states, all_states, A, r, n)
            curr_Q = np.where(P == 1, self.Q(eval_states), -np.inf)
            return curr_Q

    def singlecount_same_patient(self, batch_neighbor, onehot, indices):

        h = np.zeros((1, len(batch_neighbor)))
        for i in range(self.num_actions):
            a = onehot[:,i]
            b = np.multiply(batch_neighbor, a).T # shape = (s, 1000)
            c = np.cumsum(b, axis=0)
            pid_sum = np.vstack((c[indices[0]], c[indices[1:], :] - c[indices[:-1], :]))
            pid_sum = np.sum(pid_sum > 0, axis=0)
            h = np.vstack((h, pid_sum))

        h = h[1:]

        return h.T


    def find_possible_actions(self, cand_states, all_states, A, all_pid, r=None, n=None, batch_size=1000, model='None', kernel='None'):
        if r is None: r = self.r
        if n is None: n = self.n
        cand_states = self.model.trans_to_rff(cand_states)
        all_states = self.model.trans_to_rff(all_states)

        k, D = cand_states.shape
        all_states_T = np.transpose(all_states)

        # convert A to one-hot matrix
        onehot = np.zeros((len(A), self.num_actions))
        onehot[np.arange(len(A)), A] = 1

        # find pid change indices
        pid_indices = np.append(np.where(all_pid[:-1] != all_pid[1:])[0], len(all_pid) - 1)

        batch_num = k // batch_size + 1
        P = np.zeros((1, self.num_actions))

        for i in range(batch_num):
            begin = time.time()
            batch_cand = cand_states[i * batch_size:min((i + 1) * batch_size, k)]
            batch_dist = np.dot(batch_cand, all_states_T)
            batch_neighbor = batch_dist > r
            h = np.dot(batch_neighbor, onehot)
            # No double counting
            h = self.singlecount_same_patient(batch_neighbor, onehot, pid_indices)

            h = (h > 0).astype(int) if not self.n else (h >= n).astype(int)
            # If doesn't meet the r,n criteria, we allow the point to take nearest action
            nearest_indices = batch_dist.argmax(axis=1)
            no_action_indices = (h.sum(axis=1) == 0)
            h[no_action_indices] = onehot[nearest_indices][no_action_indices]
            P = np.vstack((P, h))
            with open('New_Results/progress_{}_{}.txt'.format(model, kernel),'a') as temp_file:
                line = 'total batch num {}, current iter num {}, seconds spent {}'.format(batch_num, i, time.time() - begin)
                temp_file.writelines(line)

        P = P[1:]
        dp_num = np.sum(P.sum(axis=1) > 1)

        return dp_num, P
