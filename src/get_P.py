from fqi_mimic import *
from Model.learn_kernel import *
import argparse


def singlecount_same_patient(batch_neighbor, onehot, indices, num_actions):
    h = np.zeros((1, len(batch_neighbor)))
    for i in range(num_actions):
        a = onehot[:, i]
        b = np.multiply(batch_neighbor, a).T  # shape = (s, 1000)
        c = np.cumsum(b, axis=0)
        pid_sum = np.vstack((c[indices[0]], c[indices[1:], :] - c[indices[:-1], :]))
        pid_sum = np.sum(pid_sum > 0, axis=0)
        h = np.vstack((h, pid_sum))

    h = h[1:]

    return h.T

def find_possible_actions(r, n, kernel, cand_states, all_states, A, all_pid, style):

    cand_states = kernel.trans_to_rff(cand_states)
    all_states = kernel.trans_to_rff(all_states)

    all_states_T = np.transpose(all_states)

    k, D = cand_states.shape

    # convert A to one-hot matrix
    num_actions = len(np.unique(A))
    onehot = np.zeros((len(A), num_actions))
    onehot[np.arange(len(A)), A] = 1

    # find pid change indices
    pid_indices = np.append(np.where(all_pid[:-1] != all_pid[1:])[0], len(all_pid) - 1)

    batch_num = k // 500 + 1
    P = np.zeros((1, num_actions))
    for i in range(batch_num):
        batch_cand = cand_states[i * 500: (i+1) * 500]
        batch_dist = np.dot(batch_cand, all_states_T)
        batch_neighbor = batch_dist > r
        if style == 'doublecounting':
            h = np.dot(batch_neighbor, onehot)
        else:
            # No double counting
            h = singlecount_same_patient(batch_neighbor, onehot, pid_indices, num_actions)

        h = (h >= n).astype(int)
        # If doesn't meet the r,n criteria, we allow the point to take nearest action
        nearest_indices = batch_dist.argmax(axis=1)
        no_action_indices = (h.sum(axis=1) == 0)
        h[no_action_indices] = onehot[nearest_indices][no_action_indices]
        P = np.vstack((P, h))

    P = P[1:]
    return P

def calculate_P(args):
    print('Start running')
    cand_state = np.load(args.candidate_file)
    partition = args.partition
    batch_size = args.batch_size
    cand_state = cand_state[partition*batch_size:(partition+1)*batch_size]
    all_state = np.load(args.all_data_file)
    A = np.load(args.a_file)
    PID = np.load(args.pid_file)
    if args.kernel_type == 'multiclass_withinds':
        kernel = LearningKernel(20,4,500)
        kernel.load_state_dict(torch.load(args.kernel_file))
    else:
        kernel = torch.load(args.kernel_file)

    P = find_possible_actions(args.radius, args.num_neighbors,
                              kernel, cand_state, all_state,
                              A, PID, args.style)

    p_path = 'P_Results/P_r={}_n={}_{}_{}_{}_{}'.format(args.radius,
                                                     args.num_neighbors,
                                                     args.kernel_type,
                                                     args.style,
                                                     args.file,
                                                     partition)

    pickle.dump(P, open(p_path, 'wb'))

with open('P_Results/progress.txt', 'a') as f:
    f.write("Start Running get P python")
parser = argparse.ArgumentParser('Calculating P')

parser.add_argument('--num_neighbors', type=int, default=1, help="minimum number of neighbors for permitted action.")
parser.add_argument('--radius', type=float, default=0.5, help="similarity threshold for two states in kernel space.")
parser.add_argument('--batch_size',  type=int, default=10000, help='batch size for neighbor identification')
parser.add_argument('--candidate_file',  type=str, help='candidate data file path')
parser.add_argument('--partition', type=int, help='counter of partition number')
parser.add_argument('--all_data_file', type=str, help='all data file path')
parser.add_argument('--a_file',  type=str, help='action data file path')
parser.add_argument('--pid_file', type=str, help='patient ID data file path')
parser.add_argument('--kernel_type', type=str, help='type of kernels')
parser.add_argument('--kernel_file', type=str, help='file that stores kernel info')
parser.add_argument('--style', type=str, help='whether allow doublecounting or norepeat')
parser.add_argument('--file', type=str, help='Whether it is for s1 train, s1 test or s2 test')

args = parser.parse_args()
calculate_P(args)
