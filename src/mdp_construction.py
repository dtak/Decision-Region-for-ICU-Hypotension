import numpy as np

def summarize_actions(curr_actions):
    '''
        Summarize the actions taken during a temporal clump into: 
        nothing, fluid at some point, vaso at some point, both fluid and vaso in the range [0, 3]. 
    '''

    some_fluid = np.sum(curr_actions % 4 > 0) > 0
    some_vaso = np.sum(curr_actions // 4 > 0) > 0
    
    return some_fluid + 2 * some_vaso

def compress(c_vec, a_vec):
    '''
        Takes in a full-length trajectory of labels (s1) and actions, and compresses to 
        eliminate repeating clusters or non-DPs. 
    '''
    assert(len(c_vec) == len(a_vec))
    
    sbar = []
    abar = []
    
    curr_cluster = -1
    curr_actions = []
    for i, c in enumerate(c_vec):
        if c >= 0: # is DP
            curr_actions.append(a_vec[i])
            if c != curr_cluster: # starting new cluster
                sbar.append(c)
            if i + 1 >= len(c_vec) or c_vec[i+1] != c: # ending this cluster
                abar.append(summarize_actions(np.array(curr_actions)))
                curr_actions = []
        curr_cluster = c
        
    return sbar, abar

def count_transitions(patient_labels1, patient_actions, patient_mortality, k): 
    '''
        Converts trajectories from original states to compressed space. 
        Calculates T(), TODO: R. 
        Input: labels of [s1] for each pid, actions for each pid, dict of mortality => 0, 1, 
        # clusters k.
    '''
    print('reloaded')
    
    # separate clusters into patient trajectories and compress
    compressed_trajectories = dict()
    for pid in patient_labels1.keys():
        c = patient_labels1[pid] # contains only s1
        a = patient_actions[pid] 
        compressed_trajectories[pid] = compress(c, a)
        
    # convert compressed trajectories to transition matrix
    T_mat = np.zeros((k + 2, 4, k + 2)) # leave space for mortality clusters
    for pid, (cvec, avec) in compressed_trajectories.items():
        if len(cvec) > 0:
            for i in range(len(cvec) - 1): # transitions within the trajectory
                T_mat[cvec[i], avec[i], cvec[i+1]] += 1
            # transition to mortality state from last clump 
            T_mat[cvec[-1], avec[-1], k + patient_mortality[pid]] += 1
        
    row_sums = T_mat.sum(axis=2, keepdims=True)
    return np.divide(T_mat, row_sums, out=np.zeros_like(T_mat), where=(row_sums > 0)), compressed_trajectories