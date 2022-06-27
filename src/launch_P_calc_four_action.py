import os
import numpy as np

slurm_template = """#!/bin/bash
#SBATCH -t 4-23:59
#SBATCH --mem=20000
#SBATCH -p doshi-velez
#SBATCH -o /dev/null
#SBATCH -e err
module load Anaconda3/5.0.1-fasrc01
source activate jzdu
python get_P.py {}
"""

def exec_slurm(args):
    slurm = slurm_template.format(args)
    with open("tmpfile.slurm", "w") as f:
        f.write(slurm)
    os.system("cat tmpfile.slurm | sbatch")
    os.system("rm tmpfile.slurm")

def launch_job(n, similarity, batch_size,
               candidate_file, partition_num,
               all_data_file, a_file, pid_file,
               kernel_type, kernel_file, style, file):
    args = "--num_neighbors {} --radius {} --batch_size {} " \
           "--candidate_file {} --partition {} " \
           "--all_data_file {} --a_file {} --pid_file {} " \
           "--kernel_type {} --kernel_file {} --style {} --file {}" \
            .format(n, similarity, batch_size,
                    candidate_file, partition_num,
                    all_data_file, a_file, pid_file,
                    kernel_type, kernel_file, style, file)
    # We calculate the decision points using computing clusters because
    # it's computationally heavy for local machines. Please replace details
    # based on your computing cluster or modify the scripts accordingly
    # to run on local machines. 
    exec_slurm(args)


for n in [15, 20]:
    for similarity in [0.9, 0.95]:
        for kernel_type in ['multiclass_withinds']:
            for style in ['noreapeat']:
                if kernel_type == 'multiclass_withinds':
                    kernel_path = '../result/model/multi_withindics_interpret_kernel_setseed888'
                    s1_train_path = '../result/intermediate/S1_train.npy'
                    s2_train_path = '../result/intermediate/S2_train.npy'
                    s1_test_path = '../result/intermediate/S1_test.npy'
                    s2_test_path = '../result/intermediate/S2_test.npy'
                a_path = '../result/intermediateA_train.npy'
                pid_path = '../result/intermediate/PID_train.npy'

                batch_size = 10000
                for file in ['s1_train', 's1_test', 's2_test']:
                    if file == 's1_train':
                        s1_train = np.load(s1_train_path)
                        s1_len = len(s1_train)
                        partition = s1_len // batch_size + 1
                        for part_num in range(partition):
                            launch_job(n, similarity, batch_size,
                                       s1_train_path, part_num,
                                       s1_train_path, a_path, pid_path,
                                       kernel_type, kernel_path, style, file)
                    elif file == 's1_test':
                        s1_test = np.load(s1_test_path)
                        s1_len = len(s1_test)
                        partition = s1_len // batch_size + 1
                        for part_num in range(partition):
                            launch_job(n, similarity, batch_size,
                                       s1_test_path, part_num,
                                       s1_train_path, a_path, pid_path,
                                       kernel_type, kernel_path, style, file)
                    elif file == 's2_train':
                        s2_train = np.load(s2_train_path)
                        s2_len = len(s2_train)
                        partition = s2_len // batch_size + 1
                        for part_num in range(partition):
                            launch_job(n, similarity, batch_size,
                                       s2_train_path, part_num,
                                       s1_train_path, a_path, pid_path,
                                       kernel_type, kernel_path, style, file)
                    elif file == 's2_test':
                        s2_test = np.load(s2_test_path)
                        s2_len = len(s2_test)
                        partition = s2_len // batch_size + 1
                        for part_num in range(partition):
                            launch_job(n, similarity, batch_size,
                                       s2_test_path, part_num,
                                       s1_train_path, a_path, pid_path,
                                       kernel_type, kernel_path, style, file)
