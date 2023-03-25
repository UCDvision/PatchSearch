import sys
import re
import glob
import numpy as np

data_seed_exps = []
means = []
print('='*(15*6))
print(('{:>15s}'*6).format('Data Seed', 'Run Seed', 'Clean Acc', 'Clean FP', 'Poisoned Acc', 'Poisoned FP'))
print('='*(15*6))

def print_exps(data_seed_exps, means):
    data_seed_exps = np.array(data_seed_exps)
    mean_clean_acc = data_seed_exps[:, 1].mean()
    mean_clean_fp = data_seed_exps[:, 2].mean()
    mean_poisoned_acc = data_seed_exps[:, 3].mean()
    mean_poisoned_fp = data_seed_exps[:, 4].mean()
    std_clean_acc = data_seed_exps[:, 1].std()
    std_clean_fp = data_seed_exps[:, 2].std()
    std_poisoned_acc = data_seed_exps[:, 3].std()
    std_poisoned_fp = data_seed_exps[:, 4].std()
    print('-'*(15*6))
    dseed = int(data_seed_exps[-1][0])
    print(f'{dseed:14d} {"MEAN":>14s} {mean_clean_acc:14.1f} {mean_clean_fp:14.1f} {mean_poisoned_acc:14.1f} {mean_poisoned_fp:14.1f}')
    print(f'{dseed:14d} {"STD":>14s} {std_clean_acc:14.1f} {std_clean_fp:14.1f} {std_poisoned_acc:14.1f} {std_poisoned_fp:14.1f}')
    print('='*(15*6))
    means.append((dseed, mean_clean_acc, mean_clean_fp, mean_poisoned_acc, mean_poisoned_fp))

for log_file in sorted(glob.glob(f'{sys.argv[1]}/eval_data_seed_*_run_seed_*/logs')):
    data_seed, run_seed = re.search(r'eval_data_seed_([0-9]+)_run_seed_([0-9]+)', log_file).groups()
    data_seed, run_seed = int(data_seed), int(run_seed)
    if data_seed != run_seed or data_seed < 4787 or data_seed > 4791:
        continue
    with open(log_file, 'r') as f:
        lines = f.readlines()
    clean_acc, poisoned_acc, clean_fp, poisoned_fp = None, None, None, None
    for line in lines[-7:]:
        if line.startswith('Clean'):
            clean_acc = -1
        elif clean_acc == -1:
            clean_acc = float(line.split()[0])
            clean_fp = int(line.split()[2])
        elif line.startswith('Poisoned'):
            poisoned_acc = -1
        elif poisoned_acc == -1:
            poisoned_acc = float(line.split()[0])
            poisoned_fp = int(line.split()[2])
    if data_seed_exps and data_seed_exps[-1][0] != data_seed:
        print_exps(data_seed_exps, means)
        data_seed_exps = []
    if clean_acc and poisoned_acc:
        data_seed_exps.append((data_seed, clean_acc, clean_fp, poisoned_acc, poisoned_fp))
        print(f'{data_seed:14d} {run_seed:14d} {clean_acc:14.1f} {clean_fp:14d} {poisoned_acc:14.1f} {poisoned_fp:14d}')

print_exps(data_seed_exps, means)
means[-1] = (0, *means[-1][1:])
print_exps(means, [])

