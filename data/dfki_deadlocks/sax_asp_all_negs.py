# Complete robot trajectory processing script
# Uses global z-normalization, correct binning of deadlock episodes, and ASP export

"""
 The script now generates negative sequences by:

    Scanning all deadlock-free segments of each trajectory.

    Slicing these segments into non-overlapping windows of size window_size.

    Ensuring these windows do not overlap with any positive intervals.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
from math import atan2
import os
import random
import string

# Parameters
window_size = 500
alphabet_size = 30
num_segments = 30
same_heading_threshold = 0.25
gap_threshold = 100

symbol_map = dict(enumerate(range(alphabet_size)))

"""
folder1 = Path("/media/nkatz/storage/EVENFLOW-DATA/DFKI/2_5_2025_deadlocks/output_robot1data")
folder2 = Path("/media/nkatz/storage/EVENFLOW-DATA/DFKI/2_5_2025_deadlocks/output_robot2data")
train_outfile = Path("/media/nkatz/storage/EVENFLOW-DATA/DFKI/2_5_2025_deadlocks/train.lp")
test_outfile = Path("/media/nkatz/storage/EVENFLOW-DATA/DFKI/2_5_2025_deadlocks/test.lp")
"""

folder1 = Path("output_robot1data")
folder2 = Path("output_robot2data")
train_outfile = Path("train.lp")
test_outfile = Path("test.lp")

numerical_features = ['px', 'py', 'vx', 'vy', 'ox', 'oy', 'oz', 'ow']
all_data, yaw_vals, dist_vals = [], [], []

for i in range(100):
    f1 = folder1 / f"out{i}.csv"
    f2 = folder2 / f"out{i}.csv"
    if f1.exists() and f2.exists():
        df1 = pd.read_csv(f1)
        df2 = pd.read_csv(f2)
        all_data.append(df1[numerical_features])
        all_data.append(df2[numerical_features])
        yaw_vals.append(np.arctan2(2 * (df1['ow'] * df1['oz'] + df1['ox'] * df1['oy']), 1 - 2 * (df1['oy']**2 + df1['oz']**2)))
        yaw_vals.append(np.arctan2(2 * (df2['ow'] * df2['oz'] + df2['ox'] * df2['oy']), 1 - 2 * (df2['oy']**2 + df2['oz']**2)))
        dist_vals.append(np.sqrt((df1['px'] - df2['px'])**2 + (df1['py'] - df2['py'])**2))

global_df = pd.concat(all_data)
global_means = global_df.mean()
global_stds = global_df.std().replace(0, 1)
all_yaws = pd.concat([pd.Series(y) for y in yaw_vals])
all_dists = pd.concat([pd.Series(d) for d in dist_vals])
global_yaw_mean = all_yaws.mean()
global_yaw_std = all_yaws.std() if all_yaws.std() > 0 else 1
global_dist_mean = all_dists.mean()
global_dist_std = all_dists.std() if all_dists.std() > 0 else 1

breakpoints = norm.ppf(np.linspace(1. / alphabet_size, 1 - 1. / alphabet_size, alphabet_size - 1))

print("Global means:")
print(global_means)
print("\nGlobal standard deviations:")
print(global_stds)
print(f"\nGlobal yaw mean: {global_yaw_mean:.4f}, std: {global_yaw_std:.4f}")
print(f"Global distance mean: {global_dist_mean:.4f}, std: {global_dist_std:.4f}")

print("\nSAX symbol breakpoints:")
for i in range(alphabet_size):
    low = -np.inf if i == 0 else breakpoints[i-1]
    high = np.inf if i == alphabet_size - 1 else breakpoints[i]
    print(f"Symbol {symbol_map[i]}: ({low:.4f}, {high:.4f}]")

def znormalize(ts, mean, std):
    return (ts - mean) / std if std > 0 else np.zeros_like(ts)

def sax(ts, mean, std):
    ts = znormalize(ts, mean, std)
    seg_len = len(ts) // num_segments
    paa = [ts[i*seg_len:(i+1)*seg_len].mean() for i in range(num_segments)]
    numeric_symbols = np.digitize(paa, breakpoints)
    return [symbol_map[s] for s in numeric_symbols]

def yaw(df):
    siny_cosp = 2 * (df['ow'] * df['oz'] + df['ox'] * df['oy'])
    cosy_cosp = 1 - 2 * (df['oy']**2 + df['oz']**2)
    return np.arctan2(siny_cosp, cosy_cosp)

def same_heading(yaw1, yaw2):
    return np.abs(np.unwrap(yaw1 - yaw2)) < same_heading_threshold

def bin_count(indices, gap_threshold=100):
    bins = {}
    current_bin = -1
    for idx in sorted(indices):
        if current_bin >= 0 and any((idx - i) in range(1, gap_threshold) for i in bins[current_bin]):
            bins[current_bin].append(idx)
        else:
            current_bin += 1
            bins[current_bin] = [idx]
    return bins

def extract_window_features(r1, r2, dist, same_head, start, label, seq_id):
    lines = []
    r1w, r2w = r1.iloc[start - window_size:start], r2.iloc[start - window_size:start]
    distw = dist[start - window_size:start]
    shw = same_head[start - window_size:start]
    goal1 = r1w['goal_status'].fillna("none").str.strip().str.lower().str.replace(" ", "_")
    goal2 = r2w['goal_status'].fillna("none").str.strip().str.lower().str.replace(" ", "_")

    features = {
        'px_1': sax(r1w['px'], global_means['px'], global_stds['px']),
        'py_1': sax(r1w['py'], global_means['py'], global_stds['py']),
        'vx_1': sax(r1w['vx'], global_means['vx'], global_stds['vx']),
        'vy_1': sax(r1w['vy'], global_means['vy'], global_stds['vy']),
        'yaw_1': sax(r1w['yaw'], global_yaw_mean, global_yaw_std),
        'px_2': sax(r2w['px'], global_means['px'], global_stds['px']),
        'py_2': sax(r2w['py'], global_means['py'], global_stds['py']),
        'vx_2': sax(r2w['vx'], global_means['vx'], global_stds['vx']),
        'vy_2': sax(r2w['vy'], global_means['vy'], global_stds['vy']),
        'yaw_2': sax(r2w['yaw'], global_yaw_mean, global_yaw_std),
        'dist':  sax(distw, global_dist_mean, global_dist_std),
    }

    for feat, seq in features.items():
        lines.append(" ".join([f"seq({seq_id},obs({feat},{s}),{t})." for t, s in enumerate(seq)]) + f" class({seq_id},{label}).")

    for t in range(num_segments):
        idx = t * (window_size // num_segments)
        g1 = goal1.iloc[idx]
        g2 = goal2.iloc[idx]
        sh = "true" if shw[idx] else "false"
        lines.append(f"seq({seq_id},obs(goal_status_1,{g1}),{t}). class({seq_id},{label}).")
        lines.append(f"seq({seq_id},obs(goal_status_2,{g2}),{t}). class({seq_id},{label}).")
        lines.append(f"seq({seq_id},obs(same_heading,{sh}),{t}). class({seq_id},{label}).")

    lines.append("")
    return lines

# Main processing loop
train_lines, test_lines = [], []
pos_count_train, pos_count_test, neg_count_train, neg_count_test = 0, 0, 0, 0
seq_id = 0

for i in range(100):
    f1 = folder1 / f"out{i}.csv"
    f2 = folder2 / f"out{i}.csv"
    if not f1.exists() or not f2.exists():
        continue

    r1 = pd.read_csv(f1)
    r2 = pd.read_csv(f2)
    r1['yaw'] = yaw(r1)
    r2['yaw'] = yaw(r2)
    dist = np.sqrt((r1['px'] - r2['px'])**2 + (r1['py'] - r2['py'])**2)
    sh = same_heading(r1['yaw'], r2['yaw'])

    dlock = ((r1['Deadlock_Bool'] == 1) | (r2['Deadlock_Bool'] == 1)).astype(int)
    dlock_indices = np.where(dlock == 1)[0]
    binned = bin_count(dlock_indices, gap_threshold)

    pos_windows = []
    episode_lines = []
    pos_count, neg_count = 0, 0
    for bin_id in binned:
        interval = binned[bin_id]
        end = interval[-1]
        start = end - window_size
        if start >= 0:
            print(f"Trajectory {i} POS start={start} end={end}")
            pos_windows.append((start, end))
            lines = extract_window_features(r1, r2, dist, sh, end, label=1, seq_id=seq_id)
            episode_lines.extend(lines)
            seq_id += 1
    pos_count = len(binned)

    # Generate negatives from deadlock-free regions
    dlock_free = (dlock == 0).astype(int)
    dlock_free_diff = np.diff(np.concatenate([[0], dlock_free, [0]]))
    starts = np.where(dlock_free_diff == 1)[0]
    ends = np.where(dlock_free_diff == -1)[0]

    for s, e in zip(starts, ends):
        for start in range(s + window_size, e + 1, window_size):
            if all((start <= ps or start - window_size >= pe) for ps, pe in pos_windows):
                print(f"Trajectory {i} NEG start={start - window_size} end={start}")
                lines = extract_window_features(r1, r2, dist, sh, start, label=0, seq_id=seq_id)
                episode_lines.extend(lines)
                seq_id += 1
                neg_count += 1

    print(f'Trajectory {i} pos: {pos_count} negs: {neg_count}')

    if i < 81:
        train_lines.extend(episode_lines)
        pos_count_train += pos_count
        neg_count_train += neg_count
    else:
        test_lines.extend(episode_lines)
        pos_count_test += pos_count
        neg_count_test += neg_count

train_outfile.write_text("\n".join(train_lines))
test_outfile.write_text("\n".join(test_lines))

print(f"\nTrain/Test pos/neg/lines: {pos_count_train}|{neg_count_train}|{len(train_lines)} {pos_count_test}|{neg_count_test}|{len(test_lines)}")
