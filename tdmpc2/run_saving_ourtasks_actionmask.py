import os
import sys
import subprocess
nproc = int(sys.argv[1])
pid = int(sys.argv[2])

# 35 tasks
models = [
    "cartpole-two_poles",
    "cartpole-three_poles",
    "manipulator3d-reach_duplo_features",
    "manipulator3d-reach_site_features",
    "manipulator-bring_ball",
    "swimmer-swimmer6",
    "swimmer-swimmer15",
    "pointmass-easy"
]

seeds = [i for i in range(0, 11)]
eval_episodes=100

cmds = []
# for seed in seeds:
for seed in seeds:
    for model in models:
        cmds += [f"python save.py cur_task={model} task={model} checkpoint=/data3/seongwoongjo/tdmpc2_expert/TDMPC2_EXPERTS/{model}-1.pt seed={seed} eval_episodes={eval_episodes}"]

ith = pid
while ith < len(cmds):
    print(f"{ith+1} / {len(cmds)}")
    print(f"{cmds[ith]}")
    subprocess.call(cmds[ith].split(" "))
    ith += nproc
