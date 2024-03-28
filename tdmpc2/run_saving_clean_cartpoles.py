import os
import sys
import subprocess
nproc = int(sys.argv[1])
pid = int(sys.argv[2])

models = [
    "cartpole-two_poles",
    "cartpole-three_poles",
    "cartpole-swingup",
]

seeds = [i for i in range(11)]
eval_episodes=100

cmds = []
for seed in seeds:
    for model in models:
        if model in ["cartpole-swingup", "walker-walk-backwards"]:
            cmds += [f"python save_clean.py task=mt30 cur_task={model} checkpoint=/data3/seongwoongjo/tdmpc2_expert/checkpoints/dmcontrol/mt30-317M.pt model_size=317 seed={seed} eval_episodes={eval_episodes}"]
        else:
            cmds += [f"python save_clean.py cur_task={model} task={model} checkpoint=/data3/seongwoongjo/tdmpc2_expert/TDMPC2_EXPERTS/{model}-1.pt seed={seed} eval_episodes={eval_episodes}"]

ith = pid
while ith < len(cmds):
    print(f"{ith+1} / {len(cmds)}")
    print(f"{cmds[ith]}")
    subprocess.call(cmds[ith].split(" "))
    ith += nproc
