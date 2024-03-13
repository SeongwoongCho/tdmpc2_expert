import os
import sys
import subprocess
nproc = int(sys.argv[1])
pid = int(sys.argv[2])
mode = str(sys.argv[3])

# 35 tasks
models = [
    # "acrobot-swingup",
    # "cartpole-balance-sparse",
    # "cartpole-balance", # HERE
    # "cheetah-jump",
    # "cheetah-run",
    # "cheetah-run-back",
    # "cheetah-run-backwards",
    # "cheetah-run-front",
    # "cartpole-swingup", # HERE
    # "cup-catch",
    # "cup-spin",
    "dog-run",
    "dog-stand",
    "dog-trot",
    # "cartpole-swingup-sparse", # HERE
    "dog-walk",
    # "finger-spin",
    # "finger-turn-easy",
    # "finger-turn-hard",
    # "walker-walk-backwards", # FUCK
    # "fish-swim",
    # "hopper-hop",
    # "hopper-hop-backwards",
    # "hopper-stand",
    "humanoid-run",
    "humanoid-stand",
    "humanoid-walk",
    # "pendulum-spin",
    # "pendulum-swingup",
    # "quadruped-run",
    # "quadruped-walk",
    # "reacher-easy",
    # "reacher-hard",
    # "reacher-three-easy",
    # "reacher-three-hard",
    # "walker-run",
    # "walker-run-backwards", 
    # "walker-stand",
    # "walker-walk",
]

# 2 tasks
models_mt30_19 = [
    "cartpole-balance",
    "cartpole-swingup-sparse"]

# 2 tasks
models_mt30_317 = [
    "cartpole-swingup",
    "walker-walk-backwards"
]

if mode == 'st':
    models = models
elif mode == 'mt30_19':
    models = models_mt30_19
elif mode == 'mt30_317':
    models = models_mt30_317

# seeds = [i for i in range(40)]
seeds = [i for i in range(11, 100)]
eval_episodes=100

cmds = []
for seed in seeds:
    for model in models:
        if model in ["cartpole-swingup", "walker-walk-backwards"]:
            cmds += [f"python save.py task=mt30 cur_task={model} checkpoint=/data1/seongwoongjo/tdmpc2/checkpoints/dmcontrol/mt30-317M.pt model_size=317 seed={seed} eval_episodes={eval_episodes}"]
        elif model in ["cartpole-balance", "cartpole-swingup-sparse"]:
            cmds += [f"python save.py task=mt30 cur_task={model} checkpoint=/data1/seongwoongjo/tdmpc2/checkpoints/dmcontrol/mt30-19M.pt model_size=19 seed={seed} eval_episodes={eval_episodes}"]
        else:
            cmds += [f"python save.py cur_task={model} task={model} checkpoint=/data1/seongwoongjo/tdmpc2/checkpoints/dmcontrol/{model}-1.pt seed={seed} eval_episodes={eval_episodes}"]

ith = pid
while ith < len(cmds):
    print(f"{ith+1} / {len(cmds)}")
    print(f"{cmds[ith]}")
    subprocess.call(cmds[ith].split(" "))
    ith += nproc
