import os
import sys
import subprocess

ALL_TASKS = [
    # >> 
    "walker-run",
    "walker-run-backwards", 
    "walker-stand",
    "walker-walk",
    "walker-walk-backwards",  
    # >> 
    "acrobot-swingup",
    # >> 
    "cartpole-balance", 
    "cartpole-swingup",
    "cartpole-two_poles",
    "cartpole-three_poles",
    # >>
    "cheetah-jump",
    "cheetah-run",
    "cheetah-run-back",
    "cheetah-run-backwards",
    "cheetah-run-front",
    # >> 
    "cup-catch",
    "cup-spin",
    # >>
    "dog-run",
    "dog-stand",
    "dog-trot",
    "dog-walk",
    # >> 
    "finger-spin",
    "finger-turn-easy",
    "finger-turn-hard",
    # >> 
    "fish-swim",
    # >>
    "hopper-hop",
    "hopper-hop-backwards",
    "hopper-stand",
    # >>
    "humanoid-run",
    "humanoid-stand",
    "humanoid-walk",
    # >> 
    "pendulum-spin",
    "pendulum-swingup",
    # >>
    "pointmass-easy"   
    # >> 
    "quadruped-run",
    "quadruped-walk",
    # >> 
    "reacher-easy",
    "reacher-hard",
    "reacher-three-easy",
    "reacher-three-hard",
    # >>
    "swimmer-swimmer6",
    "swimmer-swimmer15",
    # >>
    "manipulator-bring_ball",
    # >>
    "manipulator3d-reach_duplo_features",
    "manipulator3d-reach_site_features",
]

if __name__ == '__main__':
    import argparse
    import itertools
    import itertools
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed_min', type=int, default=0)
    parser.add_argument('--seed_max', type=int, default=10)
    parser.add_argument('--rand_prob', type=float, nargs='+') 
    parser.add_argument('--noise_sigma', type=float, nargs='+') 
    parser.add_argument('--save_root_dir', type=str)
    parser.add_argument('--ckpt_root_dir', type=str)
    parser.add_argument('--n_procs', type=int)
    parser.add_argument('--pid', type=int)
    parser.add_argument('--debug_mode', '-debug', action='store_true', default=False)
    parser.add_argument('--include', type=str, default='walker')
    args = parser.parse_args()
    
    if args.debug_mode:
        args.seed_min = 0
        args.seed_max = 0
        eval_episodes = 2
        args.rand_prob = [0.1]
        args.noise_sigma = [0.1]
        args.save_root_dir = args.save_root_dir + "_debug"
    else:
        eval_episodes = 100

    checkpoint_dict = {}
    for task in ALL_TASKS:
        if task in ["cartpole-swingup", "walker-walk-backwards"]:
            checkpoint_dict[task] = f"{args.ckpt_root_dir}/mt30-317M.pt"
        elif task in ["cartpole-balance", "cartpole-swingup-sparse"]:
            checkpoint_dict[task] = f"{args.ckpt_root_dir}/mt30-19M.pt"
        else:
            checkpoint_dict[task] = f"{args.ckpt_root_dir}/{task}-1.pt"
    seeds = [i for i in range(args.seed_min, args.seed_max + 1)] 
    cmds = []
    for task, seed, rand_prob, noise_sigma in itertools.product(ALL_TASKS, seeds, args.rand_prob, args.noise_sigma):
        if args.include is not None and args.include not in task:
            continue

        checkpoint = checkpoint_dict[task]
        if "mt30-19M.pt" in checkpoint:
            ptf = f"task=mt30 cur_task={task} model_size=19"
        elif "mt30-317M.pt" in checkpoint:
            ptf = f"task=mt30 cur_task={task} model_size=317"
        else:
            ptf = f"task={task} cur_task={task}"
        cmd = f"python save_latest.py checkpoint={checkpoint} seed={seed} eval_episodes={eval_episodes} data_root={args.save_root_dir} rand_prob={rand_prob} noise_sigma={noise_sigma} {ptf}"
        cmds += [cmd]

    ith = args.pid 
    while ith < len(cmds):
        print(f"{ith+1} / {len(cmds)}")
        print(f"{cmds[ith]}")
        subprocess.call(cmds[ith].split(" "))
        ith += args.n_procs 
