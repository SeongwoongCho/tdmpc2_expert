import os
import cv2
os.environ['MUJOCO_GL'] = 'egl'
import warnings
warnings.filterwarnings('ignore')

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2
from tqdm import tqdm
torch.backends.cudnn.benchmark = True

def get_raw_obs(env):
    try:
        raw_obs = env.unwrapped._env._env._env._env.task.get_observation(env.unwrapped._env._env._env._env.physics) # np.ndarray with shape (,) or (C, )
    except:
        raw_obs = env.unwrapped._env._env._env._env._observation_updater.get_observation()
    
    raw_obs = { k: torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v) for k, v in raw_obs.items()} 
    return raw_obs


def get_state(env):
    return torch.from_numpy(env.unwrapped._env._env._env._env._physics.get_state()) # np.ndarray


def get_acted_action(expert_action, rand_prob, noise_sigma):
    if np.random.uniform() < rand_prob or rand_prob == 1:
        acted_action = torch.rand_like(expert_action) * 2 - 1
    else:
        if noise_sigma == 0:
            acted_action = expert_action.clone()
        else:
            noise = torch.ones_like(expert_action)
            torch.nn.init.trunc_normal_(noise, mean=0, std=noise_sigma, a=-1, b=1)
            acted_action = (expert_action + noise).clamp(-1, 1)
    return acted_action


@hydra.main(config_name='config', config_path='.')
def save(cfg: dict):
    cfg = parse_cfg(cfg)
    rand_prob = cfg.rand_prob
    noise_sigma = cfg.noise_sigma
    data_root = cfg.data_root
    
    os.makedirs(f"{data_root}", exist_ok=True)
    os.makedirs(f"{data_root}/{cfg.cur_task.replace('cup', 'ball_in_cup').replace('_', '-')}", exist_ok=True)
    os.makedirs(f"{data_root}/{cfg.cur_task.replace('cup', 'ball_in_cup').replace('_', '-')}/{cfg.seed}", exist_ok=True)
    save_dir = f"{data_root}/{cfg.cur_task.replace('cup', 'ball_in_cup').replace('_', '-')}/{cfg.seed}"
    
    set_seed(cfg.seed)
    print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
    print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
    print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
    if not cfg.multitask and ('mt80' in cfg.checkpoint or 'mt30' in cfg.checkpoint):
        print(colored('Warning: single-task evaluation of multi-task models is not currently supported.', 'red', attrs=['bold']))
        print(colored('To evaluate a multi-task model, use task=mt80 or task=mt30.', 'red', attrs=['bold']))

    # Make environment
    env = make_env(cfg)

	# Load agent
    agent = TDMPC2(cfg)
    assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
    agent.load(cfg.checkpoint)
	# Evaluate
    if cfg.multitask:
        print(colored(f'Evaluating agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
    else:
        print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
	
    scores = []
    tasks = cfg.tasks if cfg.multitask else [cfg.task]
    for task_idx, task in enumerate(tasks):
        if not cfg.multitask:
            task_idx = None
        if cfg.cur_task != task:
            continue

        raw_obses_seed = {}
        obses_seed = []
        states_seed = []
        expert_actions_seed = []
        acted_actions_seed = []
        rewards_seed = []
        successes_seed = []         

        for i in tqdm(range(cfg.eval_episodes)):
            raw_obses_ep = {}
            obses_ep = []
            states_ep = []
            expert_actions_ep = []
            acted_actions_ep = []
            rewards_ep = [] 
            successes_ep = []

            obs, done, reward, t = env.reset(task_idx=task_idx), False, 0, 0 # (C,), False, 0, 0
            raw_obs = get_raw_obs(env)
            state = get_state(env) # np.ndarray
		    	
            for key in raw_obs:
                raw_obses_ep[key] = [raw_obs[key].cpu()]
            obses_ep.append(obs.cpu())
            states_ep.append(state.cpu())
            rewards_ep.append(reward)
            successes_ep.append(0.)

            while not done:
                expert_action = agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
                acted_action = get_acted_action(expert_action, rand_prob, noise_sigma)
                obs, reward, done, info = env.step(acted_action)
                
                t += 1
                raw_obs = get_raw_obs(env)
                state = get_state(env)
                
                for key in raw_obs:
                    raw_obses_ep[key].append(raw_obs[key].cpu())
                obses_ep.append(obs.cpu())
                states_ep.append(state.cpu())
                expert_actions_ep.append(expert_action.cpu())                
                acted_actions_ep.append(acted_action.cpu())                
                rewards_ep.append(reward.item())
                successes_ep.append(info['success'])
                
            for key in raw_obses_ep:
                if key not in raw_obses_seed:
                    raw_obses_seed[key] = []
                raw_obses_seed[key].append(torch.stack(raw_obses_ep[key]))
            obses_seed.append(torch.stack(obses_ep))
            states_seed.append(torch.stack(states_ep))
            expert_actions_seed.append(torch.stack(expert_actions_ep))
            acted_actions_seed.append(torch.stack(acted_actions_ep))
            rewards_seed.append(torch.tensor(rewards_ep))
            successes_seed.append(torch.tensor(successes_ep))

        for key in raw_obses_seed:
            raw_obses_seed[key] = torch.stack(raw_obses_seed[key])
        obses_seed = torch.stack(obses_seed)
        states_seed = torch.stack(states_seed)
        expert_actions_seed = torch.stack(expert_actions_seed)
        acted_actions_seed = torch.stack(acted_actions_seed)
        rewards_seed = torch.stack(rewards_seed)
        successes_seed = torch.stack(successes_seed)
        
        torch.save(raw_obses_seed, f"{save_dir}/rawobs_rp:{rand_prob}_ns:{noise_sigma}.pt")
        torch.save(obses_seed, f"{save_dir}/obs_rp:{rand_prob}_ns:{noise_sigma}.pt")
        torch.save(states_seed, f"{save_dir}/states_rp:{rand_prob}_ns:{noise_sigma}.pt")
        torch.save(expert_actions_seed, f"{save_dir}/expert_action_rp:{rand_prob}_ns:{noise_sigma}.pt")
        torch.save(acted_actions_seed, f"{save_dir}/acted_action_rp:{rand_prob}_ns:{noise_sigma}.pt")
        torch.save(rewards_seed, f"{save_dir}/rewards_rp:{rand_prob}_ns:{noise_sigma}.pt")        
        torch.save(successes_seed, f"{save_dir}/successes_rp:{rand_prob}_ns:{noise_sigma}.pt")        
        
        ep_rewards = rewards_seed.sum(-1).mean().item()
        ep_successes = successes_seed.mean().item()
        print(colored(f'  {task:<22}' \
                f'\tR: {ep_rewards:.01f}  ' \
                f'\tS: {ep_successes:.02f}', 'yellow'))


if __name__ == '__main__':
    import os
    save()
