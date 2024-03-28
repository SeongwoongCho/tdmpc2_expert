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

data_root="/data3/seongwoongjo/DATA_DMC_CLEAN"

@hydra.main(config_name='config', config_path='.')
def save(cfg: dict):
    cfg = parse_cfg(cfg)
    # data_root = cfg.data_root
    os.makedirs(f"{data_root}", exist_ok=True)
    os.makedirs(f"{data_root}/{cfg.cur_task}", exist_ok=True)
    os.makedirs(f"{data_root}/{cfg.cur_task}/{cfg.seed}", exist_ok=True)
    save_dir = f"{data_root}/{cfg.cur_task}/{cfg.seed}"
    
    set_seed(cfg.seed)
    print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
    print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
    print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
    if not cfg.multitask and ('mt80' in cfg.checkpoint or 'mt30' in cfg.checkpoint):
        print(colored('Warning: single-task evaluation of multi-task models is not currently supported.', 'red', attrs=['bold']))
        print(colored('To evaluate a multi-task model, use task=mt80 or task=mt30.', 'red', attrs=['bold']))

    cfg.save_video = False
    save_image = True 
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
    if cfg.save_video:
        video_dir = os.path.join(cfg.work_dir, 'videos')
        os.makedirs(video_dir, exist_ok=True)
	
    scores = []
    tasks = cfg.tasks if cfg.multitask else [cfg.task]
    for task_idx, task in enumerate(tasks):
        if not cfg.multitask:
            task_idx = None
        if cfg.cur_task != task:
            continue

        raw_obses_seed = {}
        obses_seed = []
        ep_actions_seed = []
        states_seed = []
        ep_rewards_seed = []
        for i in tqdm(range(cfg.eval_episodes)):
            raw_obses, obses, ep_actions, ep_actions_mask, ep_rewards, ep_successes, states = {}, [], [], [], [], [], []
            obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0 # (C,), False, 0, 0
            try:
                raw_obs = env.unwrapped._env._env._env._env.task.get_observation(env.unwrapped._env._env._env._env.physics) # np.ndarray with shape (,) or (C, )
            except:
                raw_obs = env.unwrapped._env._env._env._env._observation_updater.get_observation()
            state = env.unwrapped._env._env._env._env._physics.get_state() # np.ndarray
		    	
            # raw_obses.append(raw_obs)
            for key in raw_obs:
                raw_obses[key] = [raw_obs[key]]

            obses.append(obs)
            ep_rewards.append(ep_reward)
            ep_successes.append(0.)
            states.append(state)

            if save_image:
                frame = env.render(width=84, height=84)
                cv2.imwrite(f'{save_dir}/demo_{i}-frame_{t}.jpg', frame)

            while not done:
                action = agent.act(obs, t0=t==0, task=task_idx)
                obs, reward, done, info = env.step(action)
                
                t += 1
                try:
                    raw_obs = env.unwrapped._env._env._env._env.task.get_observation(env.unwrapped._env._env._env._env.physics) # np.ndarray with shape (,) or (C, )
                except:
                    raw_obs = env.unwrapped._env._env._env._env._observation_updater.get_observation()               

                state = env.unwrapped._env._env._env._env._physics.get_state() # np.ndarray
                ep_reward += reward.item()
                if save_image:
                    frame = env.render(width=84, height=84)
                    cv2.imwrite(f'{save_dir}/demo_{i}-frame_{t}.jpg', frame) 
                    
                for key in raw_obs:
                    raw_obses[key].append(raw_obs[key])

                obses.append(obs)
                ep_actions.append(action)
                ep_rewards.append(ep_reward)
                ep_successes.append(info['success'])
                states.append(state) 
	       
            for key in raw_obses:
                if key not in raw_obses_seed:
                    raw_obses_seed[key] = []
                raw_obses_seed[key].append(torch.from_numpy(np.stack(raw_obses[key])))
            obses_seed.append(torch.stack(obses))
            states_seed.append(torch.from_numpy(np.stack(states)))
            ep_actions_seed.append(torch.stack(ep_actions))
            ep_rewards_seed.append(torch.tensor(ep_rewards))

            # torch.save((raw_obses, obses, ep_actions, ep_rewards, ep_successes), f'{save_dir}/demo_{i}-obs.pt')
            ep_rewards = np.mean(ep_rewards)
            ep_successes = np.mean(ep_successes) 
         
        for key in raw_obses_seed:
            raw_obses_seed[key] = torch.stack(raw_obses_seed[key])
        obses_seed = torch.stack(obses_seed)
        states_seed = torch.stack(states_seed)
        ep_actions_seed = torch.stack(ep_actions_seed)
        ep_rewards_seed = torch.stack(ep_rewards_seed)

        torch.save(raw_obses_seed, f'{save_dir}/rawobs.pt')
        torch.save(states_seed, f'{save_dir}/states.pt')
        torch.save(obses_seed, f'{save_dir}/obs.pt')
        torch.save(ep_actions_seed, f'{save_dir}/action.pt')
        torch.save(ep_rewards_seed, f'{save_dir}/rewards.pt')
        

        if cfg.multitask:
            scores.append(ep_successes*100 if task.startswith('mw-') else ep_rewards/10)
        print(colored(f'  {task:<22}' \
                f'\tR: {ep_rewards:.01f}  ' \
                f'\tS: {ep_successes:.02f}', 'yellow'))
    if cfg.multitask:
        print(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))


if __name__ == '__main__':
    import os
    save()
