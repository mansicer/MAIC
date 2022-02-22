from numpy.lib.ufunclike import isposinf
from smac.env.multiagentenv import MultiAgentEnv
import numpy as np
import gym
from gym.envs.registration import register
from sys import stderr

import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

class ForagingEnv(MultiAgentEnv):

    def __init__(self,
                 field_size: int,
                 players: int,
                 max_food: int,
                 force_coop: bool,
                 partially_observe: bool,
                 sight: int,
                 is_print: bool,
                 seed: int, 
                 need_render: bool,
                 render_output_path: str = ''):
        self.n_agents = players
        self.n_actions = 6
        self._total_steps = 0
        self._episode_steps = 0
        self.is_print = is_print
        self.need_render = need_render
        np.random.seed(seed)

        self.episode_limit = 50

        self.agent_score = np.zeros(players)

        env_id = "Foraging{4}-{0}x{0}-{1}p-{2}f{3}-v1".format(field_size, players, max_food,
                                                              "-coop" if force_coop else "",
                                                              "-{}s".format(sight) if partially_observe else "")
        if is_print:
            print('Env:', env_id, file=stderr)
        self.env = gym.make(env_id)
        self.env.seed(seed)

        if self.need_render:
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            render_path = os.path.join(render_output_path, date)
            if not os.path.exists(render_path):
                os.makedirs(render_path, exist_ok=True)
            self.render_path = render_path

    def step(self, actions):
        """ Returns reward, terminated, info """
        self._total_steps += 1
        self._episode_steps += 1

        if self.is_print:
            print(f'env step {self._episode_steps}', file=stderr)
            print('t_steps: %d' % self._episode_steps, file=stderr)
            print('current position: ', file=stderr)
            print(self.env.unwrapped.get_players_position(), file=stderr)
            print('choose actions: ', file=stderr)
            print(actions.cpu().numpy().tolist(), file=stderr)
            position_record = self.env.unwrapped.get_players_position()
            action_record = actions.cpu().numpy().tolist()
            env_info = {
                'position': position_record,
                'action': action_record
            }
            import pickle
            pickle.dump(env_info, open(os.path.join(self.render_path, f'info_step_{self._episode_steps}.pkl'), 'wb'))

        if self.need_render:
            fig = plt.figure()
            data = self.env.render(mode='rgb_array')
            plt.imshow(data)
            plt.axis('off')
            fig.savefig(os.path.join(self.render_path, f'image_step_{self._total_steps}.png'), bbox_inches='tight', dpi=600)
        
        self.obs, rewards, dones, info = self.env.step(actions.cpu().numpy())

        self.agent_score += rewards

        reward = np.sum(rewards)
        terminated = np.all(dones)

        return reward, terminated, {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self.obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return np.array(self.obs[agent_id])

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.env._get_observation_space().shape[0]

    def get_state(self):
        state = self.obs[0]
        for i in range(self.n_agents - 1):
            state = np.concatenate([state, self.obs[i + 1]])
        return state

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.get_obs_size() * self.n_agents

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        res = [0] * self.n_actions
        t = self.env._valid_actions[self.env.players[agent_id]]
        for i in range(len(t)):
            res[t[i].value] = 1
        return res

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.n_actions

    def reset(self):
        """ Returns initial observations and states"""
        self._episode_steps = 0
        self.agent_score = np.zeros(self.n_agents)
        self.obs = self.env.reset()
        return self.get_obs(), self.get_state()

    def render(self, mode='human'):
        self.env.render(mode)

    def close(self):
        self.env.close()

    def seed(self):
        pass

    def save_replay(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        stats = {
            "agent_score": self.agent_score,
        }
        return stats
