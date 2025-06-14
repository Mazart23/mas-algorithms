from __future__ import annotations

import numpy as np

from ...utils.custom_objects.enums import AgentType
from ...utils import global_parameters as gp


class SoftmaxAdjuster:
    def __init__(self, supervisor: 'Supervisor'):
        self.supervisor = supervisor
        self.agent_types_lst: list[AgentType] = [agent for agent in AgentType]
        
        self.tau = gp.ADAPTATION_NUM_AGENT_TAU
        self.alpha = gp.ADAPTATION_NUM_AGENT_ALPHA
        self.gamma = gp.ADAPTATION_NUM_AGENT_GAMMA

        self.min_adjust = gp.ADAPTATION_NUM_AGENT_MIN_CHANGE
        self.max_adjust = gp.ADAPTATION_NUM_AGENT_MAX_CHANGE
        self.dynamic_speed = gp.ADAPTATION_NUM_AGENT_SPEED
        
        self.perfomance_avg_percentage = gp.ADAPTATION_NUM_AGENT_AVG_INFLUENCE
        self.perfomance_best_percentage = gp.ADAPTATION_NUM_AGENT_BEST_INFLUENCE

        self.n_actions = self.supervisor.len_agent_types * (self.supervisor.len_agent_types - 1)
        self.Q = np.zeros(self.n_actions)

        self.prev_best_value = None
        self.prev_adjust_num = (self.min_adjust + self.max_adjust) // 2

    def softmax(self, logits):
        scaled_logits = logits / self.tau
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        return exp_logits / np.sum(exp_logits)

    def select_action(self):
        probs = self.softmax(self.Q)
        action = np.random.choice(np.arange(self.n_actions), p=probs)
        return action

    def update_q(self, action, reward):
        self.Q[action] = self.Q[action] + self.alpha * (reward + self.gamma * np.max(self.Q) - self.Q[action])

    def update_adjust_num(self, improvement):
        if improvement is None:
            return

        improvement_ratio = improvement / self.prev_best_value

        new_adjust_num = self.max_adjust - improvement_ratio * self.dynamic_speed * (self.max_adjust - self.min_adjust)
        new_adjust_num = np.clip(new_adjust_num, self.min_adjust, self.max_adjust)

        self.prev_adjust_num = int(new_adjust_num)

    def step(self):
        performances = {
            agent_type: (
                self.perfomance_avg_percentage * self.supervisor.avg_perfomance_history[-1][agent_type] + \
                self.perfomance_best_percentage * self.supervisor.best_perfomance_history[-1][agent_type]
            )
            for agent_type in AgentType
        }
        
        action = self.select_action()

        from_idx = action // (self.supervisor.len_agent_types - 1)
        to_idx = action % (self.supervisor.len_agent_types - 1)
        
        if to_idx >= from_idx:
            to_idx += 1

        from_agent = self.agent_types_lst[from_idx]
        to_agent = self.agent_types_lst[to_idx]

        max_movable = min(self.prev_adjust_num, self.supervisor.num_agents[from_agent] - 1)
        
        if max_movable <= 0:
            return

        self.supervisor.remove_agents(from_agent, max_movable)
        self.supervisor.add_agents(to_agent, max_movable)
        self.supervisor.num_agents[from_agent] -= max_movable
        self.supervisor.num_agents[to_agent] += max_movable

        if self.prev_best_value is None:
            reward = 0
            improvement = None
        else:
            delta = self.prev_best_value - self.supervisor.global_best.value
            reward = delta

            if performances[to_agent] < performances[from_agent]:
                reward += 0.1 * abs(performances[to_agent] - performances[from_agent])

            improvement = delta

        self.update_adjust_num(improvement)
        self.prev_best_value = self.supervisor.global_best.value
        self.update_q(action, reward)
