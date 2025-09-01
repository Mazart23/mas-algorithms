from __future__ import annotations

import numpy as np

from ...utils.custom_objects.enums import AgentType
from ...utils import global_parameters as gp


class SoftmaxAdjuster:
    def __init__(self, supervisor: 'Supervisor'):
        self.supervisor = supervisor
        self.agent_types_lst: list[AgentType] = [agent for agent in AgentType]
        
        self.tau = gp.ADAPTATION_NUM_AGENT_TAU

        self.min_adjust = gp.ADAPTATION_NUM_AGENT_MIN_CHANGE
        self.max_adjust = gp.ADAPTATION_NUM_AGENT_MAX_CHANGE
        
        self.perfomance_avg_percentage = gp.ADAPTATION_NUM_AGENT_AVG_INFLUENCE
        self.perfomance_best_percentage = gp.ADAPTATION_NUM_AGENT_BEST_INFLUENCE

    def softmax(self, x):
        x = np.array(x) / self.tau
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def step(self):
        performances = {
            agent_type: (
                self.perfomance_avg_percentage * self.supervisor.avg_perfomance_history[-1][agent_type] +
                self.perfomance_best_percentage * self.supervisor.best_perfomance_history[-1][agent_type]
            )
            for agent_type in AgentType
        }

        perf_vals = np.array(list(performances.values()))
        inv_perf = 1.0 / (perf_vals + 1e-9)
        weights = self.softmax(inv_perf)
        
        total_agents = sum(self.supervisor.num_agents.values())
        target_counts = {
            a: int(round(w * total_agents))
            for a, w in zip(self.agent_types_lst, weights)
        }

        deltas = {
            a: target_counts[a] - self.supervisor.num_agents[a]
            for a in self.agent_types_lst
        }

        givers = {a: -d for a, d in deltas.items() if d < 0}
        takers = {a: d for a, d in deltas.items() if d > 0}

        total_give = sum(givers.values())
        total_take = sum(takers.values())
        movable = min(total_give, total_take)

        if movable == 0:
            return

        move_total = int(np.clip(movable, self.min_adjust, self.max_adjust))

        give_plan = {
            a: int(round(v / total_give * move_total))
            for a, v in givers.items()
        }

        adjusted = 0
        for a, n in list(give_plan.items()):
            max_can_give = self.supervisor.num_agents[a] - 1
            if n > max_can_give:
                adjusted += n - max_can_give
                give_plan[a] = max_can_give

        move_total = sum(give_plan.values())

        if total_take > 0:
            take_plan = {
                a: int(round(v / total_take * move_total))
                for a, v in takers.items()
            }
        else:
            take_plan = {}

        diff = move_total - sum(take_plan.values())
        if diff != 0 and take_plan:
            a = max(take_plan, key=take_plan.get)
            take_plan[a] += diff

        for a, n in give_plan.items():
            if n > 0:
                self.supervisor.remove_agents(a, n)
                self.supervisor.num_agents[a] -= n

        for a, n in take_plan.items():
            if n > 0:
                self.supervisor.add_agents(a, n)
                self.supervisor.num_agents[a] += n
