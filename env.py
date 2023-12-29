import numpy as np
import itertools
from gym import spaces
from pettingzoo import AECEnv, ParallelEnv

class CustomEnvironment(AECEnv, ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self):
        # 代理名称列表
        self.agents = ["car_0", "car_1", "car_2", "car_3"]
        # 重排序地点的数量
        self.num_locations = 5
        # 所有可能的队列排序
        self.possible_forms = list(itertools.permutations(self.agents))
        # 每个智能体的动作空间和观测空间
        self.action_spaces = {agent: spaces.Discrete(len(self.possible_forms)) for agent in self.agents}
        self.observation_spaces = {agent: spaces.Box(low=0, high=1, shape=(len(self.agents) + 1,), dtype=np.float32) for
                                   agent in self.agents}
        # 电量消耗矩阵
        self.Delta = np.array([[0.1302, 0.1334, 0.2522, 0.1868, 0.0787],
                               [0.1224, 0.1255, 0.2372, 0.1756, 0.0741],
                               [0.1170, 0.1199, 0.2266, 0.1678, 0.0708],
                               [0.1170, 0.1199, 0.2266, 0.1678, 0.0708]])
        self.current_location = 0
        self.cumulative_rewards = {agent: 0 for agent in self.agents}
        initial_soc = 1.0
        self.state = {agent: np.array([initial_soc] * (len(self.agents) + 1)) for agent in self.agents}
        self.state[-1] = self.num_locations
        self.reset()

    def reset(self):
        # 在这里初始化环境的初始状态
        # 将所有智能体的SOC设置为1
        initial_soc = 1.0
        self.state = {agent: np.array([initial_soc] * (len(self.agents) + 1)) for agent in self.agents}
        # 设置初始的重排序地点
        self.state[-1] = self.num_locations
        self.current_location = 0
        self.cumulative_rewards = {agent: 0 for agent in self.agents}

    @property
    def num_agents(self):
        return len(self.agents)

    def observe(self, agent):
        # 返回指定智能体的观测
        return self.state[agent]

    def legal_moves(self, agent):
        # 返回指定智能体的合法动作列表
        return list(range(len(self.possible_forms)))

    def step(self, action):
        # 执行动作并更新环境状态
        new_states = np.zeros((self.num_agents, self.num_agents))
        rewards = {agent: 0 for agent in self.agents}

        # 计算每个智能体根据其选择的动作产生的新状态
        for agent in self.agents:
            form = self.possible_forms[action[agent]]
            new_soc = self.state[agent] - self.Delta[self.agents.index(agent), self.current_location]
            new_states[self.agents.index(agent)] = new_soc

        # 计算每个智能体的奖励
        for agent in self.agents:
            # 个人SOC保持大的奖励
            individual_reward = new_states[self.agents.index(agent), self.agents.index(agent)]
            # 整体SOC标准差小的奖励
            std_dev_reward = -np.std(new_states[self.agents.index(agent)])
            rewards[agent] = individual_reward + std_dev_reward

        # 选择奖励最高的动作来更新状态
        best_action = max(rewards, key=lambda k: rewards[k])
        best_form = self.possible_forms[action[best_action]]

        # 更新SOC值为选择的最优排列对应的SOC
        for agent in self.agents:
            self.state[agent] = new_states[self.agents.index(best_action)][best_form.index(agent)]

        self.state[-1] -= 1
        self.cumulative_rewards[best_action] += rewards[best_action]

        # 判断是否到达最后一个重排序点
        done = self.state[-1] == 0
        if done:
            # 计算最终的奖励
            std_dev = np.std([self.state[agent] for agent in self.agents])
            final_reward = -std_dev  # 标准差越小，惩罚越小，所以取负值

            # 计算队列变化的相似度惩罚
            similarity_penalty = self.calculate_similarity_penalty(self.previous_form, best_form)

            # 计算队列变化次数的惩罚
            num_changes = self.calculate_num_changes(best_form)
            change_penalty = -num_changes

            # 计算电量向量与Delta最后一列相除后的最小值的惩罚
            efficiency_penalty = self.calculate_efficiency_penalty([self.state[agent] for agent in self.agents], self.Delta[:, -1])

            # 将所有惩罚和标准差奖励相加
            for agent in self.agents:
                self.cumulative_rewards[agent] += final_reward + similarity_penalty + change_penalty + efficiency_penalty

        infos = {}
        return self.state, self.cumulative_rewards, done, infos

    def calculate_similarity_penalty(self, prev_form, new_form):
        # 计算队列变化的相似度惩罚
        # 相似度计算为两个队列不同位置的数量
        dissimilarity = sum(p != n for p, n in zip(prev_form, new_form))
        # 惩罚与不同位置的数量成正比
        penalty = -dissimilarity
        return penalty

    def calculate_num_changes(self, form):
        # 计算队列变化次数
        changes = sum(1 for i in range(len(form) - 1) if form[i] != form[i + 1])
        return changes

    def calculate_efficiency_penalty(self, soc_vector, delta_vector):
        # 计算效率惩罚
        efficiency_vector = soc_vector / delta_vector
        penalty = -min(efficiency_vector)
        return penalty

    def render(self):
        # 渲染环境（如果需要）
        pass

    def observation_space(self, agent):
        # 返回指定智能体的观测空间
        return self.observation_spaces[agent]

    def action_space(self, agent):
        # 返回指定智能体的动作空间
        return self.action_spaces[agent]
