from pettingzoo import ParallelEnv
import numpy as np
import itertools
from gym import spaces


class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self):
        # 智能体数量
        self.num_agents = 4
        # 重排序地点的数量
        self.num_locations = 5
        # 所有可能的队列排序
        self.possible_forms = list(itertools.permutations(range(1, self.num_agents + 1)))
        # 每个智能体的动作空间和观测空间
        self.action_spaces = {agent: spaces.Discrete(len(self.possible_forms)) for agent in range(self.num_agents)}
        self.observation_spaces = {agent: spaces.Box(low=0, high=1, shape=(self.num_agents + 1,), dtype=np.float32) for
                                   agent in range(self.num_agents)}
        # 电量消耗矩阵
        self.Delta = np.array([[0.1302, 0.1334, 0.2522, 0.1868, 0.0787],
                               [0.1224, 0.1255, 0.2372, 0.1756, 0.0741],
                               [0.1170, 0.1199, 0.2266, 0.1678, 0.0708],
                               [0.1170, 0.1199, 0.2266, 0.1678, 0.0708]])
        self.current_location = 0
        self.cumulative_rewards = np.zeros(self.num_agents)
        self.reset()

    def reset(self):
        # 重置环境状态
        self.state = np.ones(self.num_agents + 1)
        self.state[-1] = self.num_locations  # 最后一个值表示剩余的重排序点数
        self.current_location = 0
        self.cumulative_rewards = np.zeros(self.num_agents)
        return self.state

    def step(self, actions):
        # 执行动作并更新环境状态
        # actions 是一个包含所有智能体动作的字典
        new_states = np.zeros((self.num_agents, self.num_agents))
        rewards = np.zeros(self.num_agents)

        # 计算每个智能体根据其选择的动作产生的新状态
        for agent, action in actions.items():
            form = self.possible_forms[action]
            new_soc = self.state[agent] - self.Delta[agent, self.current_location]
            new_states[agent] = new_soc

        # 计算每个智能体的奖励
        for agent in range(self.num_agents):
            # 个人SOC保持大的奖励
            individual_reward = new_states[agent, agent]
            # 整体SOC标准差小的奖励
            std_dev_reward = -np.std(new_states[agent])
            rewards[agent] = individual_reward + std_dev_reward

        # 选择奖励最高的动作来更新状态
        best_action = np.argmax(rewards)
        best_form = self.possible_forms[actions[best_action]]

        # 更新SOC值为选择的最优排列对应的SOC
        for agent in range(self.num_agents):
            self.state[agent] = new_states[best_action][best_form.index(agent + 1)]

        self.state[-1] -= 1
        self.cumulative_rewards += rewards

        # 判断是否到达最后一个重排序点
        done = self.state[-1] == 0
        if done:
            # 计算最终的奖励
            std_dev = np.std(self.state[:self.num_agents])
            final_reward = -std_dev  # 标准差越小，惩罚越小，所以取负值

            # 计算队列变化的相似度惩罚
            similarity_penalty = self.calculate_similarity_penalty(self.previous_form, best_form)

            # 计算队列变化次数的惩罚
            num_changes = self.calculate_num_changes(best_form)
            change_penalty = -num_changes

            # 计算电量向量与Delta最后一列相除后的最小值的惩罚
            efficiency_penalty = self.calculate_efficiency_penalty(self.state[:self.num_agents], self.Delta[:, -1])

            # 将所有惩罚和标准差奖励相加
            self.cumulative_rewards += final_reward + similarity_penalty + change_penalty + efficiency_penalty

        infos = {}
        return self.state, self.cumulative_rewards, done, infos

    def calculate_similarity_penalty(self, prev_form, new_form):
        # 计算队列变化的相似度惩罚
        similarity = sum(p == n for p, n in zip(prev_form, new_form))
        penalty = -similarity if similarity > 2 else 0  # 如果相似度超过2，则给予惩罚
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

    def calculate_similarity_penalty(self, prev_form, new_form):
        # 计算队列变化的相似度惩罚
        similarity = sum(p == n for p, n in zip(prev_form, new_form))
        penalty = -similarity if similarity > 2 else 0  # 如果相似度超过2，则给予惩罚
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
