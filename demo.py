import numpy as np
from gym import spaces
from pettingzoo import ParallelEnv
import itertools

class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self):
        super().__init__()
        self.num_agents = 4
        self.N = 4
        self.M = 7
        self.forms = list(itertools.permutations(range(self.N, 0, -1)))
        self.formaction = np.array(self.forms).T
        self.numAct = self.formaction.shape[1]
        self.Delta = np.array([[0.1302, 0.1334, 0.2522, 0.1868, 0.0787],
                               [0.1224, 0.1255, 0.2372, 0.1756, 0.0741],
                               [0.1170, 0.1199, 0.2266, 0.1678, 0.0708],
                               [0.1170, 0.1199, 0.2266, 0.1678, 0.0708]])

        self.action_spaces = {agent: spaces.Discrete(self.numAct) for agent in range(self.num_agents)}
        self.observation_spaces = {agent: spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0], dtype=np.float32),
                                                     high=np.array([1.0, 1.0, 1.0, 1.0, 7], dtype=np.float32), shape=(5,))
                                   for agent in range(self.num_agents)}
        self.state = np.array([1, 1, 1, 1, 7])
        self.agent_rewards = {agent: 0 for agent in range(self.num_agents)}

    def reset(self, seed=None, options=None):
        self.state = np.array([1, 1, 1, 1, 7])
        self.agent_rewards = {agent: 0 for agent in range(self.num_agents)}
        return {agent: self.state for agent in range(self.num_agents)}

    # 未到达终点前SOC的计算函数
    def clcSM(self, SOC, col):
        SM = SOC.copy()
        SC = np.zeros(4)
        
        # [此处假设form和Delta已经根据当前智能体设置好]
        # [form和Delta变量需要在step函数中对每个agent进行设置]

        for indEV in range(self.N):
            X = SM[indEV]
            indices = np.where(self.form == indEV + 1)[0]

            if indices.size > 0 and 0 <= int(indices) < self.N and 0 <= int(col) - 1 < len(self.Delta[0]):
                Y = self.Delta[int(indices)][int(col) - 1]
            elif col <= 5:
                # 如果索引越界且在前五次位置排序，按照前一次的SOC情况排序
                order_indices = np.argsort(SOC) + 1
                Y = self.Delta[:, order_indices[-1] - 1]
            else:
                # 在第五次排序后，一直使用第五次排序的 Y 值
                if self.fifth_order_vector is None:
                    # 如果第五次排序的向量尚未记录，则记录
                    sorted_indices = np.argsort(SOC) + 1
                    self.fifth_order_vector = self.Delta[:, sorted_indices[-1] - 1]

                Y = self.fifth_order_vector

            SC[indEV % 4] = X - Y[indEV % 4]
        
        return SC

    # 最后一个位置进行SOC排序
    def socOrderForm(self):
        sorted_indices = np.argsort(self.SOC) + 1
        sorted_SOC = self.SOC[sorted_indices - 1]
        return sorted_SOC
    

    def calculate_efficiency(state, new_state):
        # 计算车辆之间的位置差
        position_diff = np.abs(new_state[:4] - state[:4])

        # 计算编队紧密度指标，利用位置差的均值
        efficiency = np.mean(position_diff)

        return efficiency


    def calculate_reward(self, SOC, remRsq, col):
        '''# 如果任何车辆的SOC为零，返回负面奖励并终止游戏
        if any(SOC <= 0):
            return -10  # 惩罚值，可根据需求调整'''
        distances = np.abs(self.SOC)     
        # 计算惩罚，距离越远惩罚越大
        penalty = -np.sum(distances)

        position_changes = np.sum(np.abs(self.state[:4] - self.form))  # 使用self.form表示新状态
        position_change_penalty = -position_changes  # 负惩罚，越少位置变化越好
                
        fuel_efficiency_reward = self.calculate_efficiency(self.state, self.form)

        # 计算SOC的标准差，以鼓励SOC分布的多样性
        std_dev_reward = np.std(SOC) * 2  + penalty + position_change_penalty + fuel_efficiency_reward# 增加标准差奖励的权重

        # 计算平均SOC，以鼓励保持高SOC
        avg_SOC_reward = np.mean(SOC)

        # 为了避免个别车辆SOC太低而整体标准差高的情况，引入最低SOC惩罚
        min_SOC_penalty = -2 * min(SOC)  # 惩罚最低SOC值，可根据需求调整

        # 计算总奖励
        total_reward = std_dev_reward + avg_SOC_reward + min_SOC_penalty

        return total_reward



    def step(self, actions):
        rewards = {}
        dones = {}
        infos = {agent: {} for agent in range(self.num_agents)}

        # 计算每个智能体的动作奖励
        for agent, action in actions.items():
            form = self.formaction[:, action]
            SOC = self.state[:self.N]
            remRsq = self.state[self.N]
            col = self.M - remRsq + 1
            new_SOC = self.clcSM(SOC, col, form)
            new_state = np.concatenate((new_SOC, [remRsq - 1]))

            rewards[agent] = self.calculate_reward(new_SOC, remRsq, col, form)
            self.agent_rewards[agent] = rewards[agent]

        # 选择奖励最高的动作来更新状态
        best_agent = max(self.agent_rewards, key=self.agent_rewards.get)
        best_action = actions[best_agent]
        self.form = self.formaction[:, best_action]
        self.state[:self.N] = self.clcSM(SOC, col, self.form)
        self.state[self.N] -= 1

        # 检查终止条件
        done = any(self.state[:self.N] <= 0)
        for agent in range(self.num_agents):
            dones[agent] = done

        return {agent: self.state for agent in range(self.num_agents)}, rewards, dones, infos

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    # 其余函数（clcSM, calculate_reward, calculate_efficiency等）按照您之前的逻辑实现
