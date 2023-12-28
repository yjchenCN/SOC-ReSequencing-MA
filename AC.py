import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import env
import matplotlib.pyplot as plt


# Actor网络（RNN）
class RNN(nn.Module):
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args
        self.rnn = nn.GRU(input_size=input_shape, hidden_size=args.rnn_hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, x, hidden_state):
        x, h = self.rnn(x, hidden_state)
        q = self.fc1(x)
        return q, h


# Critic网络（QMIX）
class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args
        # 根据您的需求定义QMIX网络
        # 示例：一些全连接层
        self.fc1 = nn.Linear(args.state_shape, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, states, actions):
        inputs = torch.cat([states, actions], dim=1)
        x = torch.relu(self.fc1(inputs))
        x = torch.relu(self.fc2(x))
        q_total = self.fc3(x)
        return q_total


class QMIX:
    def __init__(self, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape

        # Actor Networks for each agent
        self.actors = [RNN(args.obs_shape, args) for _ in range(self.n_agents)]

        # Critic Network
        self.critic = QMixNet(args)

        # Optimizers
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=args.lr) for actor in self.actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr)

        # CUDA
        if args.use_cuda:
            self.critic.cuda()
            for actor in self.actors:
                actor.cuda()

    def select_action(self, obs, hidden_states, exploration=False):
        actions = []
        next_hidden_states = []
        for agent_id, actor in enumerate(self.actors):
            obs_tensor = torch.tensor(obs[agent_id], dtype=torch.float32).unsqueeze(0)
            if self.args.use_cuda:
                obs_tensor = obs_tensor.cuda()
            q_values, next_hidden_state = actor(obs_tensor, hidden_states[agent_id])
            next_hidden_states.append(next_hidden_state)

            if exploration:
                action = np.random.choice(self.n_actions)
            else:
                action = q_values.max(dim=-1)[1].cpu().numpy().item()
            actions.append(action)
        return actions, next_hidden_states

    # Training method and other methods will be added next

    def train(self, transitions, hidden_states):
        # 转换数据格式
        obs, actions, rewards, next_obs, done = zip(*transitions)

        # 转换成张量
        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32).view(-1, 1)

        if self.args.use_cuda:
            obs = obs.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            next_obs = next_obs.cuda()
            done = done.cuda()

        # 计算当前状态的Q值和下一个状态的Q值
        current_Q_values, _ = zip(
            *[actor(obs[:, agent_id, :], hidden_states[agent_id]) for agent_id, actor in enumerate(self.actors)])
        current_Q_values = torch.stack(current_Q_values, dim=1)

        # 选择动作的Q值
        current_Q_values = current_Q_values.gather(dim=2, index=actions.unsqueeze(-1)).squeeze(-1)

        # 计算目标Q值
        with torch.no_grad():
            next_actions, next_hidden_states = self.select_action(next_obs, hidden_states, exploration=False)
            next_actions = torch.tensor(next_actions, dtype=torch.long).unsqueeze(-1)
            next_Q_values, _ = zip(
                *[actor(next_obs[:, agent_id, :], next_hidden_states[agent_id]) for agent_id, actor in
                  enumerate(self.actors)])
            next_Q_values = torch.stack(next_Q_values, dim=1)
            next_Q_values = next_Q_values.max(dim=2)[0]

        # 计算总的Q值
        total_Q_values = self.critic(obs.view(-1, self.state_shape), current_Q_values.view(-1, self.n_agents))
        total_Q_values = total_Q_values.view(-1, 1)

        # 计算目标总Q值
        target_total_Q_values = self.critic(next_obs.view(-1, self.state_shape), next_Q_values.view(-1, self.n_agents))
        target_total_Q_values = rewards + self.args.gamma * (1 - done) * target_total_Q_values.view(-1, 1)

        # 计算损失并优化
        loss = torch.mean((total_Q_values - target_total_Q_values.detach()) ** 2)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        # 更新Actor网络
        for actor, optimizer in zip(self.actors, self.actor_optimizers):
            optimizer.zero_grad()
            actor_loss = -self.critic(obs.view(-1, self.state_shape), current_Q_values.view(-1, self.n_agents)).mean()
            actor_loss.backward()
            optimizer.step()

        return loss.item()

    def save_model(self, path):
        torch.save({'actors_state_dict': [actor.state_dict() for actor in self.actors],
                    'critic_state_dict': self.critic.state_dict()}, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        for actor, state_dict in zip(self.actors, checkpoint['actors_state_dict']):
            actor.load_state_dict(state_dict)
        self.critic.load_state_dict(checkpoint['critic_state_dict'])


class Args:
    # 环境参数
    n_actions = 24  # 假设有7种可能的动作
    n_agents = 4  # 您环境中的智能体数量
    state_shape = 5  # 状态空间的维度，您环境中的state维度
    obs_shape = 5  # 观测空间的维度，与状态空间维度相同
    rnn_hidden_dim = 64  # RNN隐藏层的维度
    lr = 0.01  # 学习率
    gamma = 0.95  # 折扣因子
    use_cuda = torch.cuda.is_available()  # 是否使用CUDA
    # ... 可以添加更多您需要的参数 ...


# 创建环境和QMIX实例
env = env.CustomEnvironment()
args = Args()
qmix_agent = QMIX(args)

# 训练参数
n_episodes = 1000
rewards_history = []

for episode in range(n_episodes):
    # 初始化环境和隐藏状态
    obs = env.reset()
    hidden_states = [torch.zeros(args.rnn_hidden_dim) for _ in range(args.n_agents)]
    total_reward = 0
    done = False

    while not done:
        # 选择动作并执行环境步骤
        actions, next_hidden_states = qmix_agent.select_action(obs, hidden_states, exploration=True)
        next_obs, rewards, dones, _ = env.step(actions)
        total_reward += sum(rewards.values())

        # 收集训练数据
        transitions = [(obs[agent], actions[agent], rewards[agent], next_obs[agent], dones[agent]) for agent in
                       range(args.n_agents)]
        loss = qmix_agent.train(transitions, hidden_states)

        obs = next_obs
        hidden_states = next_hidden_states
        done = all(dones.values())

    rewards_history.append(total_reward)
    print(f"Episode {episode}, Total Reward: {total_reward}, Loss: {loss}")

# 绘制奖励随迭代次数变化的折线图
plt.plot(rewards_history)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward over Episodes')
plt.show()
