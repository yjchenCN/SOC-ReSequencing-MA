import torch
import os
# from network.base_net import RNN
# from network.qmix_net import QMixNet
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import env
import matplotlib.pyplot as plt
import torch.optim as optim
from collections import deque
from env import CustomEnvironment
from tqdm import tqdm
import collections
import random


class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


# Critic of Central-V
class Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, 1)

    def forward(self, inputs):
        x = f.relu(self.fc1(inputs))
        x = f.relu(self.fc2(x))
        q = self.fc3(x)
        return q


class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args
        # 因为生成的hyper_w1需要是一个矩阵，而pytorch神经网络只能输出一个向量，
        # 所以就先输出长度为需要的 矩阵行*矩阵列 的向量，然后再转化成矩阵

        # args.n_agents是使用hyper_w1作为参数的网络的输入维度，args.qmix_hidden_dim是网络隐藏层参数个数
        # 从而经过hyper_w1得到(经验条数，args.n_agents * args.qmix_hidden_dim)的矩阵
        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim))
            # 经过hyper_w2得到(经验条数, 1)的矩阵
            self.hyper_w2 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim))
        else:
            self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)
            # 经过hyper_w2得到(经验条数, 1)的矩阵
            self.hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim * 1)

        # hyper_w1得到的(经验条数，args.qmix_hidden_dim)矩阵需要同样维度的hyper_b1
        self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
        # hyper_w2得到的(经验条数，1)的矩阵需要同样维度的hyper_b1
        self.hyper_b2 = nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(args.qmix_hidden_dim, 1)
                                      )

    def forward(self, q_values, states):  # states的shape为(episode_num, max_episode_len， state_shape)
        # 传入的q_values是三维的，shape为(episode_num, max_episode_len， n_agents)
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args.n_agents)  # (episode_num * max_episode_len, 1, n_agents) = (1920,1,5)
        states = states.reshape(-1, self.args.state_shape)  # (episode_num * max_episode_len, state_shape)

        w1 = torch.abs(self.hyper_w1(states))  # (1920, 160)
        b1 = self.hyper_b1(states)  # (1920, 32)

        w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim)  # (1920, 5, 32)
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)  # (1920, 1, 32)

        hidden = f.elu(torch.bmm(q_values, w1) + b1)  # (1920, 1, 32)

        w2 = torch.abs(self.hyper_w2(states))  # (1920, 32)
        b2 = self.hyper_b2(states)  # (1920, 1)

        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)  # (1920, 32, 1)
        b2 = b2.view(-1, 1, 1)  # (1920, 1， 1)

        q_total = torch.bmm(hidden, w2) + b2  # (1920, 1, 1)
        q_total = q_total.view(episode_num, -1, 1)  # (32, 60, 1)
        return q_total


class QMIX:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        if args.last_action:
            input_shape += (self.n_actions,)
        if args.reuse_network:
            input_shape += (self.n_agents,)

        # 神经网络
        self.eval_rnn = RNN(input_shape, args)  # 每个agent选动作的网络
        self.target_rnn = RNN(input_shape, args)
        self.eval_qmix_net = QMixNet(args)  # 把agentsQ值加起来的网络
        self.target_qmix_net = QMixNet(args)
        self.args = args
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()
        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                path_qmix = self.model_dir + '/qmix_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_qmix_net.load_state_dict(torch.load(path_qmix, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg QMIX')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        '''
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        '''
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'], batch['avail_u'], batch['avail_u_next'], \
                                                             batch['terminated']
        mask = 1 - batch["padded"].float()  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习

        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents， n_actions)
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)

        # 得到target_q
        q_targets[avail_u_next == 0.0] = - 9999999
        q_targets = q_targets.max(dim=3)[0]

        q_total_eval = self.eval_qmix_net(q_evals, s)
        q_total_target = self.target_qmix_net(q_targets, s_next)

        targets = r + self.args.gamma * q_total_target * (1 - terminated)

        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

    def _get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给obs添加上一个动作、agent编号

        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # 因为当前的obs三维的数据，每一维分别代表(episode编号，agent编号，obs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            q_eval, self.eval_hidden = self.eval_rnn(inputs,
                                                     self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            # 把q_eval维度重新变回(8, 5,n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '_qmix_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(), self.model_dir + '/' + num + '_rnn_net_params.pkl')


class MultiAgentReplayBuffer:
    def __init__(self, capacity, num_agents):
        self.capacity = capacity
        self.num_agents = num_agents
        self.buffers = {agent: collections.deque(maxlen=capacity) for agent in range(num_agents)}

    def add(self, states, actions, rewards, next_states, dones):
        for agent in range(self.num_agents):
            self.buffers[agent].append(
                (states[agent], actions[agent], rewards[agent], next_states[agent], dones[agent]))

    def sample(self, batch_size):
        samples = {agent: random.sample(self.buffers[agent], batch_size) for agent in range(self.num_agents)}
        states = {agent: [] for agent in range(self.num_agents)}
        actions, rewards, next_states, dones = [], [], [], []

        for agent in range(self.num_agents):
            agent_samples = samples[agent]
            for sample in agent_samples:
                state, action, reward, next_state, done = sample
                states[agent].append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

        return {agent: np.array(states[agent]) for agent in range(self.num_agents)}, np.array(actions), np.array(
            rewards), np.array(next_states), np.array(dones)

    def size(self):
        return min(len(self.buffers[agent]) for agent in range(self.num_agents))

    # 创建自定义环境的实例


env = CustomEnvironment()


def train_qmix(qmix_agent, env, num_episodes, replay_buffer, minimal_size, batch_size):
    returns = []  # 用于存储每个回合的回报
    max_q_value_list = []  # 用于存储每个回合的最大Q值

    for i in range(10):
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                obs = env.reset()
                done = False

                while not done:
                    # 根据QMIX算法选择动作，并与环境交互
                    actions = qmix_agent.select_actions(obs)  # 选择每个代理的动作
                    next_obs, rewards, done, _ = env.step(actions)  # 与环境交互
                    episode_return += sum(rewards)

                    # 将经验存储到重播缓冲区
                    replay_buffer.add(obs, actions, rewards, next_obs, done)

                    # 如果重播缓冲区已达到最小大小，进行训练
                    if replay_buffer.size() > minimal_size:
                        batch = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': batch[0],
                            'actions': batch[1],
                            'rewards': batch[2],
                            'next_states': batch[3],
                            'dones': batch[4]
                        }

                        # 根据QMIX算法更新代理的Q值
                        qmix_agent.update(transition_dict)

                        # 计算当前状态的 Q 值并将其添加到 max_q_value_list
                        current_q_values = qmix_agent.compute_q_values(obs)
                        max_q_value_list.append(max(current_q_values))

                    obs = next_obs  # 更新状态为下一个状态

                returns.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(returns[-10:])
                    })
                pbar.update(1)

    return returns, max_q_value_list


def plot_results(returns, q_values):
    # 用于绘制回报和Q值与迭代次数的图表
    # 使用matplotlib或其他绘图库创建所需的图表

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(returns)
    plt.title('回报 vs. 回合')
    plt.xlabel('回合')
    plt.ylabel('回报')

    plt.subplot(2, 1, 2)
    plt.plot(q_values)
    plt.title('Q值 vs. 回合')
    plt.xlabel('回合')
    plt.ylabel('Q值')

    plt.tight_layout()
    plt.show()


class Args:
    def __init__(self):
        # 定义 QMIX 训练参数
        self.n_actions = 24  # 动作空间大小
        self.n_agents = 4  # 代理数量
        self.state_shape = (96,)  # 状态空间形状
        self.obs_shape = (48,)  # 观测空间形状
        self.last_action = True  # 是否考虑上一个动作
        self.reuse_network = True  # 是否共享神经网络
        self.cuda = True  # 是否使用 GPU
        self.load_model = False  # 是否加载预训练模型
        self.model_dir = 'models'  # 模型保存路径
        self.lr = 0.001  # 学习率
        self.optimizer = 'RMS'  # 优化器选择
        self.gamma = 0.99  # 折扣因子
        self.rnn_hidden_dim = 64  # RNN 隐藏层维度
        self.grad_norm_clip = 10.0  # 梯度裁剪阈值
        self.target_update_cycle = 10  # 目标网络更新周期
        self.save_cycle = 1000  # 模型保存周期


# 创建 args 对象并初始化参数
args = Args()

# 创建环境实例
env = CustomEnvironment()

# 定义训练参数
num_episodes = 1000
batch_size = 128
minimal_size = 1000

# 创建 QMIX agent
qmix_agent = QMIX(args)

# 创建重播缓冲区
replay_buffer = MultiAgentReplayBuffer(capacity=10000, num_agents=args.n_agents)

# 调用 train_qmix 函数进行训练
returns, max_q_value_list = train_qmix(qmix_agent, env, num_episodes, replay_buffer, minimal_size, batch_size)

# 绘制结果图表
plot_results(returns, max_q_value_list)