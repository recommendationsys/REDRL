import torch
from collections import namedtuple
import torch.optim as optim
import random
import torch.nn as nn
import numpy as np
import itertools

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state','action', 'reward','next_state', 'next_candi'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0


    def push(self, *args):

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(object):
    def __init__(self, n_action, learning_rate, l2_norm, emb, eval_net, target_net, memory_size, eps_start, eps_end, eps_decay,
                 batch_size, gamma, tau=0.01,  double_q=True):

        self.eval_net = eval_net
        self.target_net = target_net
        self.emb = emb
        self.memory = ReplayMemory(memory_size)
        self.global_step = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.n_action = n_action
        self.batch_size = batch_size
        self.gamma = gamma
        self.start_learning = 10000
        self.tau = tau
        self.double_q = double_q
        self.optimizer = optim.Adam(itertools.chain(self.eval_net.parameters()), lr=learning_rate, weight_decay=l2_norm)
        self.loss_func = nn.MSELoss()
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def choose_action(self, state, candi, is_test=False):

        state_emb = self.emb.get_seq_emb(state).to(device)
        state_emb = torch.unsqueeze(state_emb, 0)
        candi_emb = self.emb.get_a_emb(torch.unsqueeze(torch.LongTensor(candi), 0)).to(device)
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.global_step * self.eps_decay)
        if is_test or random.random() > eps_threshold:
            actions_value = self.eval_net(state_emb,candi_emb)
            action = candi[actions_value.argmax().item()]
        else:
            action = random.randrange(self.n_action)
        return action

    def learn(self):
        if len(self.memory) < self.start_learning:
            return

        for target_param, param in zip(self.target_net.parameters(), self.eval_net.parameters()):
            target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

        self.global_step += 1

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))


        b_s = self.emb.get_emd(list(batch.state)).to(device)
        b_s_ = self.emb.get_emd(list(batch.next_state)).to(device)
        b_a = torch.LongTensor(np.array(batch.action).reshape(-1, 1))
        b_a_emb = self.emb.get_a_emb(b_a).to(device)
        b_r = torch.FloatTensor(np.array(batch.reward).reshape(-1, 1)).to(device)
        next_candi = torch.LongTensor(list(batch.next_candi))
        next_candi_emb = self.emb.get_a_emb(next_candi).to(device)

        q_eval = self.eval_net(b_s, b_a_emb,choose_action=False)


        if self.double_q:
            best_actions = torch.gather(input=next_candi.to(device), dim=1, index=self.eval_net(b_s_,next_candi_emb).argmax(dim=1).view(self.batch_size, 1).to(device)).cpu()
            best_actions_emb = self.emb.get_a_emb(best_actions).to(device)
            q_target = b_r + self.gamma *( self.target_net(b_s_, best_actions_emb,choose_action=False).detach())
        else:
            q_target = b_r + self.gamma*((self.target_net(b_s_, next_candi_emb).detach()).max(dim=1).view(self.batch_size, 1))

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
