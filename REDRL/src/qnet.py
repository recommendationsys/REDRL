import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, candi_num, emb_size, x_dim=30, state_dim=30, hidden_dim=30, layer_num=1):
        super(QNet, self).__init__()

        self.tanh = nn.Tanh()
        self.self_attn = nn.MultiheadAttention(embed_dim=state_dim, num_heads=1)
        self.candi_num = candi_num
        self.rnn = nn.GRU(x_dim,state_dim,layer_num,batch_first=True)
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        self.fc2_value = nn.Linear(hidden_dim, hidden_dim)
        self.out_value = nn.Linear(hidden_dim, 1)

        self.fc2_advantage = nn.Linear(hidden_dim+emb_size, hidden_dim)
        self.out_advantage = nn.Linear(hidden_dim,1)


    def forward(self, x,y, choose_action = True):

        out, h = self.rnn(x)
        out = self.tanh(out)
        output = self.self_attn(out, out, out)[0]
        output = torch.sum(output, dim=1, keepdim=True)
        x = F.relu(self.fc1(output))

        value = self.out_value(F.relu(self.fc2_value(x))).squeeze(dim=2)

        if choose_action:
            x = x.repeat(1,self.candi_num,1)
        state_cat_action = torch.cat((x,y),dim=2)
        advantage = self.out_advantage(F.relu(self.fc2_advantage(state_cat_action))).squeeze(dim=2)

        if choose_action:
            qsa = advantage + value - advantage.mean(dim=1, keepdim=True)
        else:
            qsa = advantage + value

        return qsa