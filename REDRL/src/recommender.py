import torch
import random
import numpy as np
from tqdm import tqdm
from env import Env
from dqn import DQN
import utils
from qnet import QNet
from lookup import Emb
import time
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def set_global_seeds(i):
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)

class Recommender(object):
    def __init__(self, args):

        set_global_seeds(28)

        #Env
        self.dataset = args.dataset
        self.max_rating = args.max_rating
        self.min_rating = args.min_rating
        self.boundary_rating = args.boundary_rating
        self.episode_length = args.episode_length
        self.env = Env(episode_length=self.episode_length, boundary_rating=self.boundary_rating, max_rating=self.max_rating, min_rating=self.min_rating, ratingfile=self.dataset)
        self.user_num, self.item_num, self.train_item_num, self.rela_num = self.env.get_init_data()
        self.item_list = self.env.get_item_list()
        self.boundary_userid = int(self.user_num)
        self.boundary_itemid = int(self.item_num * 0.8)

        #DQN
        self.tau = args.tau
        self.gamma = args.gamma
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay
        self.memory_size = args.memory_size
        self.topk = args.pop
        self.candi_num = args.candi_num
        self.duling = args.duling
        self.double_q = args.double_q


        #train
        self.max_training_step = args.max_step
        self.sample_times = args.sample
        self.update_times = args.update
        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.l2_norm = args.l2


        self.eval_net = QNet(candi_num=self.candi_num, emb_size=30).to(device)
        self.target_net = QNet(candi_num=self.candi_num, emb_size=30).to(device)

        self.emb = Emb(episode_length=self.episode_length)

        self.dqn = DQN(self.train_item_num, self.learning_rate, self.l2_norm, self.emb, self.eval_net, self.target_net, self.memory_size, self.eps_start, self.eps_end, self.eps_decay, self.batch_size,
                  self.gamma, self.tau, self.double_q)

        hot_items_path = '../data/run_time/'+self.dataset+'_pop%d' % self.topk
        if os.path.exists(hot_items_path):
            self.hot_items = utils.pickle_load('../data/run_time/'+self.dataset+'_pop%d' % self.topk).tolist()
        else:
            utils.popular_in_train_user(self.dataset, self.topk, self.boundary_rating)
            self.hot_items = utils.pickle_load('../data/run_time/'+self.dataset+'_pop%d' % self.topk).tolist()

        self.candi_dict = utils.pickle_load('../data/processed_data/'+self.dataset+'/i_neighbors')
        self.nega_items = utils.pickle_load('../data/processed_data/'+self.dataset+'/nega_user_items')
        self.result_file_path = '../data/result/' + time.strftime('%Y%m%d%H%M%S') + '_' + self.dataset
        self.storage = []

    def candidate(self,obs,mask):
        tmp = []
        tmp += self.hot_items
        for s in obs:
            if s in self.candi_dict:
                tmp += self.candi_dict[s]
        tmp = set(tmp)-set(mask)
        candi = random.sample(tmp, self.candi_num)

        return candi


    def test_candidate(self, itr, mask):
        tmp = list(range(self.boundary_itemid, self.item_num))
        tmp = list(set(tmp).difference(set(mask)))
        while len(tmp) < self.candi_num:
            if len(self.nega_items[itr].split(',')) >= self.candi_num - len(tmp):
                negative = random.sample(eval(self.nega_items[itr]), self.candi_num - len(tmp))
                tmp += negative
            else:
                tmp += eval(self.nega_items[itr])
                sup_negative = random.sample(self.item_list, self.candi_num - len(tmp))
                tmp += sup_negative

            return tmp

    def train(self):
        for itr in tqdm(range(self.sample_times), desc='sampling'):
            cumul_reward, done = 0, False
            user_id = random.randint(0, self.boundary_userid-1)
            cur_state = self.env.reset(user_id)
            mask = []
            candi = self.candidate(cur_state, mask)
            while not done:
                if len(cur_state) == 0:
                    action_chosen = random.choice(self.hot_items)
                else:
                    action_chosen = self.dqn.choose_action(cur_state, candi)
                new_state, r, done = self.env.step(action_chosen)
                mask.append(action_chosen)
                candi = self.candidate(new_state, mask)
                if len(cur_state) != 0:
                    self.dqn.memory.push(cur_state, action_chosen, r, new_state, candi)
                cur_state = new_state

        for itr in tqdm(range(self.update_times),desc='updating'):
            self.dqn.learn()

    def evaluate(self):
        tp_list = []
        for itr in tqdm(range(self.user_num), desc='evaluate'):
            cumul_reward, done = 0, False
            cur_state = self.env.test_reset(itr)
            step = 0
            mask = []
            while not done:
                cur_candi = self.test_candidate(itr,mask)
                if len(cur_state) == 0:
                    action_chosen = random.choice(self.hot_items)
                else:
                    action_chosen = self.dqn.choose_action(cur_state, cur_candi, is_test=True)
                new_state, r, done = self.env.test_step(action_chosen)
                cur_state = new_state
                cumul_reward += r
                step += 1
                mask.append(action_chosen)

            tp = float(len(cur_state))
            tp_list.append(tp)
    
        precision = np.array(tp_list)/self.episode_length
        recall = np.array(tp_list)/(self.rela_num + 1e-20)

        test_ave_precision = np.mean(precision[:self.user_num])
        test_ave_recall = np.mean(recall[:self.user_num])
    
        self.storage.append([test_ave_precision, test_ave_recall])
        utils.pickle_save(self.storage, self.result_file_path)

        print('\t test precision@%d: %.4f, test recall@%d: %.4f' % (self.episode_length, test_ave_precision, self.episode_length, test_ave_recall))


    def run(self):
        for i in range(0, self.max_training_step):
            self.train()
            self.evaluate()



