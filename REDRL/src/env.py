import utils
import os
import copy


class Env():
    def __init__(self, episode_length=10, boundary_rating=4.0, max_rating=5, min_rating=0, ratingfile='MI'):

        self.episode_length = episode_length
        self.boundary_rating = boundary_rating
        self.a = 2.0 / (float(max_rating) - float(min_rating))
        self.b = - (float(max_rating) + float(min_rating)) / (float(max_rating) - float(min_rating))
        self.positive = self.a * self.boundary_rating + self.b


        env_object_path = '../data/run_time/%s_env_objects' % ratingfile
        if os.path.exists(env_object_path):
            objects = utils.pickle_load(env_object_path)
            self.train_matrix = objects['train_matrix']
            self.user_num = objects['user_num']
            self.item_num = objects['item_num']
            self.rela_num = objects['rela_num']
            self.item_list = objects['item_list']
            self.r_matrix = objects['r_matrix']

        else:
            utils.get_envobjects(ratingfile=ratingfile)
            objects= utils.pickle_load(env_object_path)
            self.train_matrix = objects['train_matrix']
            self.user_num = objects['user_num']
            self.item_num = objects['item_num']
            self.rela_num = objects['rela_num']
            self.item_list = objects['item_list']
            self.r_matrix = objects['r_matrix']


    def get_init_data(self):
         return self.user_num, self.item_num, int(self.item_num*0.8), self.rela_num

    def get_item_list(self):
         return self.item_list

    def reset(self, user_id):
        self.user_id = user_id
        self.step_count = 0
        self.history_items = set()
        self.state = []
        
        return self.state

    def step(self, item_id):
        reward = [0.0, False]
        r = self.train_matrix[self.user_id, item_id]
        reward[0] = self.a * r + self.b
        self.step_count += 1
        self.history_items.add(item_id)

        if self.step_count == self.episode_length or len(self.history_items) == self.item_num:
            reward[1] = True

        if reward[0] >= self.positive:
            self.state.append(item_id)

        curs = copy.deepcopy(self.state)
        
        return curs, reward[0], reward[1]

    def test_reset(self, user_id):
        self.test_user_id = user_id
        self.test_step_count = 0
        self.test_history_items = set()
        self.test_state = []

        return self.test_state

    def test_step(self, item_id):
        reward = [0.0, False]
        r = self.r_matrix[self.user_id, item_id]
        reward[0] = self.a * r + self.b
        self.test_step_count += 1
        self.test_history_items.add(item_id)

        if self.test_step_count == self.episode_length or len(self.test_history_items) == self.item_num:
            reward[1] = True

        if reward[0] >= self.positive:
            self.test_state.append(item_id)

        curs = copy.deepcopy(self.test_state)

        return curs, reward[0], reward[1]