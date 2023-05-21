import argparse
from recommender import Recommender

parser = argparse.ArgumentParser()

#Env
parser.add_argument('--dataset', type=str, default='MI', help='which dataset to use')
parser.add_argument('--max_rating', type=float, default=5)
parser.add_argument('--min_rating', type=float, default=0)
parser.add_argument('--boundary_rating', type=float, default=4.0)
parser.add_argument('--episode_length', type=int, default=10)

#DQN
parser.add_argument('--pop', type=int, default=300)
parser.add_argument('--candi_num', type=int, default=200)
parser.add_argument('--tau', type=float, default=0.01)
parser.add_argument('--gamma', type=float, default=0.7)
parser.add_argument('--eps_start', type=float, default=0.9)
parser.add_argument('--eps_end', type=float, default=0.1)
parser.add_argument('--eps_decay', type=float, default=0.0001)
parser.add_argument('--memory_size', type=int, default=300000)
parser.add_argument('--duling', type=bool, default=True)
parser.add_argument('--double_q', type=bool, default=True)

# train
parser.add_argument('--max_step', type=int, default=5)
parser.add_argument('--sample', type=int, default=2000, help='sample user num')
parser.add_argument('--update', type=int, default=100, help='update times')
parser.add_argument('--batch_size',type=int, default=2000)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--l2', type=float, default=1e-6)


args = parser.parse_args()

rec = Recommender(args)
rec.run()
