import torch
import argparse
from utils_datasets import *
from deeprobust.graph.utils import *
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=1, help='gpu id')
parser.add_argument('--dataset', type=str, default='PubMed')
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--self_link', type=bool, default=False)
parser.add_argument('--balanced_syn_label', type=bool, default=False)
parser.add_argument('--reduction_rate', type=float, default=0.005)
parser.add_argument('--all_train_label', type=bool, default=True)

args = parser.parse_args()
# torch.cuda.set_device(args.gpu_id)
device='cpu'

split_data = True
if split_data==True:
    data = Datasets(args.dataset)
    data = Inductive_data(args, data)
    data = feature_propagation(data, args, device)
    if not os.path.exists('processed_data/{}'.format(args.dataset)):
        os.makedirs('processed_data/{}'.format(args.dataset))
    torch.save(data, 'processed_data/{}/data.pt'.format(args.dataset))
else:
    data = torch.load('processed_data/{}/data.pt'.format(args.dataset))
