#SIUSIAK
import sys
import torch
from io_utils import setup_neptune
from methods.hypernets.hypernet_kernel import HyperShot

def experiment(model_path):
    neptune_run = setup_neptune()

    model = HyperShot().cuda()
    tmp = torch.load(model_path)

    model.load_state_dict(tmp['state'])
    print(model.S)

    #Parameters: N, M, model, dataset
    #1. Load Model
    #2. Load dataset [(S, Q)]
    #3. For N (S, Q) pairs
    #   - Select ith (S, Q) pair
    #   - Select another one with disjoint support (S', Q')
    #   - Eval model M times on (S, Q') and generate histogram tagged as ith

if __name__ == '__main__':

    model_path = sys.argv[1]
    experiment(model_path)