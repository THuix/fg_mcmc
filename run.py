import argparse
from src.MCMC import  MALATS, FGMALATS, FGLMCTS, LMCTS
from src.baseline import LinTS, LinUCB, Random
from src.game import GameToy, GameYahoo
import torch
from tqdm import tqdm
import pickle as pkl
import json
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--config_path')

def format_agent(info):
    if info['agent'] == 'LMC':
        return LMCTS
    elif info['agent'] == 'FGLMC':
        return FGLMCTS
    elif info['agent'] == 'MALA':
        return MALATS
    elif info['agent'] == 'FGMALA':
        return FGMALATS
    elif info['agent'] == 'LinUCB':
        return LinUCB
    elif info['agent'] == 'LinTS':
        return LinTS
    elif info['agent'] == 'Random':
        return Random
    else:
        raise ValueError(info['agent'])

def load_config_file(config_path):
    f = open(config_path)
    info = json.load(f)
    info['agent'] = format_agent(info)
    if info['task_type'] == 'yahoo':
        info['phi'] = lambda x, y: x
        info['phi_a'] = lambda x, y, z: x[y, :]
        info['game'] = GameYahoo
    elif info['task_type'] == 'linear':
        info['phi'] = lambda x, nb_arms: torch.block_diag(*[x]*nb_arms)
        info['phi_a'] = lambda x, a, nb_arms: torch.block_diag(*[x]*nb_arms)[a, :]
        info['game'] = GameToy
    elif info['task_type'] == 'logistic':
        info['phi'] = lambda x, y: x
        info['phi_a'] = lambda x, a, nb_arms: x[a, :]
        info['game'] = GameToy
    else:
        raise ValueError(info['task_type'])
    return info

def run(config_path):
    info = load_config_file(config_path)
    wandb.init(name=info['project_name'], config=info)
    info['game'](info).run()
    wandb.finish()

if __name__ == '__main__':
    args = parser.parse_args()
    run(args.config_path)