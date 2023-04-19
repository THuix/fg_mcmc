import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import nn


class Random(object):
    '''
    Random bandit: Choose a random arm at each iteration
    '''
    def __init__(self, info):
        pass

    def choose_arm(self, features, arm_idx):
        return np.random.randint(0, len(arm_idx))

    def update(self, action, reward, features, arm_idx):
        pass

class LinUCB(object):
    '''
    Linear UCB bandit:
    - info['d']: parameter dimension
    - info['nb_arms']: number of arms
    - info['std_prior']: standard deviation of the gaussian prior
    - info['phi']: function (context x number of arms) -> all feature vectors
    - info['phi_a']: function (context x arm x number of arms) -> feature vector of the corresponding arm 
    - info['alpha']: controls ther size of the confidence interval
    '''
    def __init__(self, info): 
        self.info = info
        self.Vt_inv = torch.eye(self.info['d']) / self.info['std_prior']
        self.bt = torch.zeros((self.info['d'], 1))
        self.idx = 1

    def choose_arm(self, features, arm_idx):
        v = self.info['phi'](features, self.info['nb_arms'])
        norm = torch.sqrt((v @ self.Vt_inv @ v.T).diag())
        beta = self.info['alpha']
        p = v @ (self.Vt_inv @ self.bt).squeeze() + beta * norm
        return p.argmax()

    def update(self, action, reward, features, arm_idx):
        v = self.info['phi_a'](features, action, self.info['nb_arms']).unsqueeze(1)
        omega = self.Vt_inv @ v
        self.Vt_inv -= omega @ omega.T / (1 + torch.dot(omega.squeeze(), v.squeeze()))
        self.bt +=  reward * v
        self.idx += 1


class LinTS(object):
    '''
    Linear Thompson Sampling bandit:
    - info['d']: parameter dimension
    - info['eta']: inverse of temperature, controls the variance of the posterior distribution
    - info['nb_arms']: number of arms
    - info['std_prior']: standard deviation of the gaussian prior
    - info['phi']: function (context x number of arms) -> all feature vectors
    - info['phi_a']: function (context x arm x number of arms) -> feature vector of the corresponding arm 
    
    '''
    def __init__(self, info):
        self.info = info
        self.Vt_inv = torch.eye(self.info['d']) * self.info['eta'] / self.info['std_prior']
        self.bt = torch.zeros((self.info['d'], 1))
        self.idx = 1

    def choose_arm(self, feature, arm_idx):
        theta = self.sample_posterior(arm_idx)
        rewards = self.info['phi'](feature, self.info['nb_arms']) @ theta
        return rewards.argmax()
        
    def sample_posterior(self, arm_idx):
        theta = MultivariateNormal((self.Vt_inv @ self.bt).squeeze(), (1 / self.info['eta']) * self.Vt_inv).sample((1,)).T
        return theta

    def update(self, action, reward, features, arm_idx):
        v = self.info['phi_a'](features, action, self.info['nb_arms']).unsqueeze(1)
        self.bt +=  reward * v
        omega = self.Vt_inv @ v
        self.Vt_inv -= omega @ omega.T / (1 + torch.dot(omega.squeeze(), v.squeeze()))
        self.idx += 1
