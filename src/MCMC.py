import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import nn


class LMCTS(object):
    '''
    Langevin Monte Carlo Thompson Sampling bandit
    - info['d']: parameter dimension
    - info['std_prior']: standard deviation of the gaussian prior distribution
    - info['eta']: inverse of temperature, controls the variance of the posterior distribution
    - info['step_size']: step size used for the Langevin update
    - info['K']: number of gradient iterations
    - info['K_not_updated']: number of gradient iterations when the posterior has not been updated
    - info['nb_arms']: number of arms
    - info['phi']: function (context x number of arms) -> feature vectors
    - info['phi_a']: function (context x arm x number of arms) -> feature vector of the corresponding arm 
    '''
    def __init__(self, info):
        self.info = info
        self.v = torch.tensor([])
        self.r = torch.tensor([])
        self.theta = nn.Parameter(torch.normal(0, 1, size=(self.info['d'], 1)))
        self.V = torch.empty(0, 1, self.info['d'])
        self.idx = 1
        self.is_posterior_updated = True

    def loss_fct(self, theta):
        loss = self.info['eta'] * ((self.v @ theta - self.r)**2).sum()
        loss += self.info['std_prior'] * torch.norm(theta)**2
        return loss

    def train(self):
        if self.theta.grad is not None:
            self.theta.grad.zero_()
        loss = self.loss_fct(self.theta)
        loss.backward()
        self.theta.data += - self.lr * self.theta.grad + np.sqrt(2 * self.lr) * torch.randn_like(self.theta)

    def sample_posterior(self, arm_idx):
        self.lr = self.info['step_size'] / self.idx
        nb_iter = self.info['K'] if self.is_posterior_updated else self.info['K_not_updated']
        for _ in range(nb_iter):
            if self.idx == 1:
                return self.theta
            self.train()
        self.is_posterior_updated = False
        return self.theta

    def choose_arm(self, feature, arm_idx):
        theta = self.sample_posterior(arm_idx)
        rewards = self.info['phi'](feature, self.info['nb_arms']) @ theta
        return rewards.argmax()
    
    def update(self, action, reward, features, arm_idx):
        v = self.info['phi'](features, self.info['nb_arms'])
        self.v = torch.cat((self.v, v[action, :].unsqueeze(0)))
        
        dif = v.size()[0] - self.V.size()[1]
        if dif > 0:
            fill = torch.zeros((self.V.size()[0], dif, self.V.size()[2]))
            self.V = torch.cat((self.V, fill), dim=1)
        elif dif < 0:
            fill = torch.zeros((-dif, v.size()[1]))
            v = torch.cat((v, fill), dim=0)
        self.V = torch.cat((self.V, v.unsqueeze(0)))

        self.r = torch.cat((self.r, torch.tensor([reward]).unsqueeze(0)))
        self.idx += 1
        self.is_posterior_updated = True


class FGLMCTS(LMCTS):
    '''
    Feel-Good Langevin Monte Carlo Thompson Sampling bandit
    - info['lambda']: Feed good exploration term
    - info['eta']: inverse of temperature, controls the variance of the posterior distribution
    - info['std_prior']: standard deviation of the gaussian prior distribution
    - info['b']: bound for the feel good term (cf definetion of the fg term)
    '''
    def __init__(self, info):
        super(FGLMCTS, self).__init__(info)

    def get_g_star(self):
        #out = torch.stack([torch.max(v @ self.theta) for v in self.V])
        return (self.V @ self.theta).max(1).values.squeeze()
    def loss_fct(self, theta):
        loss = self.info['eta'] * ((self.v @ theta - self.r)**2).sum()
        loss -= self.info['lambda'] * torch.minimum(self.get_g_star(), torch.tensor([self.info['b']])).sum()
        loss += self.info['std_prior'] * torch.norm(theta)**2
        return loss


class MALATS(object):
    '''
    Metropolis-Adjusted Langevin Algorithm Thompson Sampling bandit:
    - info['step_size']
    - info['K']: number of gradient iterations
    - info['K_not_updated']: number of gradient iterations when the posterior has not been updated
    - info['phi']: function (context x number of arms) -> feature vectors
    - info['phi_a']: function (context x arm x number of arms) -> feature vector of the corresponding arm 
    - info['nb_arms']: number of arms
    - info['eta']: inverse of temperature, controls the variance of the posterior distribution
    - info['std_prior']: standard deviation of the gaussian prior distribution
    - info['accept_reject_step']: number of gradient descent steps before the MALA update
    '''
    def __init__(self, info):
        self.info = info
        self.theta = nn.Parameter(torch.normal(0, 1, size=(self.info['d'], 1)))
        self.v = torch.tensor([])
        self.r = torch.tensor([])
        self.V = torch.empty(0, 1, self.info['d'])
        self.is_posterior_updated = True
        self.idx = 1

    def get_potential_grad(self, theta):
        if theta.grad is not None:
                theta.grad.zero_()
        loss = self.info['eta'] * ((self.v @ theta - self.r)**2).sum()
        loss += self.info['std_prior'] * torch.norm(theta)**2
        loss.backward()
        return loss, theta.grad

    def logQ(self, x, y, grad):
        return -(torch.norm(y - x - self.lr * grad, p=2) ** 2) / (4 * self.lr)

    def gradient_descent(self):
        _, gradx = self.get_potential_grad(self.theta)
        self.theta.data = self.theta - self.lr * gradx

    def mala_step(self, theta, last_grad, last_potential):
        y = theta.detach() - self.lr * last_grad + np.sqrt(2 * self.lr) * torch.randn_like(theta)
        y.requires_grad = True
        new_potential, new_grad = self.get_potential_grad(y)
        log_ratio = - new_potential + last_potential + self.logQ(y, theta, new_grad) - self.logQ(theta, y, last_grad)
        if torch.rand(1) < torch.exp(log_ratio):
            theta = y
            last_potential = new_potential
            last_grad = new_grad
        return theta, last_potential, last_grad

    def train(self, k):
        if k < self.info['accept_reject_step']:
            self.gradient_descent()

        elif k == self.info['accept_reject_step']:
            self.last_potential, self.last_grad = self.get_potential_grad(self.theta)
            self.theta, self.last_potential, self.last_grad = self.mala_step(self.theta, self.last_grad, self.last_potential)
        else:
            self.theta, self.last_potential, self.last_grad = self.mala_step(self.theta, self.last_grad, self.last_potential)
            
    def sample_posterior(self, arm_idx):
        self.lr = self.info['step_size'] / self.idx
        nb_iter = self.info['K'] if self.is_posterior_updated else self.info['K_not_updated']
        for k in range(nb_iter):
            if self.idx == 1:
                return self.theta
            self.train(k)
        self.is_posterior_updated = False
        return self.theta

    def choose_arm(self, feature, arm_idx):
        theta = self.sample_posterior(arm_idx)
        rewards = self.info['phi'](feature, self.info['nb_arms']) @ theta
        return rewards.argmax()

    def update(self, action, reward, features, arm_idx):
        v = self.info['phi'](features, self.info['nb_arms'])
        self.v = torch.cat((self.v, v[action, :].unsqueeze(0)))

        dif = v.size()[0] - self.V.size()[1]
        if dif > 0:
            fill = torch.zeros((self.V.size()[0], dif, self.V.size()[2]))
            self.V = torch.cat((self.V, fill), dim=1)
        elif dif < 0:
            fill = torch.zeros((-dif, v.size()[1]))
            v = torch.cat((v, fill), dim=0)
        self.V = torch.cat((self.V, v.unsqueeze(0)))

        self.r = torch.cat((self.r, torch.tensor([reward]).unsqueeze(0)))
        self.idx += 1
        self.is_posterior_updated = True



class FGMALATS(MALATS):
    def __init__(self, info):
        '''
        Feel Good Metropolis Adjusted Langevin Algorithm Thompson Sampling
        - info['eta']: inverse of temperature, controls the variance of the posterior distribution
        - info['lambda']: Feed good exploration term
        - info['std_prior']: standard deviation of the gaussian prior distribution
        - info['b']: bound for the feel good term (cf definetion of the fg term)
        '''
        super(FGMALATS, self).__init__(info)

    def get_potential_grad(self, theta):
        if theta.grad is not None:
                theta.grad.zero_()
        loss = self.info['eta'] * ((self.v @ theta - self.r)**2).sum()
        loss -= self.info['lambda'] * torch.minimum(self.get_g_star(), torch.tensor([self.info['b']])).sum()
        loss += self.info['std_prior'] * torch.norm(theta)**2
        loss.backward()
        return loss, theta.grad

    def get_g_star(self):
        return (self.V @ self.theta).max(1).values.squeeze()

