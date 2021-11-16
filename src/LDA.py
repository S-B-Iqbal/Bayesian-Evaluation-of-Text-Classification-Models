import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import torch
import math 
import torch.nn as nn
import torch.nn.functional as F 
from pyro.infer import SVI, TraceMeanField_ELBO


class Encoder(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(vocab_size,hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        # Avoid Component Collapse
        self.bnmu = nn.BatchNorm1d(num_topics, affine=False)
        self.bnlv = nn.BatchNorm1d(num_topics, affine=False)

    def forward(self, inputs):
        h = F.softplus(self.fc1(inputs))
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        # \mu and \Sigma 
        logtheta_loc = self.bnmu(self.fcmu(h))
        logtheta_logvar = self.bnlv(self.fclv(h))
        logtheta_scale = (0.5*logtheta_logvar).exp()
        return logtheta_loc, logtheta_scale

class Decoder(nn.Module):
    def __init__(self, vocab_size,num_topics, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, vocab_size, bias=False)
        self.bn = nn.BatchNorm1d(vocab_size, affine=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, inputs):
        inputs = self.drop(dropout)
        # \sigma(\beta \cdot \theta)
        return F.softmax(self.bn(self.beta(inputs)), dim=1)

class LDA(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.encoder = Encoder(vocab_size, num_topics, hidden, dropout)
        self.decoder = Decoder(vocab_size, num_topics, dropout)

    def model(self, docs):
        pyro.module("decoder", self.decoder)
        with pyro.plate("documents", docs.shape[0]):
            # Replace the Drichilet Priors
            logtheta_loc = docs.new_zeros((docs.shape[0], self.num_topics))
            logtheta_scale = docs.new_ones((docs.shape[0], self.num_topics))
            logtheta = pyro.sample(
                "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))
            theta = F.softmax(logtheta, -1)

            # p(w_n | \beta, \theta)
            count_param = self.decoder(theta)

            total_count = int(docs.sum(-1).max())
            pyro.sample(
                'obs', dist.Multinomial(total_count, count_param)
            )
    def guide(self, model):
        pyro.module('encoder', self.encoder)
        with pyro.plate("documents", docs.shape[0]):
            logtheta_loc, logtheta_scale = self.encoder(docs)
            logtheta = pyro.sample("logtheta", dist.Normal(logtheta_loc,logtheta_scale).to_event(1))
    
    def beta(self):
        # For \beta_k
        return self.decoder.beta.weight.cpu().detach().T
