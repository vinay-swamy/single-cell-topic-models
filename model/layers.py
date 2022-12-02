import torch 
from torch import nn 
import torch.nn.functional as F
class NormalEncoderLDA(nn.Module):
    def __init__(self, vocab_size,n_topics, dropout):
        super(NormalEncoderLDA, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, int(n_topics*4)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(n_topics*4 ), int(n_topics*2) ),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.mu_net = nn.Linear(int(n_topics*2), n_topics)
        self.logsigma = nn.Linear(int(n_topics*2), n_topics)
    def forward(self, X):
        X = self.encoder(X)
        mu = self.mu_net(X)
        logsigma = self.logsigma(X)
        return mu, logsigma

class DirichletEncoderLDA(nn.Module):
    def __init__(self, vocab_size,n_topics, dropout):
        super(DirichletEncoderLDA, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, int(n_topics*4)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(n_topics*4 ), int(n_topics*2) ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(n_topics*2), n_topics)
        )
        
    def forward(self, X):
        X = self.encoder(X)
        return X 
## missing batch norm for these ones 
class DecoderLDA(nn.Module):
    def __init__(self, vocab_size,n_topics,normalize_beta):
        super(DecoderLDA, self).__init__()
        self.beta = nn.Parameter(torch.Tensor(n_topics, vocab_size))
        nn.init.xavier_uniform_(self.beta)
        self.normalize_beta = normalize_beta
    def forward(self, theta):
        
        if self.normalize_beta:
            _beta = F.softmax(self.beta, dim=-1)
        else:
            _beta = self.beta
        x_hat = torch.matmul(theta, _beta)
        return x_hat, _beta

class DecoderETM(nn.Module):
    def __init__(self,vocab_size,n_topics,normalize_beta, rho_path):
        ## potentially, we could make alpha a neural net 
        super(DecoderETM, self).__init__()
        rho = torch.load(rho_path)
        self.alpha = nn.Parameter(torch.Tensor(n_topics, rho.shape[1]))
        nn.init.xavier_uniform(self.alpha)
        self.register_buffer("rho", rho.T)
        self.normalize_beta = normalize_beta
    def forward(self, theta):
        beta = torch.matmul(self.alpha, self.rho)
        if self.normalize_beta:
            beta= F.softmax(beta, dim=-1)
        x_hat = torch.matmul(theta, beta)
        return x_hat, beta 

class LogNormalReparameterizer(nn.Module):
    def forward(self, x):
        mu, logvar = x
        std = torch.exp(0.5 * logvar)
        d = torch.distributions.LogNormal(mu, std)
        z = d.rsample()
        return z
    def KL(self, z, x):
        mu, logvar = x
        std = torch.exp(0.5 * logvar)
        p = torch.distributions.LogNormal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.LogNormal(mu, std)
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

class NormalReparameterizer(nn.Module):
    def forward(self, x):
        mu, logvar = x
        std = torch.exp(0.5 * logvar)
        d = torch.distributions.Normal(mu, std)
        z = d.rsample()
        return z
    def KL(self, z, x):
        mu, logvar = x
        std = torch.exp(0.5 * logvar)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

class DirichletReparameterizer(nn.Module):
    def forward(self, x):
        d = torch.distributions.Dirichlet(x)
        z = d.rsample()
        return z
    def KL(self, z, x):
        
        p = torch.distributions.Dirichlet(torch.ones_like(x))
        q = torch.distributions.Dirichlet(x)
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl


