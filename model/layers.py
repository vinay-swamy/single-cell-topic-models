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
        self.beta_do = nn.Dropout(.1)
        self.theta_do = nn.Dropout(.1)
        self.beta_batchnorm = nn.BatchNorm1d(vocab_size)
    def forward(self, theta):
        theta = self.theta_do(theta)
        theta = F.softmax(theta, dim = -1)
        
        _beta = F.relu(self.beta)
        _beta = self.beta_batchnorm(_beta)
        if self.normalize_beta:
            _beta = F.softmax(_beta, dim=-1)
        else:
            _beta = self.beta
        x_hat = torch.matmul(theta, _beta)
        ## enforce positivity of xhat bc this is a probability distribution 
        return x_hat, theta 
    def get_beta(self):
        training = False
        if self.training:
            training=True
            self.eval()
        with torch.no_grad():
            beta = self.beta.clone()
            beta = F.relu(beta)
        if self.normalize_beta:
            beta= F.softmax(beta, dim=-1)
        if training:
            self.train()
        return beta 

class DecoderETM(nn.Module):
    def __init__(self,vocab_size,n_topics,normalize_beta, rho_path):
        ## potentially, we could make alpha a neural net 
        super(DecoderETM, self).__init__()
        rho = torch.load(rho_path)
        self.alpha = nn.Parameter(torch.Tensor(n_topics, rho.shape[1]))
        nn.init.xavier_uniform(self.alpha)
        self.register_buffer("rho", rho.T)
        self.normalize_beta = normalize_beta
        self.theta_do = nn.Dropout(.1)
        self.beta_batchnorm = nn.BatchNorm1d(vocab_size)
    def forward(self, theta):
        theta = self.theta_do(theta)
        theta = F.softmax(theta, dim = -1)

        _beta = torch.matmul(self.alpha, self.rho)
        _beta = F.relu(_beta)
        _beta = self.beta_batchnorm(_beta)
        if self.normalize_beta:
            _beta= F.softmax(_beta, dim=-1)
        x_hat = torch.matmul(theta, _beta)
        ## enforce positivity bec we are trying to produce a probability distribution 
        return x_hat, theta
    def get_beta(self):
        training = False
        if self.training:
            training=True
            self.eval()
        with torch.no_grad():
            beta = torch.matmul(self.alpha, self.rho)
            beta = F.relu(beta)
        if self.normalize_beta:
            beta= F.softmax(beta, dim=-1)
        if training:
            self.train()
        return beta 

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
        # log_qzx = q.log_prob(z)
        # log_pz = p.log_prob(z)

        # kl
        kl = torch.distributions.kl_divergence(q,p).sum(-1)
        return kl
    # def KL(self, z, x):
    #     mu, logvar = x
    #     kl_theta = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    #     return kl_theta

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


