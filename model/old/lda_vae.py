#%%
from torch import nn
import torch
from torch.nn import functional as F 
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import tqdm
from itertools import combinations

class InferenceNetwork(nn.Module):

    def __init__(self, input_size, n_topics):
        super(InferenceNetwork, self).__init__()
        

        self.input_size = input_size
        self.n_topics = n_topics
        self.activation = nn.ReLU()
        
        self.input_layer = nn.Linear(input_size, 2056)


        self.hidden1 = nn.Linear(2056,1024) 
        self.hidden2 = nn.Linear(1024,512)
        self.f_mu = nn.Linear(512, n_topics)
        self.f_mu_batchnorm = nn.BatchNorm1d(n_topics, affine=False)

        self.f_sigma = nn.Linear(512, n_topics)
        self.f_sigma_batchnorm = nn.BatchNorm1d(n_topics, affine=False)

    def forward(self, x):
        """Forward pass."""
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))

        return mu, log_sigma


class GenerativeNetwork(nn.Module):

    """AVITM Network."""

    def __init__(self, input_size, n_topics):
        super(GenerativeNetwork, self).__init__()
        

        self.input_size = input_size
        #self.beta = nn.Linear(n_topics, input_size)
        ## note: we have to use a Tensor and parameter bc softmax isnt defined for a linear layer
        self.beta = torch.Tensor(n_topics, input_size)
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)
        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)

        # dropout on theta


    
    def reparameterize(self,  mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, mu, logsigma):
        """Forward pass."""
        # batch_size x n_components
    

        # generate samples from theta
        ## theta is docs x topics 
        theta = F.softmax(
            self.reparameterize(mu, logsigma), dim=1)
        topics_per_doc = theta
        beta = F.softmax(self.beta_batchnorm(self.beta), dim=1)
        topics_per_word = beta
        predicted_word_counts = torch.matmul(theta, beta)
            # word_dist: batch_size x input_size

        return predicted_word_counts, topics_per_doc, topics_per_word

class VAE_LDA(nn.Module):
    def __init__(self, vocab_size, n_topics):
        super(VAE_LDA, self).__init__()
        self.inference_net = InferenceNetwork(vocab_size, n_topics)
        self.generative_net = GenerativeNetwork(vocab_size, n_topics)
    def forward(self, observed_word_counts):
        mu,logsigma = self.inference_net(observed_word_counts)
        predicted_word_counts, topics_per_doc, topics_per_word = self.generative_net(mu,logsigma)
        return mu,logsigma, predicted_word_counts, topics_per_doc, topics_per_word
    def get_beta(self):
        with torch.no_grad():
            return F.softmax(self.generative_net.beta, dim=1).detach().cpu().numpy()



def loss( observed_counts, predicted_counts, prior_mean, prior_variance,
            posterior_mean, posterior_variance, posterior_log_variance, num_topics):
    # KL term
    # var division term
    var_division = torch.sum(posterior_variance / prior_variance, dim=1)
    # diff means term
    diff_means = prior_mean - posterior_mean
    diff_term = torch.sum( (diff_means * diff_means) / prior_variance, dim=1)
    # logvar det division term
    logvar_det_division =prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
    # combine terms
    KL = 0.5 * (var_division + diff_term - num_topics + logvar_det_division)
    # Reconstruction term
    RL = -torch.sum(observed_counts * torch.log(predicted_counts + 1e-10), dim=1) ## ignore the  scaling terms 
    loss = KL + RL

    return loss.sum()


class scRNAseqDataset(Dataset):
    def __init__(self, mat, n_genes):
        self.data = mat[:,:n_genes]
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx,:]).type(torch.float)



def train_vae_lda(train_data, test_data, vsize, ntopics,n_epochs):
    prior_mean = torch.zeros(ntopics).to("cuda")
    prior_var = (torch.ones(ntopics)/ntopics).to("cuda")
    ld_vae = VAE_LDA(vsize, ntopics).to("cuda")
    opt = torch.optim.Adam(ld_vae.parameters(),lr = 0.0001)

    train_ds = scRNAseqDataset(train_data, vsize)
    test_ds = scRNAseqDataset(test_data, vsize)
    train_loss = []
    test_loss = []

    for e in tqdm.tqdm(range(n_epochs)):
        train_dl = DataLoader(train_ds, 2056, shuffle=True)
        test_dl = DataLoader(test_ds, 2056, shuffle=True)
        epoch_training_loss_l = []
        epoch_test_loss_l = []
        for x in train_dl:
            x=x.to("cuda")
            ld_vae.zero_grad()
            mu,logsigma, predicted_word_props,_,_ = ld_vae(x)
            sigma = torch.exp(logsigma)
            c_loss = loss(x, predicted_word_props, prior_mean, prior_var, mu, sigma, logsigma, ntopics)
            c_loss.backward()
            opt.step()
            epoch_training_loss_l.append(c_loss.item())
        with torch.no_grad():
            for x in test_dl:
                x=x.to("cuda")
                ld_vae.zero_grad()
                mu,logsigma, predicted_word_props,_,_ = ld_vae(x)
                sigma = torch.exp(logsigma)
                c_loss = loss(x, predicted_word_props, prior_mean, prior_var, mu, sigma, logsigma, ntopics)
                epoch_test_loss_l.append(c_loss.item())
        train_loss.append(sum(epoch_training_loss_l)/len(epoch_training_loss_l))
        test_loss.append(sum(epoch_test_loss_l)/len(epoch_test_loss_l))


    return ld_vae.get_beta(), train_loss, test_loss

# %%
'''
UCI topic coherence 

Given a set of topics:
For each topic:
    select n most probable words
    For each pairwise combo of words:
        calculate the PMI using heldout data
        add PMI to PMI list 
'''


def UCI_coherence(beta_mat, heldout_data,n=10, e=.0001):
    topic_scores = np.zeros(beta_mat.shape[0])
    print("calculating scores")
    for t in tqdm.tqdm(range(topic_scores.shape[0])):
        topic = beta_mat[t,:]
        topic_probs_words = np.argpartition(topic, -n)[-n:] 
        all_pmi = []
        for partition in combinations(topic_probs_words, 2):
            l,r = partition 
            c_l_or_r = sum((((heldout_data[:,l] > 0)) | (heldout_data[:,r] > 0)))
            c_l_and_r = ((heldout_data[:,partition] > 0).sum(axis=1) > 1).sum()
            c_l = sum((heldout_data[:,l] > 0))
            c_r = sum((heldout_data[:,r] > 0))
            pmi = np.log((c_l_and_r/ c_l_or_r + e)/( (c_l/c_l_or_r)* (c_r/c_l_or_r) ))
            all_pmi.append(pmi)
        topic_scores[t] = 2/n/(n-1) * sum(all_pmi)
    return topic_scores
