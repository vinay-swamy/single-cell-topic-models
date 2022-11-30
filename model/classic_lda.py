# %%
from gensim.models import LdaModel
from scipy.io import mmread 
import numpy as np 
from sklearn.model_selection import train_test_split
from gensim.models.callbacks import PerplexityMetric, Callback
import logging
import logging
from io import StringIO
import re 

LOG_STREAM = StringIO()    
logging.basicConfig(stream=LOG_STREAM,format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)

def array_to_corpus(mat):
    lol=[]
    for row in range(mat.shape[0]):
        l=[]
        for col in range(mat.shape[1]):
            if mat[row,col] >0:
                l.append((col, mat[row,col]))
        lol.append(l)
    return lol 


def train_classic_lda(train_mat, test_mat, vsize, ntopics,n_epochs):
    pat =re.compile(r"\d+\.\d+ perplexity")
    train_corpus = array_to_corpus(train_mat[:,:vsize])
    test_corpus = array_to_corpus(test_mat[:,:vsize])

    perplexity_logger = PerplexityMetric(corpus=test_corpus)
    lda = LdaModel(corpus=train_corpus, num_topics=ntopics, passes=n_epochs, iterations=2056, callbacks=[perplexity_logger])

    raw_log =  LOG_STREAM.getvalue().split("\n")
    perp = [float(pat.findall(x)[0].split(" ")[0]) for x in raw_log if len(pat.findall(x)) > 0 ]
    topic_props = lda.get_topics()
    return topic_props, perp


