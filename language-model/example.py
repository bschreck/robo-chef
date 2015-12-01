from __future__ import print_function
from __future__ import division

import numpy as np
import languagemodel as lm

np.random.seed(1)  # for reproducibility

import data_preprocess as dpp

dpp.generateLanguageModelData('../data/all_recipes.p', '../data/lm_data.txt')
dpp.splitData('../data/lm_data.txt', 0.1, '../data/lm_train.txt', '../data/lm_dev.txt')

corpus_train = lm.readCorpus("../data/lm_train.txt")
corpus_dev   = lm.readCorpus("../data/lm_dev.txt")
refinement = lm.readCorpus("../data/refinement")
background = lm.readCorpus("../data/background")

# build a common index (words to integers), mapping rare words (less than 5 occurences) to index 0
# nwords = vocabulary size for the models that only see the indexes

w2index,nwords = lm.buildIndex(corpus_train+corpus_dev+refinement+background)

# find words that appear in the training set so we can deal with new words separately
count_train = np.zeros((nwords,))
for snt in corpus_train:
    for w in snt:
        count_train[w2index[w]] += 1
'''
# Bigram model as a baseline
alpha = 0.02 # add-alpha smoothing
probB           = lm.bigramLM(corpus_train, w2index, nwords,alpha)
LLB, N          = 0.0, 0
bi              = lm.ngramGen(corpus_dev, w2index, 2)
for w in bi:
    if (count_train[w[1]]>0): # for now, skip target words not seen in training
        LLB += np.log(probB[w[0], w[1]])
        N += 1
print("Bi-gram Dev LL = {0}".format(LLB / N))
'''

# Network model
print("\nNetwork model training:")
max_iters = 4

# d_vals = [5, 10, 15]
# m_vals = [30, 35, 40, 45, 50, 60, 80, 100]
# n_vals = [2,3,4]
# for m in m_vals:
#     print('m = {0}'.format(m))

n = 3    # Length of n-gram 
d = 15   # Word vector dimension
m = 50   # Hidden units
neurallm = lm.neuralLM(d, n, m, nwords)  # The network model

ngrams = lm.ngramGen(corpus_train,w2index,n)
ngrams2 = lm.ngramGen(corpus_dev,w2index,n)
# ngrams3 = lm.ngramGen(corpus_test,w2index,n)

lrate = 0.5  # Learning rate
for it in xrange(max_iters): # passes through the training data
    LL, N  = 0.0, 0 # Average log-likelihood, number of ngrams    
    for ng in ngrams:
        pr = neurallm.update(ng,lrate)
        LL += np.log(pr)
        N  += 1
    print('Train:\t{0}\tLL = {1}'.format(it, LL / N)) 

#Dev set
dLL, dN = 0.0, 0 # Average log-likelihood, number of ngrams
for ng in ngrams2:
    if (count_train[ng[-1]]>0): # for now, skip target words not seen in training
        pr = neurallm.prob(ng)
        dLL += np.log(pr)
        dN  += 1
# print('Dev:\t{0}\tLL = {1}'.format(it, LL / N))

# #Test set
# tsLL, tsN = 0.0, 0 # Average log-likelihood, number of ngrams
# for ng in ngrams3:
#     pr = neurallm.prob(ng)
#     tsLL += np.log(pr)
#     tsN  += 1

print('Train:\tLL = {0}'.format(LL / N))
print('Dev:\tLL = {0}'.format(dLL / dN))
# print('Test:\tLL = {0}'.format(tsLL / tsN))
print('\n')

'''
print('\n\n\n')

LLB, N          = 0.0, 0
bi              = lm.ngramGen(corpus_test, w2index, 2)
for w in bi:
    LLB += np.log(probB[w[0], w[1]])
    N += 1
print("Bi-gram Test LL = {0}".format(LLB / N))

print('\n')

#Test set
LL, N = 0.0, 0 # Average log-likelihood, number of ngrams
for ng in ngrams3:
    pr = neurallm.prob(ng)
    LL += np.log(pr)
    N  += 1
print('NeuralLM Test:\tLL = {0}'.format(LL / N))
'''



refinement_ngrams = lm.ngramGen(refinement,w2index,n)
background_ngrams = lm.ngramGen(background,w2index,n)

LL, N = 0.0, 0 # Average log-likelihood, number of ngrams
for ng in refinement_ngrams:
    pr = neurallm.prob(ng)
    LL += np.log(pr)
    N  += 1

print('\n\n\nScore for refinement segments:\t{0}'.format(LL/N))

LL, N = 0.0, 0 # Average log-likelihood, number of ngrams
for ng in background_ngrams:
    pr = neurallm.prob(ng)
    LL += np.log(pr)
    N  += 1

print('\nScore for background segments:\t{0}'.format(LL/N))




