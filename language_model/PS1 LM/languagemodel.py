import numpy as np
import re

# ----------------------------------
def readCorpus(filename):
    fp = open(filename, 'r')
    corpus = []  # list of sentences. A sentence is a lists of words.
    for line in fp:
        line = line.strip()
        # replace obvious numbers with <NUM>
        line = re.sub(r'\b\d+\b', r'<NUM>', line)
        line = re.sub(r'\b\d+.\d+\b', r'<NUM>', line)
        corpus.append(line.split(' '))
    return corpus

# ----------------------------------
def buildIndex(corpus, lowthreshold=5):

    # initial index, to be modified later
    tmpindex, indx = {}, 0
    for snt in corpus:
        for w in snt:
            if (not tmpindex.has_key(w)):
                tmpindex[w] = indx
                indx += 1

    # eval word counts 
    counts = np.zeros((indx,))
    for snt in corpus:
        for w in snt:
            counts[tmpindex[w]] += 1

    # map all the words with counts leq lowthreshold to index 0
    newindex = {}
    indx = 1  # 0 reserved for low occurence words
    for w in tmpindex.keys():
        if (counts[tmpindex[w]] <= lowthreshold):
            newindex[w] = 0
        else:
            newindex[w] = indx
            indx += 1
            
    # add start symbols ... <START-2> <START-1> to the index for use with up to 5-grams
    for j in range(1, 5):
        newindex["<START-" + str(j) + ">"] = indx
        indx += 1

    return newindex, indx

# ----------------------------------
def ngramGen(corpus, w2index, n):
    """ngram generator. n is the length of the ngram."""
    assert(n <= 5)
    ngrams = []
    start_snt = ["<START-" + str(j) + ">" for j in range(4, 0, -1)]
    for snt in corpus:  # sentences
        s = start_snt[-n + 1:] + snt
        for i in xrange(n - 1, len(s)):
            ngrams.append([w2index[w] for w in s[i - n + 1:i + 1]])
    return ngrams

# -----------------------------------
def unigramLM(corpus, w2index, nwords):
    uni  = ngramGen(corpus, w2index, 1)
    prob = np.zeros((nwords,))
    for w in uni:
        prob[w[0]] += 1
    return prob / float(np.sum(prob))

# -----------------------------------
def bigramLM(corpus, w2index, nwords, alpha=0.0):
    bi   = ngramGen(corpus, w2index, 2)
    prob = np.zeros((nwords,nwords))+alpha
    for w in bi:
        prob[w[0], w[1]] += 1.0
    for i in xrange(nwords):
        prob[i, :] /= np.sum(prob[i, :])
    return prob

# =====================================
class softmax(object):
    def __init__(self, dim, nwords):
        self.nwords = nwords    # output dim
        self.dim    = dim       # input dim       
        self.Wo     = np.zeros((self.nwords,))
        self.W      = np.random.randn(self.dim, self.nwords) / np.sqrt(self.dim)
        self.prob   = np.ones((self.nwords,)) / float(self.nwords)
        self.G2o    = 1e-12 * np.ones((self.nwords,)) # adagrad sum squared gradients for Wo
        self.G2     = 1e-12 * np.ones((self.nwords,)) # adagrad sum squared gradients for W
        
    def apply(self, x):
        z           = self.Wo + np.dot(x, self.W)
        self.prob   = np.exp(z - np.max(z))
        self.prob  /= np.sum(self.prob)
        return self.prob

    # update bias, accum wordvec gradient, return dlogP[y]/dx
    def backprop(self, x, lrate, y):
        grad       = -self.prob
        grad[y]   += 1.0  # dlogP[y]/dz
        xdelta     = np.dot(self.W, grad)  # dlogP[y]/dx
        xnorm2     = np.sum(x ** 2)
        self.G2o  += grad ** 2
        self.G2   += xnorm2 * grad ** 2
        self.Wo   += lrate * grad / np.sqrt(self.G2o)
        self.W    += lrate * np.outer(x, grad / np.sqrt(self.G2))
        return xdelta

# =====================================
class NNlayer(object):
    def __init__(self, idim, odim):
        self.idim = idim
        self.odim = odim
        self.W    = np.random.randn(self.idim, self.odim) / np.sqrt(self.idim)
        self.Wo   = np.zeros(self.odim,)
        # adaGrad sum squared gradients
        self.G2o  = 1e-12 * np.ones((self.odim,))
        self.G2   = 1e-12 * np.ones((self.odim,))
        self.f    = np.zeros((self.odim,))  # activation of output units

    def apply(self, x):
        self.f = np.tanh(self.Wo + np.dot(x, self.W))
        return self.f 

    def backprop(self, x, lrate, delta):
        grad       = (1.0 - self.f ** 2) * delta  # dJ/dz = df/dz * delta.  (dtanh/dx = 1 - tanh^2) 
        xdelta     = np.dot(self.W, grad)         # dJ/dx to be returned
        xnorm2     = np.sum(x ** 2)
        self.G2o  += grad ** 2
        self.G2   += xnorm2 * grad ** 2
        self.Wo   += lrate * grad / np.sqrt(self.G2o)
        self.W    += lrate * np.outer(x, grad / np.sqrt(self.G2))
        return xdelta

# =====================================
class neuralLM(object):
    def __init__(self, dim, ngram, hdim, nwords):
        self.dim    = dim       # word vector dimension
        self.ncond  = ngram - 1 # number of conditioning words
        self.hdim   = hdim      # number of hidden layer units
        self.nwords = nwords    # vocab size

        self.wvec    = np.random.randn(self.nwords, self.dim)  # word vectors
        self.G2      = 1e-12 * np.ones((self.nwords,))         # adaGrad sum of squares for word vectors
        self.hiddenL = NNlayer(self.ncond * self.dim, self.hdim)
        self.outputL = softmax(self.hdim, self.nwords)
        
    def prob(self, ngram):
        xgram, y = ngram[:-1], ngram[-1]
        x        = np.concatenate(self.wvec[xgram, :])
        fh       = self.hiddenL.apply(x)
        proba    = self.outputL.apply(fh)
        return proba[y]

    def update(self, ngram, lrate):
        # Propagate (i.e. feed-forward pass)
        xgram, y = ngram[:-1], ngram[-1]
        x        = np.concatenate(self.wvec[xgram, :])
        fh       = self.hiddenL.apply(x)
        pr       = self.outputL.apply(fh)
        # Backpropagate (and update layers)
        dh       = self.outputL.backprop(fh, lrate, y)
        dx       = self.hiddenL.backprop(x, lrate, dh)
        # Update word vectors
        grad     = np.reshape(dx, (self.ncond, self.dim))
        for i in xrange(self.ncond):
            self.G2[xgram[i]]      += np.sum(grad[i, :] ** 2)
            self.wvec[xgram[i], :] += lrate * grad[i, :] / np.sqrt(self.G2[xgram[i]])

        return pr[y]
            
        
