#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import itertools
from math import log
import sys

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

class LangModel:
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence(s)
        self.norm()

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        numOOV = self.get_num_oov(corpus)
#         print('numOOV: ',numOOV)
        return pow(2.0, self.entropy(corpus, numOOV))

    def get_num_oov(self, corpus):
        vocab_set = set(self.vocab())  # phrase(word) set
        words_set = set(itertools.chain(*corpus)) # words set
        numOOV = len(words_set - vocab_set)
        return numOOV

    def entropy(self, corpus, numOOV):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1 # for EOS
            sum_logprob += self.logprob_sentence(s, numOOV)
        return -(1.0/num_words)*(sum_logprob)

    def logprob_sentence(self, sentence, numOOV):
        p = 0.0
        for i in xrange(len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i], numOOV)            
        p += self.cond_logprob('END_OF_SENTENCE', sentence, numOOV)
        
        
        return p

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence): pass
    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self): pass
    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous, numOOV): pass
    # required, the list of words the language model suports (including EOS)
    def vocab(self): pass

class Unigram(LangModel):
    def __init__(self, unk_prob=0.0001):
        self.model = dict()
        self.lunk_prob = log(unk_prob, 2)

    def inc_word(self, w):
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')

    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.model:
            tot += self.model[word]
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) - ltot
#         print('model: ',self.model)
    def cond_logprob(self, word, previous, numOOV):
        if word in self.model:
#             print('prob',self.model[word])
            return self.model[word]
        else:
            return self.lunk_prob-log(numOOV, 2)

    def vocab(self):
        return self.model.keys()

    

class Ngram(LangModel):
    def __init__(self, ngram_size):
    
        self.phraseFrequency = dict()
        self.prefixFrequency = dict()
        self.model = dict()
        self.probs = dict()
        self.slide = ngram_size
        self.tuple_list = []
        
               

        
    def inc_word(self, w):
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0
            
            
    def inc_phrase(self, p):
        if p in self.phraseFrequency:
            self.phraseFrequency[p] += 1.0
        else:
            self.phraseFrequency[p] = 1.0
            
    def inc_prefix(self, p):
        if p in self.prefixFrequency:
            self.prefixFrequency[p] += 1.0
        else:
            self.prefixFrequency[p] = 1.0

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence):
        from collections import Counter
        ngrams_list = [] 
        prefix_list = [] 

       
        for i in range(self.slide-1):
            
            sentence.insert (0, ("START_OF_SENTECE_%d" %i)) 
            
        sentence.append('END_OF_SENTENCE')
#         print('sentence: ', sentence)
        if len(sentence) >= self.slide:
            ngrams = list(zip(*[sentence[i:] for i in range(self.slide)]))   
            prefix = list(zip(*[sentence[i:] for i in range(self.slide-1)]))
#             print()
#             print('ngrams: ',ngrams)
#             print('prefix: ',prefix)
        
            for n in ngrams:
                    self.inc_phrase(n)
            for p in prefix[:-1]:
                    self.inc_prefix(p)
        """
        dict = {'word']}
        
        """
        for w in sentence:
            if 'START_OF_SENTECE_' in w or 'END_OF_SENTENCE' in w:
                continue
#             print(w)
            self.inc_word(w)
            
#         self.inc_word('END_OF_SENTENCE')
#         self.inc_phrase('END_OF_SENTENCE')
#         self.inc_prefix('END_OF_SENTENCE')
#         print()
#         print('-'*40)
#         print('phrase frequency: ', self.phraseFrequency)
#         print()
#         print('prefix frequency: ', self.prefixFrequency)
#         print()
#         print('word frequency: ', self.model)
#         print('-'*40)
    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self):
        k= 0.3
        tot = len(self.model) * k
        for ng in self.phraseFrequency.keys():
            count = self.phraseFrequency[ng] + k
            ng_prior = ng[:-1]
            prior_count = self.prefixFrequency[ng_prior] + tot
            self.probs[ng] = log((count / prior_count),2)

#         pass

    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous, numOOV): 
        """
        p(living| she is) = Number(she is living) / Number(she is)
        
        """ 
        # abcdefg  word=g  n=3 efg previous：abcdef prev:ef
        prev = previous[len(previous)-self.slide+1:]    
        if word == 'END_OF_SENTENCE': # the prob of end_of_sentence will be calculated in the LangModel Class, so here we return 0
            return 0       
        phrase = prev.copy()
        phrase.append(word)       #efg  
        if len(phrase) !=self.slide:
#             print(phrase)
            return 0
        
#         print('Befor Enter Phrase: ',phrase,', Prefix:', prev)
        if self.slide == 1:
             if word in self.model:
                return self.model[word]
             else:
                return self.lunk_prob-log(numOOV, 2)

        if tuple(phrase) in self.phraseFrequency:
#                 print('!!')                
                prob = (0.5+self.phraseFrequency[tuple(phrase)]) / ((len(self.model.keys())*0.5 + self.prefixFrequency[tuple(prev)]))
#                 print('Enter', 'Phrase: ',self.phraseFrequency[tuple(phrase)],'Prefix: ',self.prefixFrequency[tuple(prev)],'Prob: ',prob,'\n')                
#                 return log(prob,2)
                return self.probs[tuple(phrase)]
        else:
#                 print('~~')
                prob = (1/(len(self.model.keys())))
#                 print('No enter Prob: ',prob,'\n')
#                 prob = 1/self.model[word]
                return log(prob,2)

       
    # required, the list of words the language model suports (including EOS)
    def vocab(self):
        return self.model.keys()
