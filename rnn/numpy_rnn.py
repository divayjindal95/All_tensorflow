'''
Naive impementation of Rnn using numpy
Refer wildml for futher reference
'''


import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
import itertools
import csv

# nltk.download('punkt')

class RNN:
    def __init__(self):
        self.hidden = 5
        self.visible = 10
        self.learning = 0.001

        self.U = np.random.uniform((self.hidden, self.visible))  # matrix b/w input and hidden
        self.W = np.random.uniform((self.hidden, self.hidden))  # matrix b/w hidden and hidden
        self.V = np.random.uniform((self.visible, self.hidden))  # matrix b/w output and hidden

    def softmax(self, X):
        return 1 / (1 + np.exp(-X))

    def forward_prop(self, X):

        steps = len(X)
        hidden_units = np.zeros([self.hidden, steps + 1])
        hidden_units[:, -1] = 0
        output_units = np.zeros([self.visible, steps])

        for i in xrange(steps):
            hidden_units[:, i] = np.tanh(np.matmul(self.W, hidden_units[:, i - 1]) + np.matmul(self.U, X[:, i]))
            output_units[:, i] = self.softmax(np.matmul(self.V, hidden_units[:, i]))

            '''hidden state at time t: tanh(W * s(t - 1) + U * xt)
               output at time t : sigmoid( V*st )
            '''

        return [hidden_units, output_units]

    def error(self, y_pred, y):

        steps = len(y)
        error = 0
        for i in xrange(steps):
            error += -y * np.log(y_pred)

        '''
        error at time step t : -yt*log(ot) where yt is desired output and ot is predicted
        '''

        error = np.dot(error, error)
        return error

    def back_prop(self, hidden_units, output_units, X, Y):

        steps = len(X)
        dldv = np.zeros([self.visible, self.hidden])
        dldw = np.zeros([self.hidden, self.hidden])
        dldu = np.zeros([self.hidden, self.visible])

        one_hot_y = np.zeros([self.visible, 1])
        one_hot_y[Y] = 1

        for t in xrange(steps,0,-1):
            dldv += np.multiply(one_hot_y * (output_units[:, t] - 1), hidden_units[:, t].T)

            '''
            dEt/dVt[i,j] = dEt/dOt * dOt/dZt * dZt/dV[i,j]
            Now Zit= sum over k ( V[i,k]*st[k] )
            Hence dEt[i]/dVt[i,j] = dEt[i]/dOt[i] * dOt[i]/dZt[i] * dZt[i]/dV[i,j]

            dEt[i]/dVt[i,j]=-yt[i] * (1-ot[i]) * st[j]

            Et is the error at time step t
            Vt[i,j] is the weights between hidden unit and output layer at time t between hidden state j and out put neuron i
            yt[i] is the desired output at time t of neuron i
            ot[i] is the predicted output at time t of neuron i
            st[j] is the hidden state at time t for neuron j
            '''
            bptt_steps=steps-t-1
            alpha=self.V.T.dot(one_hot_y * (output_units[:, t] - 1))*(1-hidden_units[:,t]**2)

            for t_bppt in xrange(t,max(0,t- bptt_steps),-1):

                dldw += np.multiply(alpha,hidden_units[:,t_bppt-1].T)

                dldu += np.multiply(alpha,hidden_units[:,t_bppt-1].T)


    def train(self, X, Y):
        hidden_units, output_units = self.forward_prop(X)
        cost = self.error(output_units, Y)
        self.back_prop(hidden_units, output_units, X, Y)


def data_processing(path):

    vocabulary_size = 8000
    unknown_token = "UNKNOWN_TOKEN"
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"

    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print "Reading CSV file..."
    with open(path, 'rb') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        print reader.next()
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print "Parsed %d sentences." % (len(sentences))

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print "Found %d unique words tokens." % len(word_freq.items())

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    print "Using vocabulary size %d." % vocabulary_size
    print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    print "\nExample sentence: '%s'" % sentences[0]
    print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    rnn = RNN()
    #rnn.train(X_train, y_train)


if __name__ == "__main__":

    data_processing('./reddit-comments-2015-08.csv')
