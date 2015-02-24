# Back-Propagation Neural Networks
# 
# Yang Liu <gloolar@gmail.com>


import math
import numpy as np
import scipy as sp


np.random.seed(0)


# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return sp.tanh(x)


# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2


class NNet:

    def __init__(self, layers, r=2.0):
        self.layers = layers
        self.nlayers = len(layers) - 1

        # activations for nodes
        self.X = []
        # neurons with +1 bias (the last)
        for d in layers:
            self.X.append(np.ones((d+1,1)))
        
        # weights
        self.W = []
        self.C = []
        for l in range(self.nlayers):
            self.W.append((np.random.rand(layers[l]+1, layers[l+1])-0.5)*2*r) # [-r,r)
            # last change in weights for momentum
            self.C.append(np.zeros((layers[l]+1,layers[l+1])))

        self.deltas = [0.0]*self.nlayers # for layer 1->nlayers

        print self.X
        print self.W
        print self.C


    def update(self, inputs):
        if len(inputs) != self.layers[0]:
            raise ValueError, 'wrong dimension of inputs'

        # copy inputs
        self.X[0][:-1] = np.array(inputs).reshape(len(inputs), 1)

        # hidden and output activations
        for l in range(self.nlayers):
            # print l
            # print self.W[l].shape
            # print self.X[l].shape
            s = np.array(np.mat(self.W[l]).T*np.mat(self.X[l]))
            self.X[l+1][:-1] = sigmoid(s)

        return self.X[self.nlayers][:-1]


    def back_propagate(self, targets, learning_rate, M):
        if len(targets) != self.layers[-1]:
            raise ValueError, 'wrong number of target values'

        # error terms for output layer
        error = (-2) * (np.array(targets).reshape(len(targets),1) - self.X[self.nlayers][:-1]) 
        self.deltas[self.nlayers-1] = dsigmoid(self.X[-1][:-1]) * error

        # error terms for hidden layers
        for l in range(self.nlayers-1, 0, -1):
            # print l
            # print self.W[l].shape
            # print self.deltas[l].shape
            error = np.array(np.mat(self.W[l][:-1,:]) * np.mat(self.deltas[l]))
            self.deltas[l-1] = dsigmoid(self.X[l][:-1]) * error

        # update weights
        for l in range(self.nlayers):
            (d0, d1) = self.W[l].shape
            change = np.zeros((d0, d1))
            for i in range(d0):
                for j in range(d1):
                    change[i,j] = self.X[l][i] * self.deltas[l][j]
            self.W[l] = self.W[l] - learning_rate * change - M * self.C[l]
            self.C[l] = change
            # print change

        # least square error
        error = np.sum((np.array(targets).reshape(len(targets),1) - self.X[self.nlayers][:-1]) ** 2)
        return error


    def predict(self, patterns):
        targets = []
        for p in patterns:
            targets.append(self.update(p[0]).tolist())
            print p[0], '->', self.update(p[0])
        return targets


    def score(self, patterns):
        # MLS error
        error = 0.0
        for p in patterns:
            error = error + np.sum((self.update(p[0]) - np.array(p[1]).reshape(len(p[1]),1)) ** 2)
        return error/len(patterns)


    def weights(self):
        # print 'Input weights:'
        # for i in range(self.ni):
        #     print self.wi[i]
        # print
        # print 'Output weights:'
        # for j in range(self.nh):
        #     print self.wo[j]
        pass


    def train(self, patterns, iterations=1000, learning_rate=0.1, M=0.0):
        # M: momentum factor
        for i in xrange(iterations):
            error = 0.0
            for p in patterns: # shuffle?
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.back_propagate(targets, learning_rate, M)
            if i % 1000 == 0:
                pass #print 'iter: %5d, error: %-14f' % (i, error)
        print 'iter: %5d, error: %-14f' % (i+1, error)


def demo():
    # Teach network XOR function
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    # create a network with two input, two hidden, and one output nodes
    nn = NNet([2, 2, 1])
    # train it with some patterns
    nn.train(pat)
    # test it
    print "\npredict: \n"
    labels = nn.predict(pat)
    print labels
    print "\nscore:\n"
    print nn.score(pat)


if __name__ == '__main__':
    demo()
