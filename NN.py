import numpy as np 

class Neural_Network(object):
    def __init__(self):
        inputLayerSize = 2
        hiddenLayerSize = 3
        outputLayerSize = 1

        self.learningRate = 0.7
        self.momentum = 0.3

        self.dW2 = 0
        self.dW1 = 0

        self.w1 = np.random.randn(inputLayerSize, hiddenLayerSize)
        self.w2 = np.random.randn(hiddenLayerSize, outputLayerSize)

    def backprop(self, x, y):
        a3 = self.forward(x)

        delta3 = np.multiply(self.error(y, a3), self.sigmoidPrime(self.z3))
        dEdW2  = np.dot(self.a2.T, delta3)

        delta2 = np.multiply(self.sigmoidPrime(self.z2), np.dot(delta3, self.w2.T))
        dEdW1  = np.dot(x.T, delta2)

        self.w2 += dEdW2
        self.w1 += dEdW1

    def forward(self, x):
        self.z2 = np.dot(x, self.w1)
        self.a2 = self.sigmoid(self.z2)

        self.z3 = np.dot(self.a2, self.w2)
        self.a3 = self.sigmoid(self.z3)

        return self.a3

    def error(self, y, o):
        return (y - o)

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def sigmoidPrime(self, z):
        return np.exp(-z)/(np.square(1 + np.exp(-z)))

nn = Neural_Network()
x = np.matrix('0 0; 0 1; 1 0; 1 1')
y = np.matrix('0; 1; 1; 0')
for i in range(100000):
    nn.backprop(x, y)
print(nn.forward(x))